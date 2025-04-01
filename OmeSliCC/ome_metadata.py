from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from OmeSource import OmeSource

import os
import toml
from uuid import uuid4

from OmeSliCC.color_conversion import *
from OmeSliCC.image_util import *
from OmeSliCC.util import *
from OmeSliCC.XmlDict import dict2xml, XmlDict


# https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome.html
OME_URI = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
OME_XSI = "http://www.w3.org/2001/XMLSchema-instance"
OME_SCHEMA_LOC = f"{OME_URI} {OME_URI}/ome.xsd"


def create_ome_metadata(source: OmeSource,
                        output_filename: str,
                        combine_rgb: bool = True,
                        pyramid_sizes_add: list = []) -> str:

    file_name = os.path.basename(output_filename)
    file_title = get_filetitle(file_name)
    uuid = f'urn:uuid:{uuid4()}'
    software_name = toml.load("pyproject.toml")["project"]["name"]
    software_version = toml.load("pyproject.toml")["project"]["version"]

    ome = source.get_metadata().copy() if source.has_ome_metadata else XmlDict()
    ome['@xmlns'] = OME_URI
    ome['@xmlns:xsi'] = OME_XSI
    ome['@xsi:schemaLocation'] = OME_SCHEMA_LOC
    ome['@UUID'] = uuid
    ome['@Creator'] = f'{software_name} {software_version}'

    experimenter = ome.get('Experimenter')
    instrument = ome.get('Instrument')
    objective = ome.get('Instrument', {}).get('Objective')

    # currently only supporting single image
    nimages = 1

    images = []
    for imagei in range(nimages):
        images0 = ensure_list(ome.get('Image', []))
        if len(images0) > 0:
            image0 = images0[imagei]
        else:
            image0 = {}
        pixels0 = image0.get('Pixels', {})

        nchannels = source.get_nchannels()
        channels0 = source.get_channels()
        n = len(channels0)
        samples_per_pixel = nchannels // n

        channels = []
        for channeli, channel0 in enumerate(channels0):
            channel = XmlDict({'@Name': channel0.get('label', ''),
                               '@SamplesPerPixel': samples_per_pixel})
            color = channel0.get('color')
            if color:
                channel['@Color'] = rgba_to_int(color)
            channels.append(channel)

        if combine_rgb and len(channels) == 3:
            channel = channels[0].copy()
            channel['@SamplesPerPixel'] = nchannels
            channel.pop('@Color', None)
            channels = [channel]
        elif not combine_rgb and len(channels) < nchannels:
            channel = channels[0].copy()
            if '@Color' in channel:
                channel['@Color'] = rgba_to_int(channel['@Color'])
            channel['@SamplesPerPixel'] = 1
            channels = [channel] * nchannels

        for channeli, channel in enumerate(channels):
            channel['@ID'] = f'Channel:{imagei}:{channeli}'

        image = {
            '@ID': f'Image:{imagei}',
            '@Name': file_title,
        }

        xyzct = source.get_size_xyzct()
        pixels = {
            '@ID': f'Pixels:{imagei}',
            '@SizeX': xyzct[0],
            '@SizeY': xyzct[1],
            '@SizeZ': xyzct[2],
            '@SizeC': nchannels,
            '@SizeT': xyzct[4],
            '@Type': str(ensure_unsigned_type(source.get_pixel_type())),
            '@DimensionOrder': 'XYZCT',
            'Channel': channels,
            'TiffData': {'UUID': {'@FileName': file_name, '#text': uuid}},
        }

        pixel_size = source.get_pixel_size()
        if len(pixel_size) > 0 and pixel_size[0][0] != 0:
            pixels['@PhysicalSizeX'] = pixel_size[0][0]
        if len(pixel_size) > 1 and pixel_size[1][0] != 0:
            pixels['@PhysicalSizeY'] = pixel_size[1][0]
        if len(pixel_size) > 2 and pixel_size[2][0] != 0:
            pixels['@PhysicalSizeZ'] = pixel_size[2][0]
        if len(pixel_size) > 0 and pixel_size[0][1] != '':
            pixels['@PhysicalSizeXUnit'] = pixel_size[0][1]
        if len(pixel_size) > 1 and pixel_size[1][1] != '':
            pixels['@PhysicalSizeYUnit'] = pixel_size[1][1]
        if len(pixel_size) > 2 and pixel_size[2][1] != '':
            pixels['@PhysicalSizeZUnit'] = pixel_size[2][1]

        # TODO: create plane metadata if not exists
        planes = ensure_list(pixels0.get('Plane', []))
        if len(planes) > 0:
            pixels['Plane'] = planes

        if 'AcquisitionDate' in image0:
            image['AcquisitionDate'] = image0['AcquisitionDate']
        if 'Description' in image0:
            image['Description'] = image0['Description']
        # Set image refs
        if experimenter:
            image['ExperimenterRef'] = {'@ID': experimenter['@ID']}
        if instrument:
            image['InstrumentRef'] = {'@ID': instrument['@ID']}
        if objective:
            image['ObjectiveSettings'] = {'@ID': objective['@ID']}
        # (end image refs)
        if 'StageLabel' in image0:
            image['StageLabel'] = image0['StageLabel']
        image['Pixels'] = pixels
        images.append(image)

    ome['Image'] = images

    if 'StructuredAnnotations' not in ome:
        ome['StructuredAnnotations'] = {}

    # filter source pyramid sizes
    map_annotations0 = ensure_list(ome['StructuredAnnotations'].get('MapAnnotation', []))
    map_annotations = [annotation for annotation in map_annotations0
                       if 'resolution' not in annotation.get('ID', '').lower()]
    # add pyramid sizes
    if pyramid_sizes_add:
        key_value_map = {'M': [{'@K': i + 1, '#text': f'{" ".join([str(size) for size in pyramid_size])}'}
                               for i, pyramid_size in enumerate(pyramid_sizes_add)]}
        map_annotations.insert(0, {
            '@ID': 'Annotation:Resolution:0',
            '@Namespace': 'openmicroscopy.org/PyramidResolution',
            'Value': key_value_map
        })
    ome['StructuredAnnotations']['MapAnnotation'] = map_annotations

    # filter original metadata elements
    xml_annotations0 = ensure_list(ome.get('StructuredAnnotations', {}).get('XMLAnnotation', []))
    xml_annotations = [annotation for annotation in xml_annotations0
                       if 'originalmetadata' not in annotation.get('Namespace', '').lower()]
    ome['StructuredAnnotations']['XMLAnnotation'] = xml_annotations

    return dict2xml({'OME': ome})
