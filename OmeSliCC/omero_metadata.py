from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from OmeSource import OmeSource

import os
import omero.gateway
import omero.model
import toml
from uuid import uuid4

from OmeSliCC.color_conversion import rgba_to_int
from OmeSliCC.image_util import *
from OmeSliCC.util import *
from OmeSliCC.XmlDict import dict2xml, XmlDict


# https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome.html
OME_URI = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
OME_XSI = "http://www.w3.org/2001/XMLSchema-instance"
OME_SCHEMA_LOC = f"{OME_URI} {OME_URI}/ome.xsd"


def create_ome_metadata_from_omero(source: OmeSource,
                                   image_object: omero.gateway.ImageWrapper,
                                   output_filename: str,
                                   combine_rgb: bool = True,
                                   pyramid_sizes_add: list = None) -> str:

    file_name = os.path.basename(output_filename)
    file_title = get_filetitle(file_name)
    uuid = f'urn:uuid:{uuid4()}'
    software_name = toml.load("pyproject.toml")["project"]["name"]
    software_version = toml.load("pyproject.toml")["project"]["version"]

    ome = XmlDict()
    ome['@xmlns'] = OME_URI
    ome['@xmlns:xsi'] = OME_XSI
    ome['@xsi:schemaLocation'] = OME_SCHEMA_LOC
    ome['@UUID'] = uuid
    ome['@Creator'] = f'{software_name} {software_version}'

    instrument = None
    objective = None
    instrument0 = image_object.getInstrument()
    if instrument0 is not None:
        instrument = {'@ID': 'Instrument:0'}
        objectives0 = instrument0.getObjectives()
        if objectives0 is not None:
            objective0 = objectives0[0]
            wd = objective0.getWorkingDistance()
            if wd is not None:
                working_distance = wd.getValue()
                working_distance_unit = wd.getSymbol()
            else:
                working_distance = None
                working_distance_unit = None
            objective = {
                '@ID': 'Objective:0',
                '@Manufacturer': objective0.getManufacturer(),
                '@Model': objective0.getModel(),
                '@LotNumber': objective0.getLotNumber(),
                '@SerialNumber': objective0.getSerialNumber(),
                '@NominalMagnification': source.get_mag(),
                '@CalibratedMagnification': source.get_mag(),
                '@LensNA': objective0.getLensNA(),
                '@WorkingDistance': working_distance,
                '@WorkingDistanceUnit': working_distance_unit,
                '@Iris': objective0.getIris(),
            }
            correction = objective0.getCorrection().getValue()
            if correction is not None and correction.lower() not in ['', 'unknown']:
                objective['@Correction'] = correction
            immersion = objective0.getImmersion().getValue()
            if immersion is not None and immersion.lower() not in ['', 'unknown']:
                objective['@Immersion'] = immersion
            instrument['Objective'] = objective
        ome['Instrument'] = instrument

    # currently only supporting single image
    nimages = 1

    images = []
    for imagei in range(nimages):
        channels = []
        planes = []

        pixels0 = image_object.getPrimaryPixels()

        stage0 = image_object.getStageLabel()
        if stage0:
            for plane0 in pixels0.copyPlaneInfo():
                planes.append({
                    '@TheC': plane0.getTheC(),
                    '@TheT': plane0.getTheT(),
                    '@TheZ': plane0.getTheZ(),
                    'DeltaT': plane0.getDeltaT(),
                    'ExposureTime': plane0.getExposureTime(),
                    'PositionX': stage0.getPositionX().getValue(),
                    'PositionY': stage0.getPositionY().getValue(),
                    'PositionZ': stage0.getPositionZ().getValue(),
                    'PositionXUnit': stage0.getPositionX().getSymbol(),
                    'PositionYUnit': stage0.getPositionY().getSymbol(),
                    'PositionZUnit': stage0.getPositionZ().getSymbol(),
                })

        channelso = image_object.getChannels()
        for channeli, channelo in enumerate(channelso):
            channell = channelo.getLogicalChannel()
            light_path = channell.getLightPath()
            if light_path is None:
                light_path = {}
            channel = {
                '@ID': f'Channel:{imagei}:{channeli}',
                '@Name': channelo.getName(),
                '@Color': channelo.getColor().getInt(),
                '@EmissionWave': channelo.getEmissionWave(),
                '@ExcitationWave': channelo.getExcitationWave(),
                '@PockelCellSetting': channell.getPockelCellSetting(),
                '@Fluor': channell.getFluor(),
                '@ContrastMethod': channell.getContrastMethod(),
                '@PinHoleSize': channell.getPinHoleSize(),
                '@SamplesPerPixel': channell.getSamplesPerPixel(),
                '@NDFilter': channell.getNdFilter(),
                'LightPath': light_path,
            }
            channels.append(channel)

        nchannels = source.get_nchannels()
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

        image = {
            '@ID': f'Image:{imagei}',
            '@Name': file_title,
            'AcquisitionDate': image_object.getAcquisitionDate().isoformat(),
            'Description': image_object.getDescription(),
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

        if len(planes) > 0:
            pixels['Plane'] = planes

        # Set image refs
        if instrument:
            image['InstrumentRef'] = {'@ID': instrument['@ID']}
        if objective:
            image['ObjectiveSettings'] = {'@ID': objective['@ID']}
        # (end image refs)
        if stage0:
            image['StageLabel'] = {'@Name': stage0.getName()}
        image['Pixels'] = pixels
        images.append(image)

    ome['Image'] = images

    annotations = {}
    for annotations0 in image_object.listAnnotations():
        annotation_type = annotations0.OMERO_TYPE.__name__
        if annotation_type.endswith('I'):
            annotation_type = annotation_type[:-1]
        if annotation_type not in annotations:
            annotations[annotation_type] = []
        value = annotations0.getValue()
        if annotation_type == 'MapAnnotation':
            value = {'M': [{'@K': k, '#text': v} for k, v in value]}
        annotations[annotation_type].append({
            '@ID': f'Annotation:{len(annotations[annotation_type])}',
            '@Namespace': annotations0.getNs(),
            'Value': value
        })
    # add pyramid sizes
    if pyramid_sizes_add:
        if 'MapAnnotation' not in annotations:
            annotations['MapAnnotation'] = []
        key_value_map = {'M': [{'@K': i + 1, '#text': f'{" ".join([str(size) for size in pyramid_size])}'}
                               for i, pyramid_size in enumerate(pyramid_sizes_add)]}
        annotations['MapAnnotation'].insert(0, {
            '@ID': 'Annotation:Resolution:0',
            '@Namespace': 'openmicroscopy.org/PyramidResolution',
            'Value': key_value_map
        })

    ome['StructuredAnnotations'] = annotations

    return dict2xml({'OME': filter_dict(ome)})
