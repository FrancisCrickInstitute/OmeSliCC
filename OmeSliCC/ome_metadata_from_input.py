# https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome.html
# Test with xmlvalid validator tools provided as part of bftools
# OME sub-elements are required in particular order

import os.path
from uuid import uuid4

from OmeSliCC.color_conversion import *
from OmeSliCC.ome_util import *
from OmeSliCC.util import *
from OmeSliCC.XmlDict import dict2xml

OME_URI = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
OME_XSI = "http://www.w3.org/2001/XMLSchema-instance"
OME_SCHEMA_LOC = f"{OME_URI} {OME_URI}/ome.xsd"


def create_ome_metadata(metadata: dict,
                        data: np.ndarray,
                        data_dimension_order: str,
                        output_filename: str,
                        combine_rgb: bool = True,
                        pyramid_sizes_add: list = []) -> str:

    file_name = os.path.basename(output_filename)
    file_title = get_filetitle(file_name)
    uuid = f'urn:uuid:{uuid4()}'

    ome = metadata.copy()
    ome['@xmlns'] = OME_URI
    ome['@xmlns:xsi'] = OME_XSI
    ome['@xsi:schemaLocation'] = OME_SCHEMA_LOC
    ome['@UUID'] = uuid

    instrument = ome.get('Instrument')
    if instrument is not None:
        ensure_ome_id(ome, 'Instrument')
        ensure_ome_id(instrument, 'Detector')
        ensure_ome_id(instrument, 'LightSourceGroup', 'LightSource')
        ensure_ome_id(instrument, 'LightSourceGroup.Laser')
    experimenter = ome.get('Experimenter')
    ensure_ome_id(ome, 'Experimenter')
    roi = ome.get('ROI')
    ensure_ome_id(ome, 'ROI')
    annotation = ome.get('Annotation')
    ensure_ome_id(ome, 'Annotation')

    # currently only supporting single image
    nimages = 1

    for imagei in range(nimages):
        image = ensure_list(ome.get('Image', {}))[imagei]
        image['@ID'] = f'Image:{imagei}'
        image['@Name'] = file_title
        pixels = image.get('Pixels', {})

        nchannels = data.shape[-1]
        channels0 = pixels.get('channels', [{}])
        n = len(channels0)
        samples_per_pixel = nchannels // n

        channels = []
        for channeli, channel0 in enumerate(channels0):
            channel = {'@Name': f'{channeli}',
                       '@SamplesPerPixel': samples_per_pixel}
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

        xyzct = []
        for dim in 'xyzct':
            if dim in data_dimension_order:
                xyzct.append(data.shape[data_dimension_order.index(dim)])
            else:
                xyzct.append(1)

        pixels['@ID'] = f'Pixels:{imagei}'
        pixels['@SizeX'] = xyzct[0]
        pixels['@SizeY'] = xyzct[1]
        pixels['@SizeZ'] = xyzct[2]
        pixels['@SizeC'] = xyzct[3]
        pixels['@SizeT'] = xyzct[4]
        pixels['@Type'] = str(data.dtype)
        pixels['@DimensionOrder'] = 'XYZCT' #data_dimension_order.upper()
        pixels['Channel'] = channels
        pixels['TiffData'] = {'UUID': {'@FileName': file_name, '#text': uuid}}

        # Set image refs
        if experimenter is not None:
            image['ExperimenterRef'] = {'@ID': experimenter['@ID']}
        if instrument is not None:
            image['InstrumentRef'] = {'@ID': instrument['@ID']}

        if 'Plane' in pixels:
            plane = pixels['Plane']
            if '@TheZ' not in plane:
                plane['@TheZ'] = 0
            if '@TheT' not in plane:
                plane['@TheT'] = 0
            if '@TheC' not in plane:
                plane['@TheC'] = 0
            pixels['Plane'] = pixels.pop('Plane')   # reorder
        image['Pixels'] = image.pop('Pixels')   # reorder

        if roi is not None:
            image['ROIRef'] = {'@ID': roi['@ID']}
        if annotation is not None:
            image['AnnotationRef'] = {'@ID': annotation['@ID']}

    ome['Image'] = ome.pop('Image')  # put Image at the end, before annotations

    # filter source pyramid sizes
    map_annotations0 = ensure_list(ome.get('StructuredAnnotations', {}).get('MapAnnotation', []))
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

    if map_annotations:
        if 'StructuredAnnotations' not in ome:
            ome['StructuredAnnotations'] = {}
        ome['StructuredAnnotations']['MapAnnotation'] = map_annotations

    # filter original metadata elements
    xml_annotations0 = ensure_list(ome.get('StructuredAnnotations', {}).get('XMLAnnotation', []))
    xml_annotations = [annotation for annotation in xml_annotations0
                       if 'originalmetadata' not in annotation.get('Namespace', '').lower()]

    if xml_annotations:
        if 'StructuredAnnotations' not in ome:
            ome['StructuredAnnotations'] = {}
        ome['StructuredAnnotations']['XMLAnnotation'] = xml_annotations

    return dict2xml({'OME': ome})
