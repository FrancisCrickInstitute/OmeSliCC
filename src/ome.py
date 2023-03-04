from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.OmeSource import OmeSource

import Ice
import os
import omero.gateway
import omero.model
from uuid import uuid4
import xmltodict

from src.image_util import ensure_unsigned_type
from src.util import get_filetitle, ensure_list, filter_dict
from version import __version__


# https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome.html
OME_URI = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
OME_XSI = "http://www.w3.org/2001/XMLSchema-instance"
OME_SCHEMA_LOC = f"{OME_URI} {OME_URI}/ome.xsd"


def create_ome_metadata(source: OmeSource,
                        output_filename: str,
                        channel_output: str = '',
                        pyramid_sizes_add: list = None) -> str:

    file_name = os.path.basename(output_filename)
    file_title = get_filetitle(file_name)
    uuid = f'urn:uuid:{uuid4()}'

    ome = source.get_metadata().copy()
    ome['@xmlns'] = OME_URI
    ome['@xmlns:xsi'] = OME_XSI
    ome['@xsi:schemaLocation'] = OME_SCHEMA_LOC
    ome['@UUID'] = uuid
    ome['@Creator'] = f'OmeSliCC {__version__}'

    experimenter = ome.get('Experimenter')

    mag = source.get_mag()
    instrument = ome.get('Instrument')
    objective = ome.get('Instrument', {}).get('Objective')
    if mag != 0:
        if instrument is None:
            instrument = {'@ID': 'Instrument:0'}
        if objective is None:
            objective = {'@ID': 'Objective:0'}
            instrument['Objective'] = objective
        objective['@NominalMagnification'] = mag
        ome['Instrument'] = instrument

    # currently only supporting single image
    nimages = 1

    images = []
    for imagei in range(nimages):
        channels = []
        planes = []

        images0 = ensure_list(ome.get('Image', []))
        if len(images0) > 0:
            image0 = images0[imagei]
        else:
            image0 = {}
        pixels0 = image0.get('Pixels', {})
        planes0 = ensure_list(pixels0.get('Plane', []))

        combine_rgb = ('combine' in channel_output.lower())
        channel_info = source.channel_info
        nchannels = sum([info[1] for info in channel_info])
        if combine_rgb and len(channel_info) == 3:
            channel_info = [(channel_info[0][0], nchannels)]
        elif not combine_rgb and len(channel_info) < nchannels:
            channel_info = [(channel_info[0][0], 1) for _ in range(nchannels)]
        channels0 = ensure_list(pixels0.get('Channel', []))
        channeli = 0

        for channeli0, info in enumerate(channel_info):
            channel = channels0[channeli0].copy() if channeli0 < len(channels0) else {}
            color = channel.get('Color')
            channel['@ID'] = f'Channel:{imagei}:{channeli}'
            if info[0] != '':
                channel['@Name'] = info[0]
            channel['@SamplesPerPixel'] = info[1]
            if color is not None:
                channel['@Color'] = color
            channels.append(channel)

            for plane0 in planes0:
                plane = plane0.copy()
                plane_channel0 = plane0.get('@TheC')
                if plane_channel0 is None or int(plane_channel0) == channeli0:
                    planes.append(plane)

            channeli += 1

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

        if len(planes) > 0:
            pixels['Plane'] = planes

        if 'AcquisitionDate' in image0:
            image['AcquisitionDate'] = image0['AcquisitionDate']
        if 'Description' in image0:
            image['Description'] = image0['Description']
        # Set image refs
        if experimenter is not None:
            image['ExperimenterRef'] = {'@ID': experimenter['@ID']}
        if instrument is not None:
            image['InstrumentRef'] = {'@ID': instrument['@ID']}
        if objective is not None:
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
                       if 'resolution' not in annotation.get('@ID', '').lower()]
    # add pyramid sizes
    if pyramid_sizes_add is not None:
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
                       if 'originalmetadata' not in annotation.get('@Namespace', '').lower()]
    ome['StructuredAnnotations']['XMLAnnotation'] = xml_annotations

    return xmltodict.unparse({'OME': ome}, short_empty_elements=True, pretty=True)


def create_ome_metadata_from_omero(source: OmeSource,
                                   image_object: omero.gateway.ImageWrapper,
                                   output_filename: str,
                                   channel_output: str = '',
                                   pyramid_sizes_add: list = None) -> str:

    file_name = os.path.basename(output_filename)
    file_title = get_filetitle(file_name)
    uuid = f'urn:uuid:{uuid4()}'

    ome = {}
    ome['@xmlns'] = OME_URI
    ome['@xmlns:xsi'] = OME_XSI
    ome['@xsi:schemaLocation'] = OME_SCHEMA_LOC
    ome['@UUID'] = uuid
    ome['@Creator'] = f'OmeSliCC {__version__}'

    instrument = None
    objective = None
    instrument0 = image_object.getInstrument()
    if instrument0 is not None:
        instrument = {'@ID': 'Instrument:0'}
        objectives0 = instrument0.getObjectives()
        if objectives0 is not None:
            objective0 = objectives0[0]
            objective = {
                '@ID': 'Objective:0',
                '@Manufacturer': objective0.getManufacturer(),
                '@Model': objective0.getModel(),
                '@LotNumber': objective0.getLotNumber(),
                '@SerialNumber': objective0.getSerialNumber(),
                '@NominalMagnification': source.get_mag(),
                '@CalibratedMagnification': source.get_mag(),
                #'@Correction': objective0.getCorrection().getValue(),
                '@LensNA': objective0.getLensNA(),
                '@WorkingDistance': objective0.getWorkingDistance().getValue(),
                '@WorkingDistanceUnit': objective0.getWorkingDistance().getSymbol(),
                '@Iris': objective0.getIris(),
                '@Immersion': objective0.getImmersion().getValue()
            }
            instrument['Objective'] = objective
        ome['Instrument'] = instrument

    # currently only supporting single image
    nimages = 1

    images = []
    for imagei in range(nimages):
        channels = []
        planes = []

        pixels0 = image_object.getPrimaryPixels()

        planes0 = []
        stage0 = image_object.getStageLabel()
        if stage0 is not None:
            for plane0 in pixels0.copyPlaneInfo():
                planes0.append({
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

        combine_rgb = ('combine' in channel_output.lower())
        channelso = image_object.getChannels()
        channels0 = []
        for channelo in channelso:
            channell = channelo.getLogicalChannel()
            light_path = channell.getLightPath()
            if light_path is None:
                light_path = {}
            channel0 = {
                '@Name': channelo.getName(),
                '@Color': channelo.getColor().getInt(),
                '@Lut': channelo.getLut(),
                '@EmissionWave': channelo.getEmissionWave(),
                '@ExcitationWave': channelo.getExcitationWave(),
                '@Fluor': channell.getFluor(),
                '@Otf': channell.getOtf(),
                '@PhotometricInterpretation': channell.getPhotometricInterpretation(),
                '@ContrastMethod': channell.getContrastMethod(),
                '@PinHoleSize': channell.getPinHoleSize(),
                '@SamplesPerPixel': channell.getSamplesPerPixel(),
                'LightPath': light_path,
            }
            channels0.append(channel0)
        nchannels = sum([channel0['@SamplesPerPixel'] for channel0 in channels0])
        if combine_rgb and len(channels0) == 3:
            channels0 = channels0[0:1]
            channels0[0]['@SamplesPerPixel'] = nchannels
            channels0[0].pop('Color', None)
        elif not combine_rgb and len(channels0) < nchannels:
            channel1 = channels0[0].copy()
            channel1['@SamplesPerPixel'] = 1
            channels0 = [channel0 for channel0 in range(nchannels)]
        channeli = 0

        for channeli0, channel0 in enumerate(channels0):
            channel = channels0[channeli0] if channeli0 < len(channels0) else {}
            channel['@ID'] = f'Channel:{imagei}:{channeli}'
            channels.append(channel)

            for plane0 in planes:
                plane = plane0.copy()
                plane_channel0 = plane0.get('@TheC')
                if plane_channel0 is None or int(plane_channel0) == channeli0:
                    planes.append(plane)

            channeli += 1

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
        if instrument is not None:
            image['InstrumentRef'] = {'@ID': instrument['@ID']}
        if objective is not None:
            image['ObjectiveSettings'] = {'@ID': objective['@ID']}
        # (end image refs)
        if stage0 is not None:
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
    if pyramid_sizes_add is not None:
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

    return xmltodict.unparse({'OME': filter_dict(ome)}, short_empty_elements=True, pretty=True)


def get_omero_metadata_dict(image_object):
    # semi-automatically extract metadata dict (currently unused)
    ome = get_omero_metadata_dict_it(image_object)
    ome['Image'] = {
        '@ID': 'Image:0',
        '@Name': ome.pop('Name'),
        '@AcquisitionDate': ome.pop('AcquisitionDate'),
        'ObjectiveSettings': ome.pop('ObjectiveSettings'),
        'StageLabel': ome.pop('StageLabel'),
        'Pixels': ome.pop('Pixels')
    }
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
    ome['StructuredAnnotations'] = annotations
    return ome


def get_omero_metadata_dict_it(omero_obj, parents=[]):
    # recursively extract metadata dict (currently unused)
    metadata = {}
    if isinstance(omero_obj, omero.gateway.BlitzObjectWrapper):
        obj = omero_obj._obj
    else:
        obj = omero_obj
    for field in obj._field_info._fields:
        name = field[0].capitalize() + field[1:]
        if field != 'details' and field != 'relatedTo' \
                and not (name == 'Image' or name == 'Images' or name in parents) \
                and not ('ObjectiveSettings' in parents and name == 'Objective'):
            try:
                if hasattr(omero_obj, 'get' + name):
                    if name == 'Pixels':
                        value0 = omero_obj.getPrimaryPixels()
                    else:
                        value0 = getattr(omero_obj, 'get' + name)()
                    if value0 is not None:
                        data = []
                        for value in ensure_list(value0):
                            if isinstance(value, omero.gateway.BlitzObjectWrapper) or isinstance(value, Ice.Object):
                                if hasattr(value, 'getValue'):
                                    while hasattr(value, 'getValue'):
                                        value = value.getValue()
                                else:
                                    value = get_omero_metadata_dict_it(value, parents + [name])
                                data.append(value)
                            else:
                                data.append(value)
                        if len(data) > 0:
                            if len(data) == 1:
                                data = data[0]
                            metadata[name] = data
            except:
                pass
    return metadata
