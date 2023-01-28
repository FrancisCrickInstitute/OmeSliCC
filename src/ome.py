from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.OmeSource import OmeSource

import os
import omero.gateway
import omero.model
from uuid import uuid4
from ome_types import OME
from ome_types.model import Image, Pixels, Plane, Channel, Instrument, Objective, StageLabel, Map, MapAnnotation, \
    CommentAnnotation, InstrumentRef, AnnotationRef, TiffData
from ome_types.model.map import M
from ome_types.model.tiff_data import UUID
import xmltodict

from src.util import get_filetitle, ensure_list

OME_URI = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
OME_XSI = "http://www.w3.org/2001/XMLSchema-instance"
OME_SCHEMA_LOC = f"{OME_URI} {OME_URI}/ome.xsd"


def create_ome_metadata(source: OmeSource, output_filename: str, channel_output: str = '', pyramid_sizes_add: list = None) -> str:
    file_name = os.path.basename(output_filename)
    file_title = get_filetitle(file_name)
    uuid = f'urn:uuid:{uuid4()}'
    ome = {'@UUID': uuid, '@xmlns': OME_URI, '@xmlns:xsi': OME_XSI, '@xsi:schemaLocation': OME_SCHEMA_LOC, 'Image': []}
    source_metadata = source.get_metadata()

    # only supporting single image
    nimages = 1

    for imagei in range(nimages):
        ome_channels = []

        imetadata = ensure_list(source_metadata.get('Image', {}))[imagei]
        pmetadata = imetadata.get('Pixels', {})
        description = imetadata.get('Description', '')
        if description != '':
            description += ' '

        combine_channels = ('combine' in channel_output.lower())
        channel_info = source.channel_info
        nchannels = sum([info[1] for info in channel_info]) if not channel_output.isnumeric() else 1
        if combine_channels and len(channel_info) > 1:
            channel_info = [(channel_info[0][0], nchannels)]
        elif not combine_channels and len(channel_info) < nchannels:
            channel_info = [(channel_info[0][0], 1) for _ in range(nchannels)]

        channels = ensure_list(pmetadata.get('Channel', {}))
        for channeli, info in enumerate(channel_info):
            if not channel_output.isnumeric() or channeli == int(channel_output):
                channel = channels[channeli] if channeli < len(channels) else {}
                color = channel.get('Color')
                channel['@ID'] = f'Channel:0:{len(ome_channels)}'
                if info[0] != '':
                    channel['@Name'] = info[0]
                channel['@SamplesPerPixel'] = info[1]
                if color is not None:
                    channel['@Color'] = color
                ome_channels.append(channel)

        description += 'converted by OmeSliCC'

        xyzct = source.sizes_xyzct[0]
        pixel_size = source.pixel_size
        image = {
            '@ID': f'Image:{imagei}',
            '@Name': file_title,
            'Pixels': {
                '@ID': 'Pixels:0',
                '@SizeX': xyzct[0],
                '@SizeY': xyzct[1],
                '@SizeZ': xyzct[2],
                '@SizeC': nchannels,
                '@SizeT': xyzct[4],
                '@Type': str(source.get_pixel_type()),
                '@DimensionOrder': 'XYZCT',
                'Channel': ome_channels,
                'TiffData': {'UUID': {'@FileName': file_name, '#text': uuid}},
            },
        }
        if len(pixel_size) > 0 and pixel_size[0][0] != 0:
            image['Pixels']['@PhysicalSizeX'] = pixel_size[0][0]
        if len(pixel_size) > 1 and pixel_size[1][0] != 0:
            image['Pixels']['@PhysicalSizeY'] = pixel_size[1][0]
        if len(pixel_size) > 2 and pixel_size[2][0] != 0:
            image['Pixels']['@PhysicalSizeZ'] = pixel_size[2][0]
        if len(pixel_size) > 0 and pixel_size[0][1] != '':
            image['Pixels']['@PhysicalSizeXUnit'] = pixel_size[0][1]
        if len(pixel_size) > 1 and pixel_size[1][1] != '':
            image['Pixels']['@PhysicalSizeYUnit'] = pixel_size[1][1]
        if len(pixel_size) > 2 and pixel_size[2][1] != '':
            image['Pixels']['@PhysicalSizeZUnit'] = pixel_size[2][1]

        if description != '':
            image['Description'] = description
        if 'AcquisitionDate' in imetadata:
            image['AcquisitionDate'] = imetadata['AcquisitionDate']

        if 'StageLabel' in imetadata:
            image['StageLabel'] = imetadata['StageLabel']
        if 'Plane' in pmetadata:
            image['Pixels']['Plane'] = pmetadata['Plane']
        ome['Image'].append(image)

    instrument = None
    mag = source.get_mag()
    if 'Instrument' in source_metadata:
        instrument = source_metadata['Instrument']
    elif mag != 0:
        instrument = {'@ID': 'Instrument:1', 'Objective': {'@ID': 'Objective:1', '@NominalMagnification': mag}}

    if instrument is not None:
        ome['Instrument'] = instrument
        for image in ome['Image']:
            image['InstrumentRef'] = {'@ID': instrument['@ID']}

    if pyramid_sizes_add is not None:
        key_value_map = {'M': [{'@K': i + 1, '#text': f'{pyramid_size[0]} {pyramid_size[1]}'}
                         for i, pyramid_size in enumerate(pyramid_sizes_add)]}
        ome['StructuredAnnotations'] = [{'MapAnnotation': {
            '@ID': 'Annotation:Resolution:0',
            '@Namespace': 'openmicroscopy.org/PyramidResolution',
            'Value': key_value_map}
        }]

    for annotation in ensure_list(source_metadata.get('StructuredAnnotations', [])):
        annotation_type, value = next(iter(annotation.items()))
        # filter metadate dump as xml annotations, filter source pyramid sizes
        if 'xml' not in annotation_type.lower() and (
                not isinstance(value, dict) or 'resolution' not in next(iter(annotation.values())).get('@ID').lower()):
            ome['StructuredAnnotations'].append(annotation)

    return xmltodict.unparse({'OME': ome}, short_empty_elements=True, pretty=True)


def create_ome_metadata_ome_types(source: OmeSource, output_filename: str, channel_output: str = '', pyramid_sizes_add: list = None) -> str:
    file_name = os.path.basename(output_filename)
    file_title = get_filetitle(file_name)
    uuid = f'urn:uuid:{uuid4()}'
    ome = OME(uuid=uuid)
    source_metadata = source.get_metadata()
    rgb_colors = [(0xFF, 0, 0), (0, 0xFF, 0), (0, 0, 0xFF)]

    tiff_datas = [TiffData(uuid=UUID(file_name=file_name, value=uuid))]

    # only supporting single image
    nimages = 1

    for imagei in range(nimages):
        ome_channels = []

        imetadata = ensure_list(source_metadata.get('Image', {}))[imagei]
        pmetadata = imetadata.get('Pixels', {})
        acquisition_date = imetadata.get('AcquisitionDate')
        description = imetadata.get('Description', '')
        if description != '':
            description += ' '

        combine_channels = ('combine' in channel_output.lower())
        channel_info = source.channel_info
        nchannels = sum([info[1] for info in channel_info]) if not channel_output.isnumeric() else 1
        if combine_channels and len(channel_info) > 1:
            channel_info = [(channel_info[0][0], nchannels)]
        elif not combine_channels and len(channel_info) < nchannels:
            channel_info = [(channel_info[0][0], 1) for _ in range(nchannels)]
        is_rgb = (len(channel_info) == 3 and not channel_output.isnumeric())

        channels = ensure_list(pmetadata.get('Channel', {}))
        for channeli, info in enumerate(channel_info):
            if not channel_output.isnumeric() or channeli == int(channel_output):
                channel = channels[channeli] if channeli < len(channels) else {}
                color = channel.get('Color', -1)
                if color == -1 and is_rgb:
                    color = rgb_colors[channeli]
                ome_channels.append(Channel(
                    id=f'Channel:{len(ome_channels)}',
                    name=info[0],
                    samples_per_pixel=info[1],
                    fluor=channel.get('Fluor'),
                    color=color,
                    light_path=channel.get('LightPath'),
                    illumination_type=channel.get('IlluminationType'),
                    detector_settings=channel.get('DetectorSettings'),
                    acquisition_mode=channel.get('AcquisitionMode'),
                    filter_set_ref=channel.get('FilterSetRef'),
                ))

        description += 'converted by OmeSliCC'

        xyzct = source.sizes_xyzct[0]
        pixel_size = source.pixel_size
        image = Image(
            id=f'Image:{imagei}',
            name=file_title,
            pixels=Pixels(
                id='Pixels:0',
                size_x=xyzct[0],
                size_y=xyzct[1],
                size_z=xyzct[2],
                size_c=nchannels,
                size_t=xyzct[4],
                type=str(source.get_pixel_type()),
                dimension_order='XYZCT',
                channels=ome_channels,
                tiff_data_blocks=tiff_datas
            ),
        )
        if len(pixel_size) > 0 and pixel_size[0][0] != 0:
            image.pixels.physical_size_x = pixel_size[0][0]
        if len(pixel_size) > 1 and pixel_size[1][0] != 0:
            image.pixels.physical_size_y = pixel_size[1][0]
        if len(pixel_size) > 2 and pixel_size[2][0] != 0:
            image.pixels.physical_size_z = pixel_size[2][0]
        if len(pixel_size) > 0 and pixel_size[0][1] != '':
            image.pixels.physical_size_x_unit = pixel_size[0][1]
        if len(pixel_size) > 1 and pixel_size[1][1] != '':
            image.pixels.physical_size_y_unit = pixel_size[1][1]
        if len(pixel_size) > 2 and pixel_size[2][1] != '':
            image.pixels.physical_size_z_unit = pixel_size[2][1]

        if description != '':
            image.description = description
        if acquisition_date is not None:
            image.acquisition_date = acquisition_date

        #image.stage_label = imetadata.get('StageLabel')
        #image.pixels.planes = ensure_list(pmetadata.get('Plane'))
        # copy dict unsupported - work-around: manually create or copy formatted ome values
        if source.ome_metadata is not None and source.ome_metadata != OME():
            imeta = source.ome_metadata.images[imagei]
            image.stage_label = imeta.stage_label
            image.pixels.planes = imeta.pixels.planes
        ome.images.append(image)

    instrument = None
    mag = source.get_mag()
    ome.instruments = ensure_list(source_metadata.get('Instrument', []))
    if len(ome.instruments) > 0:
        instrument = ome.instruments[0]
    if instrument is None and mag != 0:
        instrument = Instrument(objectives=[Objective()])
        instrument.objectives[0].nominal_magnification = mag
        ome.instruments.append(instrument)

    if instrument is not None:
        for image in ome.images:
            image.instrument_ref = InstrumentRef(id=instrument.id)

    if pyramid_sizes_add is not None:
        key_value_map = Map()
        for i, pyramid_size in enumerate(pyramid_sizes_add):
            key_value_map.m.append(M(k=(i + 1), value=' '.join(map(str, pyramid_size))))
        annotation = MapAnnotation(value=key_value_map,
                                   namespace='openmicroscopy.org/PyramidResolution',
                                   id='Annotation:Resolution:0')
        ome.structured_annotations.append(annotation)

    #for annotation in ensure_list(source_metadata.get('StructuredAnnotations', []):
    # copy dict unsupported - work-around: manually create or copy formatted ome values
    if source.ome_metadata is not None and source.ome_metadata != OME():
        for annotation in source.ome_metadata.structured_annotations:
            if 'resolution' not in annotation.id.lower():
                ome.structured_annotations.append(annotation)

    return ome.to_xml()


def create_ome_metadata_from_omero(image_object: omero.gateway.ImageWrapper, filetitle: str,
                                   pyramid_sizes_add: list = None) -> OME:
    uuid = f'urn:uuid:{uuid4()}'
    ome = OME(uuid=uuid)

    nchannels = image_object.getSizeC()
    pixels = image_object.getPrimaryPixels()
    channels = []
    channels0 = image_object.getChannels(noRE=True)
    if channels0 is not None and len(channels0) > 0:
        channel = channels0[0]
        channels.append(Channel(
                id='Channel:0',
                name=channel.getName(),
                fluor=channel.getName(),
                samples_per_pixel=nchannels
            ))

    tiff_datas = [TiffData(uuid=UUID(file_name=filetitle, value=uuid))]

    planes = []
    stage = image_object.getStageLabel()
    if stage is not None:
        for plane in pixels.copyPlaneInfo():
            planes.append(Plane(
                the_c=plane.getTheC(),
                the_t=plane.getTheT(),
                the_z=plane.getTheZ(),
                delta_t=plane.getDeltaT(),
                exposure_time=plane.getExposureTime(),
                position_x=stage.getPositionX().getValue(),
                position_y=stage.getPositionY().getValue(),
                position_z=stage.getPositionZ().getValue(),
                position_x_unit=stage.getPositionX().getSymbol(),
                position_y_unit=stage.getPositionY().getSymbol(),
                position_z_unit=stage.getPositionZ().getSymbol(),
            ))
        stage_label = StageLabel(
            name=stage.getName(),
            x=stage.getPositionX().getValue(),
            y=stage.getPositionY().getValue(),
            z=stage.getPositionZ().getValue(),
            x_unit=stage.getPositionX().getSymbol(),
            y_unit=stage.getPositionY().getSymbol(),
            z_unit=stage.getPositionZ().getSymbol()
        )

    image = Image(
        id='Image:0',
        name=image_object.getName(),
        description=image_object.getDescription(),
        acquisition_date=image_object.getAcquisitionDate(),
        pixels=Pixels(
            size_c=image_object.getSizeC(),
            size_t=image_object.getSizeT(),
            size_x=image_object.getSizeX(),
            size_y=image_object.getSizeY(),
            size_z=image_object.getSizeZ(),
            physical_size_x=image_object.getPixelSizeX(),   # get size in default unit (micron)
            physical_size_y=image_object.getPixelSizeY(),   # get size in default unit (micron)
            physical_size_z=image_object.getPixelSizeZ(),   # get size in default unit (micron)
            physical_size_x_unit='µm',
            physical_size_y_unit='µm',
            physical_size_z_unit='µm',
            type=image_object.getPixelsType(),
            dimension_order=pixels.getDimensionOrder().getValue(),
            channels=channels,
            tiff_data_blocks=tiff_datas
        ),
    )
    if stage is not None:
        image.stage_label = stage_label
        image.pixels.planes = planes
    ome.images.append(image)

    objective_settings = image_object.getObjectiveSettings()
    if objective_settings is not None:
        objective = objective_settings.getObjective()
        instrument = Instrument(objectives=[
            Objective(id=objective.getId(),
                      manufacturer=objective.getManufacturer(),
                      model=objective.getModel(),
                      lot_number=objective.getLotNumber(),
                      serial_number=objective.getSerialNumber(),
                      nominal_magnification=objective.getNominalMagnification(),
                      calibrated_magnification=objective.getCalibratedMagnification(),
                      #correction=objective.getCorrection().getValue(),
                      lens_na=objective.getLensNA(),
                      working_distance=objective.getWorkingDistance().getValue(),
                      working_distance_unit=objective.getWorkingDistance().getSymbol(),
                      iris=objective.getIris(),
                      immersion=objective.getImmersion().getValue()
                      )])
        ome.instruments.append(instrument)

        for image in ome.images:
            image.instrument_ref = InstrumentRef(id=instrument.id)

    if pyramid_sizes_add is not None:
        key_value_map = Map()
        for i, pyramid_size in enumerate(pyramid_sizes_add):
            key_value_map.m.append(M(k=(i + 1), value=' '.join(map(str, pyramid_size))))
        annotation = MapAnnotation(value=key_value_map,
                                   namespace='openmicroscopy.org/PyramidResolution',
                                   id='Annotation:Resolution:0')
        ome.structured_annotations.append(annotation)

    for omero_annotation in image_object.listAnnotations():
        id = omero_annotation.getId()
        type = omero_annotation.OMERO_TYPE
        annotation = None
        if type == omero.model.MapAnnotationI:
            key_value_map = Map()
            for annotation in omero_annotation.getMapValue():
                key_value_map.m.append(M(k=annotation.name, value=annotation.value))
            annotation = MapAnnotation(value=key_value_map,
                                       namespace=omero_annotation.getNs(),
                                       id=f'urn:lsid:export.openmicroscopy.org:Annotation:{id}')
        elif type == omero.model.CommentAnnotationI:
            annotation = CommentAnnotation(value=omero_annotation.getValue(),
                                           namespace=omero_annotation.getNs(),
                                           id=f'urn:lsid:export.openmicroscopy.org:Annotation:{id}')
        if annotation is not None:
            ome.structured_annotations.append(annotation)
            for image in ome.images:
                image.annotation_ref.append(AnnotationRef(id=annotation.id))

    return ome
