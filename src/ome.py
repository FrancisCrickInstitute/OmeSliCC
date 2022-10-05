import os
import omero
from uuid import uuid4
from ome_types import OME
from ome_types.model import Image, Pixels, Plane, Channel, Instrument, Objective, StageLabel, Map, MapAnnotation, \
    CommentAnnotation, InstrumentRef, AnnotationRef, TiffData, LightPath
from ome_types.model.map import M
from ome_types.model.tiff_data import UUID

from src.util import get_filetitle, ensure_list


def create_ome_metadata(source, output_filename, pyramid_sizes_add=None):
    file_name = os.path.basename(output_filename)
    file_title = get_filetitle(file_name)
    uuid = f'urn:uuid:{uuid4()}'
    ome = OME(uuid=uuid)

    tiff_datas = [TiffData(uuid=UUID(file_name=file_name, value=uuid))]

    for i, imetadata in enumerate(ensure_list(source.get_metadata().get('Image', {}))):
        description = ''
        acquisition_date = None
        ome_channels = []
        if source.ome_metadata is not None:
            pmetadata = imetadata.get('Pixels', {})
            description = imetadata.get('Description')
            if description is None:
                description = ''
            elif description != '':
                description += ' '
            acquisition_date = imetadata.get('AcquisitionDate')
            xyzct = [pmetadata['SizeX'], pmetadata['SizeY'], pmetadata['SizeZ'], pmetadata['SizeC'], pmetadata['SizeT']]
            physical_size = [pmetadata.get('PhysicalSizeX', 0),
                             pmetadata.get('PhysicalSizeY', 0),
                             pmetadata.get('PhysicalSizeZ', 0)]
            physical_size_unit = [pmetadata.get('PhysicalSizeXUnit', ''),
                                  pmetadata.get('PhysicalSizeYUnit', ''),
                                  pmetadata.get('PhysicalSizeZUnit', '')]
            for c, channel in enumerate(ensure_list(pmetadata.get('Channel', {}))):
                ome_channels.append(Channel(
                    id=f'Channel:{c}',
                    name=channel.get('Name'),
                    fluor=channel.get('Fluor'),
                    color=channel.get('Color', -1),
                    samples_per_pixel=channel.get('SamplesPerPixel', 1),
                    light_path=LightPath()
                ))
        else:
            xyzct = source.sizes_xyzct[0]
            physical_size = [pixel_size[0] * size for size, pixel_size in zip(xyzct, source.pixel_size)]
            physical_size_unit = [pixel_size[1] for pixel_size in source.pixel_size]
            for c, channel_info in enumerate(source.channel_info):
                ome_channels.append(Channel(
                    id=f'Channel:{c}',
                    name=channel_info[0],
                    samples_per_pixel=channel_info[1],
                    light_path=LightPath()
                ))
        description += 'converted by OmeSliCC'

        image = Image(
            id=f'Image:{i}',
            name=file_title,
            pixels=Pixels(
                id='Pixels:0',
                size_x=xyzct[0],
                size_y=xyzct[1],
                size_z=xyzct[2],
                size_c=xyzct[3],
                size_t=xyzct[4],
                type=str(source.get_pixel_type()),
                dimension_order='XYZCT',
                channels=ome_channels,
                tiff_data_blocks=tiff_datas
            ),
        )
        if len(physical_size) > 0 and physical_size[0] != 0:
            image.pixels.physical_size_x = physical_size[0]
        if len(physical_size) > 1 and physical_size[1] != 0:
            image.pixels.physical_size_y = physical_size[1]
        if len(physical_size) > 2 and physical_size[2] != 0:
            image.pixels.physical_size_z = physical_size[2]
        if len(physical_size_unit) > 0 and physical_size_unit[0] != '':
            image.pixels.physical_size_x_unit = physical_size_unit[0]
        if len(physical_size_unit) > 1 and physical_size_unit[1] != '':
            image.pixels.physical_size_y_unit = physical_size_unit[1]
        if len(physical_size_unit) > 2 and physical_size_unit[2] != '':
            image.pixels.physical_size_z_unit = physical_size_unit[2]
        if description != '':
            image.description = description
        if acquisition_date is not None:
            image.acquisition_date = acquisition_date

        if source.ome_metadata is not None:
            imeta = source.ome_metadata.images[i]
            image.stage_label = imeta.stage_label
            image.pixels.planes = imeta.pixels.planes
        ome.images.append(image)

    instrument = None
    mag = source.get_mag()
    if source.ome_metadata is not None:
        ome.instruments = source.ome_metadata.instruments
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

    if source.ome_metadata is not None:
        ome.structured_annotations = source.ome_metadata.structured_annotations

    return ome


def create_ome_metadata_from_omero(self, image_object, filetitle, pyramid_sizes_add=None):
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
