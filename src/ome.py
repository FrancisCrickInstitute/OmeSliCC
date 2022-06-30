import os
from uuid import uuid4
from ome_types import OME
from ome_types.model import Image, Pixels, Plane, Channel, Instrument, Objective, StageLabel, Map, MapAnnotation, \
    CommentAnnotation, InstrumentRef, AnnotationRef, TiffData, LightPath
from ome_types.model.map import M
from ome_types.model.tiff_data import UUID


def create_ome_metadata(file_name0, image_info, channels, stage=None, planes=[], objective=None,
                        map_annotations=[], comment_annotations=[], pyramid_sizes_add=None):
    file_name = os.path.basename(file_name0)
    file_title = os.path.splitext(file_name)[0]
    if file_title.endswith('.ome'):
        file_title = os.path.splitext(file_title)[0]
    uuid = f'urn:uuid:{uuid4()}'
    ome = OME(uuid=uuid)

    ome_channels = []
    for c, channel in enumerate(channels):
        ome_channels.append(Channel(
            id=f'Channel:{c}',
            name=channel.get('name'),
            fluor=channel.get('fluor'),
            color=channel.get('color', -1),
            samples_per_pixel=channel.get('samples_per_channel', 1),
            light_path=LightPath()
        ))

    tiff_datas = [TiffData(
        uuid=UUID(file_name=file_name, value=uuid),
        first_c=0, first_t=0, first_z=0, ifd=0, plane_count=1
    )]

    ome_planes = []
    if stage is not None:
        for plane in planes:
            ome_planes.append(Plane(
                the_c=plane.get('the_c'), the_t=plane.get('the_t'), the_z=plane.get('the_z'),
                delta_t=plane.get('delta_t'),
                exposure_time=plane.get('exposure_time'),
                position_x=stage.get('position_x'),
                position_x_unit=stage.get('position_x_unit'),
                position_y=stage.get('position_y'),
                position_y_unit=stage.get('position_y_unit'),
                position_z=stage.get('position_z'),
                position_z_unit=stage.get('position_z_unit')
            ))
        stage_label = StageLabel(
            name=stage.get('name'),
            x=stage.get('x'),
            x_unit=stage.get('x_unit'),
            y=stage.get('y'),
            y_unit=stage.get('y_unit'),
            z=stage.get('z'),
            z_unit=stage.get('z_unit')
        )

    image = Image(
        id='Image:0',
        name=file_title,
        description=image_info.get('description'),
        acquisition_date=image_info.get('acquisition_date'),
        pixels=Pixels(
            id='Pixels:0',
            size_c=image_info.get('size_c'),
            size_t=image_info.get('size_t'),
            size_x=image_info.get('size_x'),
            size_y=image_info.get('size_y'),
            size_z=image_info.get('size_z'),
            physical_size_x=image_info.get('physical_size_x'),
            physical_size_y=image_info.get('physical_size_y'),
            physical_size_z=image_info.get('physical_size_z'),
            type=image_info.get('type'),
            dimension_order=image_info.get('dimension_order'),
            channels=ome_channels,
            tiff_data_blocks=tiff_datas
        ),
    )
    if stage is not None:
        image.stage_label = stage_label
        image.pixels.planes = ome_planes
    ome.images.append(image)

    if objective is not None:
        instrument = Instrument(objectives=[
            Objective(id=objective.getId(),
                      manufacturer=objective.get('manufacturer'),
                      model=objective.get('model'),
                      lot_number=objective.get('lot_number'),
                      serial_number=objective.get('serial_number'),
                      nominal_magnification=objective.get('nominal_magnification'),
                      calibrated_magnification=objective.get('calibrated_magnification'),
                      # correction=objective.get('correction'),
                      lens_na=objective.get('lens_na'),
                      working_distance=objective.get('working_distance'),
                      working_distance_unit=objective.get('working_distance_unit'),
                      iris=objective.get('iris'),
                      immersion=objective.get('immersion')
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

    for map_annotation in map_annotations:
        key_value_map = Map()
        for annotation in map_annotation['value']:
            key_value_map.m.append(M(k=annotation.key, value=annotation.value))
        ome_annotation = MapAnnotation(value=key_value_map,
                                       namespace=map_annotation['namespace'],
                                       id=f'urn:lsid:export.openmicroscopy.org:Annotation:{map_annotation["id"]}')
        ome.structured_annotations.append(ome_annotation)
        for image in ome.images:
            image.annotation_ref.append(AnnotationRef(id=ome_annotation.id))

    for annotation in comment_annotations:
        ome_annotation = CommentAnnotation(value=annotation['value'],
                                           namespace=annotation['namespace'],
                                           id=f'urn:lsid:export.openmicroscopy.org:Annotation:{annotation["id"]}')
        ome.structured_annotations.append(ome_annotation)
        for image in ome.images:
            image.annotation_ref.append(AnnotationRef(id=ome_annotation.id))

    return ome
