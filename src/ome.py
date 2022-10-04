import os
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

    tiff_datas = [TiffData(
        uuid=UUID(file_name=file_name, value=uuid),
        first_c=0, first_t=0, first_z=0, ifd=0, plane_count=1
    )]

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
            id='Image:0',
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

    if source.ome_metadata is not None:
        ome.instruments = source.ome_metadata.instruments
        if len(ome.instruments) > 0:
            instrument = ome.instruments[0]
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
