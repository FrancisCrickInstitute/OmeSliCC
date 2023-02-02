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

from src.image_util import ensure_unsigned_type
from src.util import get_filetitle, ensure_list
from version import __version__


OME_URI = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
OME_XSI = "http://www.w3.org/2001/XMLSchema-instance"
OME_SCHEMA_LOC = f"{OME_URI} {OME_URI}/ome.xsd"


def create_ome_metadata(source: OmeSource, output_filename: str, channel_output: str = '', pyramid_sizes_add: list = None) -> str:
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
        if instrument is not None:
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

        combine_rgb = ('combine' in channel_output.lower())
        split_channel_files = channel_output.isnumeric()
        channel_info = source.channel_info
        nchannels = sum([info[1] for info in channel_info]) if not split_channel_files else 1
        if combine_rgb and len(channel_info) == 3:
            channel_info = [(channel_info[0][0], nchannels)]
        elif not combine_rgb and len(channel_info) < nchannels:
            channel_info = [(channel_info[0][0], 1) for _ in range(nchannels)]
        channels0 = ensure_list(pixels0.get('Channel', []))
        channeli = 0
        planes0 = ensure_list(pixels0.get('Plane', []))

        for channeli0, info in enumerate(channel_info):
            if not split_channel_files or channeli0 == int(channel_output):
                channel = channels0[channeli0].copy() if channeli0 < len(channels0) else {}
                color = channel.get('Color')
                channel['@ID'] = f'Channel:{imagei}:{channeli}'
                if info[0] != '':
                    channel['@Name'] = info[0]
                channel['@SamplesPerPixel'] = info[1] if not split_channel_files else 1
                if color is not None:
                    channel['@Color'] = color
                channels.append(channel)

                for plane0 in planes0:
                    plane = plane0.copy()
                    plane_channel0 = plane0.get('@TheC')
                    if split_channel_files:
                        if int(plane_channel0) == int(channel_output):
                            plane['@TheC'] = channeli
                            planes.append(plane)
                    elif plane_channel0 is None or int(plane_channel0) == channeli0:
                        planes.append(plane)

                channeli += 1

        xyzct = source.get_size_xyzct()
        pixel_size = source.pixel_size

        image = {
            '@ID': f'Image:{imagei}',
            '@Name': file_title,
        }

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
        if len(planes) > 0:
            pixels['Plane'] = planes

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
            'Value': key_value_map})
    ome['StructuredAnnotations']['MapAnnotation'] = map_annotations

    # filter original metadata elements
    xml_annotations0 = ensure_list(ome.get('StructuredAnnotations', {}).get('XMLAnnotation', []))
    xml_annotations = [annotation for annotation in xml_annotations0
                       if 'originalmetadata' not in annotation.get('@Namespace', '').lower()]
    ome['StructuredAnnotations']['XMLAnnotation'] = xml_annotations

    return xmltodict.unparse({'OME': ome}, short_empty_elements=True, pretty=True)


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
