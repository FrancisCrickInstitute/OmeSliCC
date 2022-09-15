import logging
import os
from uuid import uuid4
import numpy as np
import omero
from ome_types import OME
from ome_types.model import Image, Pixels, Plane, Channel, Instrument, Objective, StageLabel, Map, MapAnnotation, \
    CommentAnnotation, InstrumentRef, AnnotationRef, TiffData
from ome_types.model.map import M
from ome_types.model.tiff_data import UUID
from omero.gateway import BlitzGateway
from tqdm import tqdm

from src.conversion import save_tiff
from src.image_util import calc_pyramid, get_image_size_info, show_image
from src.omero_credentials import decrypt_credentials


class Omero:
    def __init__(self, params):
        self.params = params
        self.private_key_filename = params['credentials']['private_key']
        self.credentials_filename = params['credentials']['credentials']
        self.connected = False

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def init(self):
        self.connect()
        self.switch_user_group()

    def switch_user_group(self):
        self.conn.SERVICE_OPTS.setOmeroGroup('-1')

    def connect(self):
        logging.info('Connecting to Omero...')
        usr, pwd = decrypt_credentials(self.private_key_filename, self.credentials_filename)
        self.conn = BlitzGateway(usr, pwd, host=self.params['input']['omero'], secure=True)
        if not self.conn.connect():
            self.disconnect()
            logging.error('Connection error')
            raise ConnectionError
        self.conn.c.enableKeepAlive(60)
        self.connected = True
        #mds = self.conn.getMetadataService()
        logging.info(f'Connected as {self.conn.getUser().getName()}')

    def disconnect(self):
        self.conn.close()
        self.connected = False

    def list_projects(self):
        projects = self.conn.listProjects()      # may include other users' data
        for project in projects:
            print_omero_object(project)

    def get_project(self, project_id):
        project = self.conn.getObject('Project', project_id)
        return project

    def get_image_object(self, image_id):
        image_object = self.conn.getObject('Image', image_id)
        return image_object

    def get_annotation_image_ids(self, project_id, target_labels):
        image_ids = []
        image_names = []
        image_annotations = []
        project = self.get_project(project_id)
        for dataset in project.listChildren():
            for image_object in dataset.listChildren():
                name = image_object.getName()
                # filter _label and _macro items
                if not name.endswith('_label') and not name.endswith('_macro'):
                    annotations = self.get_image_annotations(image_object, target_labels)
                    if len(annotations) == len(target_labels):
                        image_ids.append(image_object.getId())
                        image_names.append(image_object.getName())
                        image_annotations.append(annotations)
        return image_ids, image_names, image_annotations

    def get_project_images(self, project_id):
        image_objects = []
        project = self.get_project(project_id)
        for dataset in project.listChildren():
            for image_object in dataset.listChildren():
                image_objects.append(image_object)
        return image_objects

    def get_image_info(self, image_id):
        image_object = self.get_image_object(image_id)
        xyzct = self.get_size(image_object)
        pixels = image_object.getPrimaryPixels()
        pixel_type = pixels.getPixelsType()
        type_size_bytes = pixel_type.getBitSize() / 8
        image_info = f'{image_id} {image_object.getName()} ' + get_image_size_info(xyzct, type_size_bytes)
        logging.info(image_info)
        return image_info

    def extract_thumbnail(self, image_id, outpath):
        image_object = self.get_image_object(image_id)
        filetitle = image_object.getName() + '.jpg'
        outfilename = os.path.join(outpath, filetitle)
        if not os.path.exists(outfilename):
            # do not overwrite existing files
            w, h, zs, cs, ts = self.get_size(image_object)
            w2, h2 = int(round(w / 256)), int(round(h / 256))
            try:
                thumb_bytes = image_object.getThumbnail((w2, h2))
                with open(outfilename, 'wb') as file:
                    file.write(thumb_bytes)
            except Exception as e:
                logging.error(e)

    def convert_slides(self, image_ids, outpath):
        # currently only ome-tiff supported
        self.convert_slides_to_ometiff(image_ids, outpath)

    def convert_slides_to_ometiff(self, image_ids, outpath):
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for image_id in tqdm(image_ids):
            self.convert_slide_to_ometiff(image_id, outpath)

    def convert_slide_to_ometiff(self, image_id, outpath):
        output = self.params['output']
        image_object = self.get_image_object(image_id)
        filetitle = image_object.getName() + '.ome.tiff'
        outfilename = os.path.join(outpath, filetitle)
        if not os.path.exists(outfilename):
            # do not overwrite existing files
            xyzct = self.get_size(image_object)
            w, h, zs, cs, ts = xyzct
            pixels = image_object.getPrimaryPixels()
            pixel_nbytes = pixels.getPixelsType().getBitSize() / 8
            logging.info(f'{image_id} {image_object.getName()} {get_image_size_info(xyzct, pixel_nbytes)}')

            #tiff_content = image_object.exportOmeTiff()    # not working (~image too large)
            #with open(outfilename, 'wb') as writer:
            #    writer.write(tiff_content)

            # slide_image = pixels.getPlane()   # not working (~image too large)

            pyramid_add = output['pyramid_add']
            pyramid_downsample = output['pyramid_downsample']
            pyramid_sizes_add = calc_pyramid((w, h), pyramid_add, pyramid_downsample)
            metadata = self.get_metadata(image_object, filetitle, pyramid_sizes_add)
            xml_metadata = metadata.to_xml()

            # test saving blank image
            #slide_image = np.zeros((h, w, cs), dtype=np.uint8)
            #save_tiff(outfilename, slide_image, xml_metadata=xml_metadata, tile_size=output['tile_size'], compression=output['compression'], pyramid_sizes_add=pyramid_sizes_add)

            slide_image = self.get_slide_image(image_object, pixels)
            if slide_image is not None:
                save_tiff(outfilename, slide_image, xml_metadata=xml_metadata,
                          tile_size=output['tile_size'],
                          compression=output['compression'],
                          pyramid_sizes_add=pyramid_sizes_add)

    def get_slide_image0(self, image_object, pixels):
        w, h, zs, cs, ts = self.get_size(image_object)
        read_size = 10240
        ny = int(np.ceil(h / read_size))
        nx = int(np.ceil(w / read_size))

        dtype = np.dtype(pixels.getPixelsType().getValue()).type
        slide_image = np.zeros((h, w, cs), dtype=dtype)

        for y in range(ny):
            for x in range(nx):
                sx, sy = x * read_size, y * read_size
                tw, th = read_size, read_size
                if sx + tw > w:
                    tw = w - sx
                if sy + th > h:
                    th = h - sy
                xywh = (sx, sy, tw, th)
                tile_coords = [(0, c, 0, xywh) for c in range(cs)]
                image_gen = pixels.getTiles(tile_coords)
                for c, image in enumerate(image_gen):
                    slide_image[sy:sy + th, sx:sx + tw, c] = image

    def get_slide_image(self, image_object, pixels):
        w, h, zs, cs, ts = self.get_size(image_object)
        read_size = 10240
        ny = int(np.ceil(h / read_size))
        nx = int(np.ceil(w / read_size))

        dtype = np.dtype(pixels.getPixelsType().getValue()).type
        slide_image = np.zeros((h, w, cs), dtype=dtype)

        try:
            pixels_store = self.conn.createRawPixelsStore()
            pixels_id = image_object.getPixelsId()
            pixels_store.setPixelsId(pixels_id, False, self.conn.SERVICE_OPTS)
            for y in range(ny):
                for x in range(nx):
                    sx, sy = x * read_size, y * read_size
                    tw, th = read_size, read_size
                    if sx + tw > w:
                        tw = w - sx
                    if sy + th > h:
                        th = h - sy
                    for c in range(cs):
                        tile0 = pixels_store.getTile(0, c, 0, sx, sy, tw, th)
                        tile = np.frombuffer(tile0, dtype=dtype)
                        tile.resize(th, tw)
                        slide_image[sy:sy + th, sx:sx + tw, c] = tile
        except Exception as e:
            logging.error(e)
            slide_image = None
        finally:
            pixels_store.close()
        return slide_image

    def get_size(self, image_object):
        xs, ys = image_object.getSizeX(), image_object.getSizeY()
        zs, cs, ts = image_object.getSizeZ(), image_object.getSizeC(), image_object.getSizeT()
        return xs, ys, zs, cs, ts

    def get_metadata(self, image_object, filetitle, pyramid_sizes_add=None):
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

        tiff_datas = [TiffData(
            uuid=UUID(file_name=filetitle, value=uuid)
        )]

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
                    position_x_unit=stage.getPositionX().getSymbol(),
                    position_y=stage.getPositionY().getValue(),
                    position_y_unit=stage.getPositionY().getSymbol(),
                    position_z=stage.getPositionZ().getValue(),
                    position_z_unit=stage.getPositionZ().getSymbol(),
                ))
            stage_label = StageLabel(
                name=stage.getName(),
                x=stage.getPositionX().getValue(),
                x_unit=stage.getPositionX().getSymbol(),
                y=stage.getPositionY().getValue(),
                y_unit=stage.getPositionY().getSymbol(),
                z=stage.getPositionZ().getValue(),
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
                physical_size_x=image_object.getPixelSizeX(),
                physical_size_y=image_object.getPixelSizeY(),
                physical_size_z=image_object.getPixelSizeZ(),
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

    def get_original_slide_files(self, image_object):
        return image_object.getFileset().listFiles()

    def get_magnification(self, image_object):
        return image_object.getObjectiveSettings().getObjective().getNominalMagnification()

    def get_image_annotations(self, image_object, annotation_keys):
        annotations = {}
        for omero_annotation in image_object.listAnnotations():
            if omero_annotation.OMERO_TYPE == omero.model.MapAnnotationI:
                for annotation_key in annotation_keys:
                    for annotation in omero_annotation.getMapValue():
                        if annotation.name.lower() == annotation_key.lower():
                            annotations[annotation_key] = annotation.value
        return annotations


def print_omero_object(object, indent=0):
    """
    Helper method to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    logging.info("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        object.OMERO_CLASS,
        object.getId(),
        object.getName(),
        object.getOwnerOmeName()))

    for child in object.listChildren():
        logging.info('\t', child.getName())
