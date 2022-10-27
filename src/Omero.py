import logging
import os
import numpy as np
import omero
from omero.gateway import BlitzGateway
import omero.model
from tqdm import tqdm

from src.conversion import save_tiff
from src.image_util import calc_pyramid, get_image_size_info
from src.ome import create_ome_metadata_from_omero
from src.omero_credentials import decrypt_credentials


class Omero:
    def __init__(self, params: dict):
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
        logging.info(f'Connected as {self.conn.getUser().getName()}')

    def disconnect(self):
        self.conn.close()
        self.connected = False

    def list_projects(self):
        projects = self.conn.listProjects()      # may include other users' data
        for project in projects:
            print_omero_object(project)

    def get_project(self, project_id: int) -> omero.gateway.ProjectWrapper:
        project = self.conn.getObject('Project', project_id)
        return project

    def get_image_object(self, image_id: int) -> omero.gateway.ImageWrapper:
        image_object = self.conn.getObject('Image', image_id)
        return image_object

    def get_annotation_image_ids(self, project_id: int, target_labels: list, filter_label_macro: bool = False) -> tuple:
        image_ids = []
        image_names = []
        image_annotations = []
        project = self.get_project(project_id)
        for dataset in project.listChildren():
            for image_object in dataset.listChildren():
                name = image_object.getName()
                # filter _label and _macro items
                if not filter_label_macro or (not name.endswith('_label') and not name.endswith('_macro')):
                    annotations = self.get_image_annotations(image_object, target_labels)
                    if len(annotations) == len(target_labels):
                        image_ids.append(image_object.getId())
                        image_names.append(image_object.getName())
                        image_annotations.append(annotations)
        return image_ids, image_names, image_annotations

    def get_project_images(self, project_id: int) -> list:
        image_objects = []
        project = self.get_project(project_id)
        for dataset in project.listChildren():
            for image_object in dataset.listChildren():
                image_objects.append(image_object)
        return image_objects

    def get_image_info(self, image_id: int) -> str:
        image_object = self.get_image_object(image_id)
        xyzct = self.get_size(image_object)
        pixels = image_object.getPrimaryPixels()
        pixels_type = pixels.getPixelsType()
        pixel_type = pixels_type.getValue()
        type_size_bytes = pixels_type.getBitSize() / 8
        channel_info = []
        for channel in image_object.getChannels():
            channel_info.append((channel.getName(), 1))
        image_info = f'{image_id} {image_object.getName()} ' \
                     + get_image_size_info(xyzct, type_size_bytes, pixel_type, channel_info)
        return image_info

    def extract_thumbnail(self, image_id: int, outpath: str):
        image_object = self.get_image_object(image_id)
        filetitle = image_object.getName() + '.jpg'
        outfilename = os.path.join(outpath, filetitle)
        if not os.path.exists(outfilename):
            # do not overwrite existing files
            w, h, zs, cs, ts = self.get_size(image_object)
            w2, h2 = int(round(w / 256)), int(round(h / 256))
            try:
                thumb_bytes = image_object.getThumbnail((w2, h2)).encode()
                with open(outfilename, 'wb') as file:
                    file.write(thumb_bytes)
            except Exception as e:
                logging.error(e)

    def convert_images(self, image_ids: list, outpath: str):
        # currently only ome-tiff supported
        self.convert_images_to_ometiff(image_ids, outpath)

    def convert_images_to_ometiff(self, image_ids: list, outpath: str):
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for image_id in tqdm(image_ids):
            self.convert_image_to_ometiff(image_id, outpath)

    def convert_image_to_ometiff(self, image_id: int, outpath: str):
        output = self.params['output']
        image_object = self.get_image_object(image_id)
        filetitle = image_object.getName() + '.ome.tiff'
        outfilename = os.path.join(outpath, filetitle)
        if not os.path.exists(outfilename):
            # do not overwrite existing files
            xyzct = self.get_size(image_object)
            w, h, zs, cs, ts = xyzct
            logging.info(f'{image_id} {image_object.getName()}')

            npyramid_add = output.get('npyramid_add', 0)
            pyramid_downsample = output.get('pyramid_downsample', 0)
            pyramid_sizes_add = calc_pyramid((w, h), npyramid_add, pyramid_downsample)
            metadata = create_ome_metadata_from_omero(image_object, filetitle, pyramid_sizes_add)
            xml_metadata = metadata.to_xml()

            image = self.get_omero_image(image_object)
            if image is not None:
                save_tiff(outfilename, image, xml_metadata=xml_metadata,
                          tile_size=output.get('tile_size'),
                          compression=output.get('compression'),
                          pyramid_sizes_add=pyramid_sizes_add)

    def get_omero_image0(self, image_object: omero.gateway.ImageWrapper, pixels: omero.gateway.PixelsWrapper):
        w, h, zs, cs, ts = self.get_size(image_object)
        read_size = 10240
        ny = int(np.ceil(h / read_size))
        nx = int(np.ceil(w / read_size))

        dtype = np.dtype(pixels.getPixelsType().getValue()).type
        image = np.zeros((h, w, cs), dtype=dtype)

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
                tile_gen = pixels.getTiles(tile_coords)
                for c, tile in enumerate(tile_gen):
                    image[sy:sy + th, sx:sx + tw, c] = tile

    def get_omero_image(self, image_object: omero.gateway.ImageWrapper, read_size: int = 10240) -> np.ndarray:
        w, h, zs, cs, ts = self.get_size(image_object)
        pixels = image_object.getPrimaryPixels()
        dtype = np.dtype(pixels.getPixelsType().getValue()).type
        image = np.zeros((h, w, cs), dtype=dtype)

        try:
            pixels_store = self.conn.createRawPixelsStore()
            pixels_id = image_object.getPixelsId()
            pixels_store.setPixelsId(pixels_id, False, self.conn.SERVICE_OPTS)
            ny = int(np.ceil(h / read_size))
            nx = int(np.ceil(w / read_size))
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
                        image[sy:sy + th, sx:sx + tw, c] = tile
        except Exception as e:
            logging.error(e)
            image = None
        finally:
            pixels_store.close()
        return image

    def get_size(self, image_object: omero.gateway.ImageWrapper) -> tuple:
        xs, ys = image_object.getSizeX(), image_object.getSizeY()
        zs, cs, ts = image_object.getSizeZ(), image_object.getSizeC(), image_object.getSizeT()
        return xs, ys, zs, cs, ts

    def get_original_files(self, image_object: omero.gateway.ImageWrapper) -> list:
        return image_object.getFileset().listFiles()

    def get_magnification(self, image_object: omero.gateway.ImageWrapper) -> float:
        return image_object.getObjectiveSettings().getObjective().getNominalMagnification()

    def get_image_annotations(self, image_object: omero.gateway.ImageWrapper, annotation_keys: list) -> dict:
        annotations = {}
        for omero_annotation in image_object.listAnnotations():
            if omero_annotation.OMERO_TYPE == omero.model.MapAnnotationI:
                for annotation_key in annotation_keys:
                    for annotation in omero_annotation.getMapValue():
                        if annotation.name.lower() == annotation_key.lower():
                            annotations[annotation_key] = annotation.value
        return annotations


def print_omero_object(omero_object: omero.gateway.BlitzObjectWrapper, indent: int = 0):
    logging.info("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        omero_object.OMERO_CLASS,
        omero_object.getId(),
        omero_object.getName(),
        omero_object.getOwnerOmeName()))

    for child in omero_object.listChildren():
        logging.info('\t', child.getName())
