# TODO: refactor this class as OmeSource, so image export/conversion can be reused

from __future__ import annotations
import logging
import os
import numpy as np
import omero
from omero.gateway import BlitzGateway
import omero.model
from tqdm import tqdm
from types import TracebackType

from src.conversion import save_tiff
from src.image_util import calc_pyramid, get_image_size_info
from src.ome import create_ome_metadata_from_omero
from src.omero_credentials import decrypt_credentials
from src.util import ensure_list


class Omero:
    """Omero image and metadata extraction"""
    
    def __init__(self, params: dict):
        self.params = params
        self.private_key_filename = params['credentials']['private_key']
        self.credentials_filename = params['credentials']['credentials']
        self.connected = False

    def __enter__(self) -> Omero:
        self.init()
        return self

    def __exit__(self, exc_type: type[BaseException], exc_value: BaseException, traceback: TracebackType):
        self.close()

    def init(self):
        self._connect()
        self._switch_user_group()

    def close(self):
        self._disconnect()

    def _connect(self):
        logging.info('Connecting to Omero...')
        usr, pwd = decrypt_credentials(self.private_key_filename, self.credentials_filename)
        self.conn = BlitzGateway(usr, pwd, host=self.params['input']['omero']['host'], secure=True)
        if not self.conn.connect():
            self._disconnect()
            logging.error('Connection error')
            raise ConnectionError
        self.conn.c.enableKeepAlive(60)
        self.connected = True
        logging.info(f'Connected as {self.conn.getUser().getName()}')

    def _disconnect(self):
        self.conn.close()
        self.connected = False

    def _switch_user_group(self):
        self.conn.SERVICE_OPTS.setOmeroGroup('-1')

    def _get_project(self, project_id: int) -> omero.gateway.ProjectWrapper:
        project = self.conn.getObject('Project', project_id)
        return project

    def _get_dataset(self, dataset_id: int) -> omero.gateway.DatasetWrapper:
        dataset = self.conn.getObject('Dataset', dataset_id)
        return dataset

    def _get_image_object(self, image_id: int) -> omero.gateway.ImageWrapper:
        image_object = self.conn.getObject('Image', image_id)
        return image_object

    def get_annotation_image_ids(self) -> list:
        input_omero = self.params['input'].get('omero', {})
        include = input_omero['include']
        exclude = input_omero.get('exclude', {})
        target_labels = ensure_list(input_omero.get('labels', []))
        # include
        image_ids = set(ensure_list(include.get('image', [])))
        for dataset_id in ensure_list(include.get('dataset', [])):
            image_ids.update(self._get_dataset_annotation_image_ids(dataset_id, target_labels))
        for project_id in ensure_list(include.get('project', [])):
            project = self._get_project(project_id)
            for dataset in project.listChildren():
                image_ids.update(self._get_dataset_annotation_image_ids(dataset.getId(), target_labels))
        # exclude
        for id in ensure_list(exclude.get('image', [])):
            if id in image_ids:
                image_ids.remove(id)
        for dataset_id in ensure_list(exclude.get('dataset', [])):
            for id in self._get_dataset_annotation_image_ids(dataset_id):
                if id in image_ids:
                    image_ids.remove(id)
        for project_id in ensure_list(exclude.get('project', [])):
            project = self._get_project(project_id)
            for dataset in project.listChildren():
                for id in self._get_dataset_annotation_image_ids(dataset.getId()):
                    if id in image_ids:
                        image_ids.remove(id)
        return list(image_ids)

    def _get_dataset_annotation_image_ids(self, dataset_id: int, target_labels: list = []) -> list:
        dataset = self._get_dataset(dataset_id)
        image_ids = []
        for image_object in dataset.listChildren():
            annotations = self._get_image_annotations(image_object, target_labels)
            if len(annotations) == len(target_labels):
                image_ids.append(image_object.getId())
        return image_ids

    def _get_image_annotation(self, image_id: int, target_labels: list) -> tuple:
        image_object = self._get_image_object(image_id)
        name = image_object.getName()
        annotations = self._get_image_annotations(image_object, target_labels)
        return name, annotations

    def get_image_info(self, image_id: int) -> str:
        image_object = self._get_image_object(image_id)
        xyzct = self._get_size(image_object)
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

    def extract_thumbnail(self, image_id: int, outpath: str, target_size: float, overwrite: bool = True):
        image_object = self._get_image_object(image_id)
        filetitle = image_object.getName() + '.jpg'
        output_filename = os.path.join(outpath, filetitle)
        if overwrite or not os.path.exists(output_filename):
            w, h, zs, cs, ts = self._get_size(image_object)
            size = w, h

            if target_size < 1:
                factor = target_size
            else:
                factor = np.max(np.divide(size, target_size))
            thumb_size = np.round(np.divide(size, factor)).astype(int)
            try:
                thumb_bytes = image_object.getThumbnail(thumb_size)
                with open(output_filename, 'wb') as file:
                    file.write(thumb_bytes)
            except Exception as e:
                logging.error(e)

    def convert_images(self, image_ids: list, outpath: str, overwrite: bool = True):
        # currently only ome-tiff supported
        self._convert_images_to_ometiff(image_ids, outpath, overwrite=overwrite)

    def _convert_images_to_ometiff(self, image_ids: list, outpath: str, overwrite: bool = True):
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for image_id in tqdm(image_ids):
            self._convert_image_to_ometiff(image_id, outpath, overwrite=overwrite)

    def _convert_image_to_ometiff(self, image_id: int, outpath: str, overwrite: bool = True):
        output = self.params['output']
        image_object = self._get_image_object(image_id)
        filetitle = image_object.getName() + '.ome.tiff'
        output_filename = os.path.join(outpath, filetitle)
        if overwrite or not os.path.exists(output_filename):
            xyzct = self._get_size(image_object)
            w, h, zs, cs, ts = xyzct
            logging.info(f'{image_id} {image_object.getName()}')

            npyramid_add = output.get('npyramid_add', 0)
            pyramid_downsample = output.get('pyramid_downsample', 0)
            pyramid_sizes_add = calc_pyramid((w, h), npyramid_add, pyramid_downsample)
            metadata = create_ome_metadata_from_omero(image_object, filetitle, pyramid_sizes_add)
            xml_metadata = metadata.to_xml()

            image = self._get_omero_image(image_object)
            if image is not None:
                save_tiff(output_filename, image, xml_metadata=xml_metadata,
                          tile_size=output.get('tile_size'),
                          compression=output.get('compression'),
                          pyramid_sizes_add=pyramid_sizes_add)

    def _get_omero_image0(self, image_object: omero.gateway.ImageWrapper, pixels: omero.gateway.PixelsWrapper) -> np.ndarray:
        w, h, zs, cs, ts = self._get_size(image_object)
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
        return image

    def _get_omero_image(self, image_object: omero.gateway.ImageWrapper, read_size: int = 10240) -> np.ndarray:
        w, h, zs, cs, ts = self._get_size(image_object)
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

    def _get_size(self, image_object: omero.gateway.ImageWrapper) -> tuple:
        xs, ys = image_object.getSizeX(), image_object.getSizeY()
        zs, cs, ts = image_object.getSizeZ(), image_object.getSizeC(), image_object.getSizeT()
        return xs, ys, zs, cs, ts

    def _get_original_files(self, image_object: omero.gateway.ImageWrapper) -> list:
        return image_object.getFileset().listFiles()

    def _get_magnification(self, image_object: omero.gateway.ImageWrapper) -> float:
        return image_object.getObjectiveSettings().getObjective().getNominalMagnification()

    def _get_image_annotations(self, image_object: omero.gateway.ImageWrapper, annotation_keys: list) -> dict:
        annotations = {}
        for omero_annotation in image_object.listAnnotations():
            if omero_annotation.OMERO_TYPE == omero.model.MapAnnotationI:
                for annotation_key in annotation_keys:
                    for annotation in omero_annotation.getMapValue():
                        if annotation.name.lower() == annotation_key.lower():
                            annotations[annotation_key] = annotation.value
        return annotations

    def print_projects(self):
        projects = self.conn.listProjects()      # may include other users' data
        for project in projects:
            print_omero_object(project)


def print_omero_object(omero_object: omero.gateway.BlitzObjectWrapper, indent: int = 0):
    logging.info("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        omero_object.OMERO_CLASS,
        omero_object.getId(),
        omero_object.getName(),
        omero_object.getOwnerOmeName()))

    for child in omero_object.listChildren():
        logging.info('\t', child.getName())
