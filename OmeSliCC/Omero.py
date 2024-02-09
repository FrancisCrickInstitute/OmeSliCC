from __future__ import annotations
import logging
import omero
from omero.gateway import BlitzGateway
import omero.model
from types import TracebackType
import re

from OmeSliCC.omero_credentials import decrypt_credentials
from OmeSliCC.util import *


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
            logging.error('Omero connection error')
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

    def get_image_object(self, image_id: int) -> omero.gateway.ImageWrapper:
        image_object = self.conn.getObject('Image', image_id)
        return image_object

    def create_pixels_store(self, image_object: omero.gateway.ImageWrapper) -> omero.gateway.ProxyObjectWrapper:
        pixels_store = self.conn.createRawPixelsStore()
        pixels_store.setPixelsId(image_object.getPixelsId(), False, self.conn.SERVICE_OPTS)
        return pixels_store

    def get_annotation_image_ids(self) -> dict:
        images_final = {}
        input_omero = self.params['input'].get('omero', {})
        include_params = input_omero['include']
        include_regex = ensure_list(include_params.get('regex', []))
        exclude_params = input_omero.get('exclude', {})
        exclude_regex = ensure_list(exclude_params.get('regex', []))
        # include
        image_ids = set(ensure_list(include_params.get('image', [])))
        images = {image_id: self.get_image_object(image_id) for image_id in image_ids}
        for dataset_id in ensure_list(include_params.get('dataset', [])):
            images.update(self._get_dataset_images(dataset_id))
        for project_id in ensure_list(include_params.get('project', [])):
            project = self._get_project(project_id)
            for dataset in project.listChildren():
                images.update(self._get_dataset_images(dataset.getId()))
        # exclude
        for image_id in ensure_list(exclude_params.get('image', [])):
            images.pop(image_id, None)
        for dataset_id in ensure_list(exclude_params.get('dataset', [])):
            for image_id in self._get_dataset_images(dataset_id):
                images.pop(image_id, None)
        for project_id in ensure_list(exclude_params.get('project', [])):
            project = self._get_project(project_id)
            for dataset in project.listChildren():
                for image_id in self._get_dataset_images(dataset.getId()):
                    images.pop(image_id, None)

        # regex
        for image_id, image in images.items():
            name = image.getName()
            include = True
            if include_regex:
                include = False
                for pattern in include_regex:
                    if re.search(pattern, name, re.IGNORECASE):
                        include = True
            if exclude_regex:
                for pattern in exclude_regex:
                    if re.search(pattern, name, re.IGNORECASE):
                        include = False
            if include:
                images_final[image_id] = image
        return images_final

    def _get_dataset_images(self, dataset_id: int) -> dict:
        dataset = self._get_dataset(dataset_id)
        return {image.getId(): image for image in dataset.listChildren()}

    def get_image_annotation(self, image_id: int, target_labels: list) -> tuple:
        image_object = self.get_image_object(image_id)
        name = image_object.getName()
        annotations = self._get_image_annotations(image_object, target_labels)
        return name, annotations

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
