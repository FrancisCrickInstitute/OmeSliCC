from __future__ import annotations
import logging
import omero
from omero.gateway import BlitzGateway
import omero.model
from types import TracebackType

from src.omero_credentials import decrypt_credentials
from src.util import ensure_list, get_default


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

    def create_pixels_store(self, image_object):
        pixels_store = self.conn.createRawPixelsStore()
        pixels_store.setPixelsId(image_object.getPixelsId(), False, self.conn.SERVICE_OPTS)
        return pixels_store

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
