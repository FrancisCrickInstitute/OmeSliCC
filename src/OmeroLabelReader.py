from __future__ import annotations
import logging
import os
import pandas as pd
from types import TracebackType

from src.Omero import Omero


class OmeroLabelReader:
    """Omero metadata extraction to label file"""

    params: dict
    """input parameters"""
    omero: Omero
    """Omero instance"""
    manage_omero: bool
    """If responsible for managing Omero instance"""

    def __init__(self, params: dict, omero: Omero = None):
        self.params = params
        self.manage_omero = (omero is None)
        if self.manage_omero:
            self.omero = Omero(params)
        else:
            self.omero = omero

    def __enter__(self) -> OmeroLabelReader:
        if self.manage_omero:
            self.omero.init()
        return self

    def __exit__(self, exc_type: type[BaseException], exc_value: BaseException, traceback: TracebackType):
        if self.manage_omero:
            self.omero.close()

    def create_label_csv(self, image_ids):
        image_names = []
        image_annotations = []
        input = self.params['input']
        output = self.params['output']
        input_labels = input.get('omero', {}).get('labels', [])
        logging.info(f'Matching images: {len(image_ids)}')
        for id in image_ids:
            name, annotations = self.omero._get_image_annotation(id, input_labels)
            image_names.append(name)
            image_annotations.append(annotations)
        df = pd.DataFrame(index=image_ids, data=image_annotations)
        df.index.name = 'omero_id'
        for input_label in input_labels:
            if input_label in df:
                logging.info(f'Label {input_label}:\n' + df[input_label].value_counts().to_string())
        df.insert(0, 'omero_name', image_names)
        df['path'] = [image_name + '.' + output['format'] for image_name in image_names]
        log_path = os.path.dirname(output['csv'])
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        df.to_csv(output['csv'])
