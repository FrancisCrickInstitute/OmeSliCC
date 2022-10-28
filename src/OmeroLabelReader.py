from __future__ import annotations
import logging
import os
import pandas as pd
from types import TracebackType

from src.Omero import Omero


class OmeroLabelReader:
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

    def create_label_csv(self):
        input = self.params['input']
        output = self.params['output']
        ids = input['omero_ids']
        if input['omero_type'] == 'project' and not isinstance(ids, list):
            project_id = ids
        else:
            logging.error("Label extraction only supports single project id")
            project_id = -1
        input_labels = input['omero_labels']
        image_ids, image_names, image_annotations = self.omero.get_annotation_image_ids(project_id, input_labels, filter_label_macro=True)
        logging.info(f'Matching images found: {len(image_ids)}')
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
