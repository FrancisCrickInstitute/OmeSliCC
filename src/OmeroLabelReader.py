import logging
import pandas as pd

from src.Omero import Omero


class OmeroLabelReader:
    def __init__(self, params, omero=None):
        self.params = params
        self.manage_omero = (omero is None)
        if self.manage_omero:
            self.omero = Omero(params)
        else:
            self.omero = omero

    def __enter__(self):
        if self.manage_omero:
            self.omero.init()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.manage_omero:
            self.omero.disconnect()

    def create_label_csv(self):
        input = self.params['input']
        output = self.params['output']
        if input['type'] == 'project':
            project_id = input['ids']
        else:
            logging.error("Please use project type in params file")
            project_id = -1
        genes_list = input['gene']
        image_ids, image_names, image_annotations = self.omero.get_annotation_image_ids(project_id, genes_list)
        logging.info(f'Matching images found: {len(image_ids)}')
        df = pd.DataFrame(index=image_ids, data=image_annotations)
        df.index.name = 'omero_id'
        for gene in genes_list:
            if gene in df:
                logging.info(f'Gene {gene}:\n' + df[gene].value_counts().to_string())
        df.insert(0, 'omero_name', image_names)
        df['path'] = [image_name + '.' + output['format'] for image_name in image_names]
        df.to_csv(output['csv'])
