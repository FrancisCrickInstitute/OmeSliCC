import argparse
import logging
import os
import yaml

from src.Omero import Omero
from src.OmeroLabelReader import OmeroLabelReader
from src.parameters import *
from src.util import ensure_list, check_versions


def run_actions(params):
    input = params['input']
    output = params['output']
    type = input['type'].lower()
    ids = input['ids']
    output_folder = output['folder']

    actions = params.get('actions')

    with Omero(params) as omero:
        if 'project' in type:
            target_annotations = input['gene']
            image_ids = []
            for proj_id in ensure_list(ids):
                image_ids.extend(omero.get_annotation_image_ids(proj_id, target_annotations)[0])
        elif 'image' in type:
            image_ids = ensure_list(ids)
        else:
            image_ids = []

        for action0 in actions:
            action = action0.lower()
            logging.info(f'Starting {action}')
            if 'info' in action:
                for id in image_ids:
                    omero.get_image_info(id)
            elif 'thumb' in action:
                for id in image_ids:
                    omero.extract_thumbnail(id, output_folder)
            elif 'convert' in action:
                omero.convert_slides_to_tiff(image_ids, output_folder)
            elif 'label' in action:
                with OmeroLabelReader(params, omero=omero) as label_reader:
                    label_reader.create_label_csv()
            logging.info(f'Done {action}')

    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OmeSliCC')
    parser.add_argument('--params',
                        help='The location of the parameters file',
                        default=PARAMETER_FILE)

    #check_versions()

    args = parser.parse_args()
    with open(args.params, 'r') as file:
        params = yaml.safe_load(file)

    log_params = params['log']
    basepath = os.path.dirname(log_params['filename'])
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    logging.basicConfig(level=logging.INFO, format=log_params['log_format'],
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_params['filename'])])

    run_actions(params)
