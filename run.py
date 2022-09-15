import argparse
import glob
import logging
import os
import yaml
from tqdm import tqdm

from src.Omero import Omero
from src.OmeroLabelReader import OmeroLabelReader
from src.conversion import get_image_info, extract_thumbnail, convert_slide
from src.image_util import check_versions
from src.util import ensure_list
from src.parameters import *


def run_actions(params):
    input = params['input']
    output = params['output']
    type = input['type'].lower()
    ids = input.get('ids')
    output_folder = output['folder']

    actions = params.get('actions')

    if 'omero' in input:
        with Omero(params) as omero:
            if 'project' in type:
                input_labels = input['labels']
                image_ids = []
                for proj_id in ensure_list(ids):
                    image_ids.extend(omero.get_annotation_image_ids(proj_id, input_labels)[0])
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
                    omero.convert_slides(image_ids, output_folder)
                elif 'label' in action:
                    with OmeroLabelReader(params, omero=omero) as label_reader:
                        label_reader.create_label_csv()
                logging.info(f'Done {action}')
    else:
        input_folder = input['folder']
        filenames = []
        if ids is not None:
            filenames = [os.path.join(input_folder, id) for id in ids]
        if len(filenames) == 0:
            input_path = input_folder
            if os.path.isdir(input_path):
                input_path = os.path.join(input_path, '*')
            filenames = [file for file in glob.glob(input_path) if os.path.isfile(file)]
        if len(filenames) > 0:
            for action0 in actions:
                action = action0.lower()
                logging.info(f'Starting {action}')
                for filename in tqdm(filenames):
                    try:
                        if 'info' in action:
                            get_image_info(filename)
                        elif 'thumb' in action:
                            extract_thumbnail(filename, output_folder)
                        elif 'convert' in action:
                            convert_slide(filename, output)
                    except Exception as e:
                        logging.exception(e)
                logging.info(f'Done {action}')
        else:
            logging.warning('No files to process')

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
