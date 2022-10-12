import argparse
import glob
import logging
import os
import validators
import yaml
from tqdm import tqdm

from src.Omero import Omero
from src.OmeroLabelReader import OmeroLabelReader
from src.conversion import get_image_info, extract_thumbnail, convert
from src.util import ensure_list
from src.parameters import *
from version import __version__


def run_actions(params):
    input = params['input']
    output = params['output']
    type = input['type'].lower()
    ids = input.get('ids')
    output_folder = output['folder']

    actions = ensure_list(params.get('actions'))

    if 'omero' in input:
        with Omero(params) as omero:
            if 'project' in type:
                input_labels = input['labels']
                image_ids = []
                for proj_id in ensure_list(ids):
                    image_ids.extend(omero.get_annotation_image_ids(proj_id, input_labels, filter_label_macro=True)[0])
            elif 'image' in type:
                image_ids = ensure_list(ids)
            else:
                image_ids = []

            for action0 in actions:
                action = action0.lower()
                logging.info(f'Starting {action}')
                if 'info' in action:
                    for id in image_ids:
                        image_info = omero.get_image_info(id)
                        logging.info(image_info)
                elif 'thumb' in action:
                    for id in image_ids:
                        omero.extract_thumbnail(id, output_folder)
                elif 'convert' in action:
                    omero.convert_images(image_ids, output_folder)
                elif 'label' in action:
                    with OmeroLabelReader(params, omero=omero) as label_reader:
                        label_reader.create_label_csv()
                logging.info(f'Done {action}')
    else:
        input_source = input['source']
        sources = []
        if ids is not None:
            sources = [os.path.join(input_source, id) for id in ids]
        if len(sources) == 0:
            if validators.url(input_source):
                sources = [input_source]
            else:
                input_path = input_source
                if os.path.isdir(input_path) and not input_path.lower().endswith('zarr'):
                    input_path = os.path.join(input_path, '*')
                sources = [file for file in glob.glob(input_path) if os.path.isfile(file) or file.lower().endswith('zarr')]
        if len(sources) > 0:
            for action0 in actions:
                action = action0.lower()
                logging.info(f'Starting {action}')
                for source in tqdm(sources):
                    try:
                        if 'info' in action:
                            logging.info(get_image_info(source, params))
                        elif 'thumb' in action:
                            extract_thumbnail(source, params)
                        elif 'convert' in action:
                            convert(source, params)
                    except Exception as e:
                        logging.exception(e)
                logging.info(f'Done {action}')
        else:
            logging.warning('No files to process')

    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OmeSliCC ' + __version__)
    parser.add_argument('--params',
                        help='The parameters file',
                        default=PARAMETER_FILE)

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
