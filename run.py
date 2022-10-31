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


def run_actions(params: dict):
    """
    Run OmeSliCC pipeline using parameters
    See resources/params.yml for parameter details
    params: parameters defining pipeline input, output and actions
    """
    input_params = params['input']
    output_params = params['output']
    omero_type = input_params['omero_type'].lower()
    omero_ids = input_params.get('omero_ids')
    output_folder = output_params['folder']
    thumbnail_size = output_params.get('thumbnail_size', 1000)
    overwrite = output_params.get('overwrite', True)

    actions = ensure_list(params.get('actions'))

    if 'omero' in input_params:
        with Omero(params) as omero:
            if 'project' in omero_type:
                omero_labels = input_params.get('omero_labels', [])
                image_ids = []
                for proj_id in ensure_list(omero_ids):
                    image_ids.extend(omero.get_annotation_image_ids(proj_id, omero_labels, filter_label_macro=True)[0])
            elif 'image' in omero_type:
                image_ids = ensure_list(omero_ids)
            else:
                image_ids = []

            for action0 in actions:
                action = action0.lower()
                logging.info(f'Starting {action}')
                if 'info' in action:
                    for image_id in image_ids:
                        image_info = omero.get_image_info(image_id)
                        logging.info(image_info)
                elif 'thumb' in action:
                    for image_id in image_ids:
                        omero.extract_thumbnail(image_id, output_folder, target_size=thumbnail_size, overwrite=overwrite)
                elif 'convert' in action:
                    omero.convert_images(image_ids, output_folder, overwrite=overwrite)
                elif 'label' in action:
                    with OmeroLabelReader(params, omero=omero) as label_reader:
                        label_reader.create_label_csv()
                logging.info(f'Done {action}')
    else:
        input_source = input_params['source']
        if isinstance(input_source, list):
            sources = input_source
        else:
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
