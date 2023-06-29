import argparse
import glob
import logging
import os
import toml
import validators
import yaml
from tqdm import tqdm

from OmeSliCC.Omero import Omero
from OmeSliCC.conversion import *
from OmeSliCC.util import *
from OmeSliCC.parameters import *


name = toml.load("pyproject.toml")["project"]["name"]
version = toml.load("pyproject.toml")["project"]["version"]


def run_actions(params: dict):
    """
    Run OmeSliCC pipeline using parameters
    See resources/params.yml for parameter details
    params: parameters defining pipeline input, output and actions
    """
    input_params = params['input']
    source_ref = input_params.get('source')
    actions = ensure_list(params.get('actions'))
    is_omero = input_params.get('omero') is not None
    multi_file = ('combine' in actions)
    omero = None

    if is_omero:
        omero = Omero(params)
        omero.init()
        source_refs = omero.get_annotation_image_ids()
    elif isinstance(source_ref, list):
        # list of filenames
        source_refs = source_ref
        if multi_file:
            source_refs = [sorted(glob.glob(source_ref1)) for source_ref1 in source_ref]
    else:
        if validators.url(source_ref):
            # URL
            source_refs = [source_ref]
        else:
            # filename or path
            input_path = source_ref
            if os.path.isdir(input_path) and not input_path.lower().endswith('.zarr'):
                input_path = os.path.join(input_path, '*')
            source_refs = [file for file in glob.glob(input_path) if os.path.isfile(file) or file.lower().endswith('.zarr')]
    if len(source_refs) > 0:
        for action0 in actions:
            action = action0.lower()
            logging.info(f'Starting {action}')
            if multi_file:
                it = zip(*source_refs)
            else:
                it = source_refs
            for source_ref in tqdm(it):
                try:
                    if multi_file:
                        sources = [create_source(str(source_ref1), params, omero) for source_ref1 in source_ref]
                        source = sources[0]
                    else:
                        source = create_source(str(source_ref), params, omero)
                    if 'info' in action:
                        logging.info(get_image_info(source))
                    elif 'thumb' in action:
                        extract_thumbnail(source, params)
                    elif 'convert' in action:
                        print(str(source_ref))
                        convert_image(source, params)
                    elif 'combine' in action:
                        combine_images(sources, params)
                    source.close()
                except Exception as e:
                    logging.exception(e)
            logging.info(f'Done {action}')
    else:
        logging.warning('No files to process')
    if is_omero:
        omero.close()

    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{name} {version}')
    parser.add_argument('--params',
                        help='The parameters file',
                        default=PARAMETER_FILE)

    args = parser.parse_args()
    with open(args.params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    log_params = params['log']
    basepath = os.path.dirname(log_params['filename'])
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    logging.basicConfig(level=logging.INFO, format=log_params['log_format'],
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_params['filename'])])

    run_actions(params)
