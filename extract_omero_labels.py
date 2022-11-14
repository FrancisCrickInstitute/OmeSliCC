import argparse
import logging
import os
import yaml

from src.Omero import Omero
from src.OmeroLabelReader import OmeroLabelReader
from src.parameters import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Omero label reader')
    parser.add_argument('--params',
                        help='The parameters file',
                        default=PARAMETER_FILE)
    args = parser.parse_args()
    with open(args.params) as file:
        params = yaml.safe_load(file)

    log_params = params['log']
    basepath = os.path.dirname(log_params['filename'])
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    logging.basicConfig(format=log_params['log_format'], level=logging.INFO,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_params['filename'])])

    with Omero(params) as omero:
        with OmeroLabelReader(params, omero=omero) as label_reader:
            image_ids = omero.get_annotation_image_ids()
            label_reader.create_label_csv(image_ids)
    logging.info('Done')
