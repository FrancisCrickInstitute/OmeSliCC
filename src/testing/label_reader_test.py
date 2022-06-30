import os
import yaml

from src.OmeroLabelReader import OmeroLabelReader
from src.parameters import PARAMETER_FILE


if __name__ == '__main__':
    os.chdir('../../')
    params = yaml.safe_load(PARAMETER_FILE)
    with OmeroLabelReader(params) as label_reader:
        label_reader.create_label_csv()
