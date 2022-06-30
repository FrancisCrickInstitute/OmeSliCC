import argparse
import yaml

from src.omero_credentials import *
from src.parameters import PARAMETER_FILE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode Omero credentials')
    parser.add_argument('--params',
                        help='The location of the parameters file',
                        default=PARAMETER_FILE)
    with open(PARAMETER_FILE, 'r') as file:
        params = yaml.safe_load(file)

    params_cred = params['credentials']
    private_key_filename = params_cred['private_key']
    public_key_filename = params_cred['public_key']
    credentials_filename = params_cred['credentials']

    generate_asymmetric_keys(private_key_filename, public_key_filename)
    create_credentials_file(public_key_filename, credentials_filename)
    # test/decrypt:
    print(decrypt_credentials(private_key_filename, credentials_filename))
