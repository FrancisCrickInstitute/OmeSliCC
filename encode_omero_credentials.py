import argparse
import yaml

from OmeSliCC.omero_credentials import *
from OmeSliCC.parameters import PARAMETER_FILE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode Omero credentials')
    parser.add_argument('--params',
                        help='The parameters file',
                        default=PARAMETER_FILE)

    args = parser.parse_args()
    with open(args.params, 'r') as file:
        params = yaml.safe_load(file)

    params_cred = params['credentials']
    private_key_filename = params_cred['private_key']
    public_key_filename = params_cred['public_key']
    credentials_filename = params_cred['credentials']

    generate_asymmetric_keys(private_key_filename, public_key_filename)
    create_credentials_file(public_key_filename, credentials_filename)
    # test/decrypt:
    print(decrypt_credentials(private_key_filename, credentials_filename))
