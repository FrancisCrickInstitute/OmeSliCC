from getpass import getpass
import rsa


def generate_asymmetric_keys(private_key_filename, public_key_filename):
    pub_key, pri_key = rsa.newkeys(2048)
    with open(public_key_filename, 'w') as pub_file, open(private_key_filename, 'w') as pri_file:
        pub_file.write(pub_key.save_pkcs1().decode())
        pri_file.write(pri_key.save_pkcs1().decode())


def create_credentials_file(public_key_filename, credentials_filename):
    with open(public_key_filename, 'r') as key_file, open(credentials_filename, 'wb') as enc_file:
        keydata = key_file.read().encode()
        pub_key = rsa.PublicKey.load_pkcs1(keydata)
        enc_cred = rsa.encrypt((input('Username: ') + '\t' + getpass('Password: ')).encode(), pub_key)
        enc_file.write(enc_cred)
        print('credentials file generated')


def decrypt_credentials(private_key_filename, credentials_filename):
    with open(credentials_filename, 'rb') as cred_f, open(private_key_filename, 'rb') as key_file:
        keydata = key_file.read()
        pri_key = rsa.PrivateKey.load_pkcs1(keydata)
        cred = cred_f.read()
        dec_cred = rsa.decrypt(cred, pri_key).decode().split()
        usr = dec_cred[0]
        pwd = dec_cred[1]
    return usr, pwd
