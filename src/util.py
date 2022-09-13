import imagecodecs
import numpy as np
import os
import tifffile


def check_versions():
    print(f'tifffile {tifffile.__version__}')
    print(imagecodecs.version())


def ensure_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def tags_to_dict(tags):
    tag_dict = {}
    for tag in tags.values():
        tag_dict[tag.name] = tag.value
    return tag_dict


def print_dict(d, compact=False, indent=0):
    s = ''
    for key, value in d.items():
        if not isinstance(value, list):
            if not compact: s += '\t' * indent
            s += str(key)
            s += ':' if compact else '\n'
        if isinstance(value, dict):
            s += print_dict(value, indent=indent + 1)
        elif isinstance(value, list):
            for v in value:
                s += print_dict(v)
        else:
            if not compact: s += '\t' * (indent + 1)
            s += str(value)
        s += ' ' if compact else '\n'
    return s


def stringdict_to_dict(string_dict):
    metadata = {}
    if isinstance(string_dict, dict):
        dict_list = string_dict.items()
    else:
        dict_list = string_dict
    for key, value in dict_list:
        keys = key.split('|')
        add_dict_tree(metadata, keys, value)
    return metadata


def add_dict_tree( metadata, keys, value):
    key = keys[0]
    if len(keys) > 1:
        if key not in metadata:
            metadata[key] = {}
        add_dict_tree(metadata[key], keys[1:], value)
    else:
        metadata[key] = value


def print_hbytes(bytes):
    exps = ['', 'K', 'M', 'G', 'T']
    div = 1024
    exp = 0

    while bytes > div:
        bytes /= div
        exp +=1
    return f'{bytes:.1f}{exps[exp]}B'


def round_significants(a, significant_digits):
    if a != 0:
        round_decimals = significant_digits - int(np.floor(np.log10(abs(a)))) - 1
        return round(a, round_decimals)
    return a


def get_filetitle(filename, remove_all_ext=False):
    filebase = os.path.basename(filename)
    if remove_all_ext:
        return filebase.split('.')[0]
    else:
        return os.path.splitext(filebase)[0]
