import ast
import os
import numpy as np


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


def desc_to_dict(desc):
    desc_dict = {}
    sep = ''
    if desc.startswith('{'):
        try:
            metadata = ast.literal_eval(desc)
            return metadata
        except:
            pass
    if '\n' in desc:
        sep = '\n'
    elif '\t' in desc:
        sep = '\t'
    elif '|' in desc:
        sep = '|'
    elif ',' in desc:
        sep = ','
    for item in desc.split(sep):
        item_sep = '='
        if ':' in item:
            item_sep = ':'
        items = item.split(item_sep)
        key = items[0]
        value = items[1]
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except:
            try:
                value = bool(value)
            except:
                pass
        desc_dict[key] = value
    return desc_dict


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


def print_hbytes(bytes):
    exps = ['', 'K', 'M', 'G', 'T']
    div = 1024
    exp = 0

    while bytes > div:
        bytes /= div
        exp +=1
    return f'{bytes:.1f}{exps[exp]}B'


def check_round_significants(a, significant_digits):
    rounded = round_significants(a, significant_digits)
    if a != 0:
        dif = 1 - rounded / a
    else:
        dif = rounded - a
    if abs(dif) < 10 ** -significant_digits:
        return rounded
    return a


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
