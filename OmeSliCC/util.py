import ast
from datetime import datetime
import json
import os
import re
import numpy as np
import xmltodict
import yaml


def get_default(x, default):
    return default if x is None else x


def ensure_list(x) -> list:
    if x is None:
        return []
    elif isinstance(x, list):
        return x
    else:
        return [x]


def reorder(items: list, old_order: str, new_order: str, default_value: int = 0) -> list:
    new_items = []
    for label in new_order:
        if label in old_order:
            item = items[old_order.index(label)]
        else:
            item = default_value
        new_items.append(item)
    return new_items


def file_to_dict(filename: str) -> dict:
    ext = os.path.splitext(filename)[1]
    content = open(filename, 'r').read()
    if ext == '.xml':
        data = xmltodict.parse(content)
    elif ext in ['.yml', '.yaml']:
        data = yaml.safe_load(content)
    else:   # assume json
        data = json.loads(content)
    return data


def filter_dict(dict0: dict) -> dict:
    new_dict = {}
    for key, value0 in dict0.items():
        if value0 is not None:
            values = []
            for value in ensure_list(value0):
                if isinstance(value, dict):
                    value = filter_dict(value)
                values.append(value)
            if len(values) == 1:
                values = values[0]
            new_dict[key] = values
    return new_dict


def desc_to_dict(desc: str) -> dict:
    desc_dict = {}
    if desc.startswith('{'):
        try:
            metadata = ast.literal_eval(desc)
            return metadata
        except:
            pass
    for item in re.split(r'[\r\n\t|]', desc):
        item_sep = '='
        if ':' in item:
            item_sep = ':'
        if item_sep in item:
            items = item.split(item_sep)
            key = items[0].strip()
            value = items[1].strip()
            for dtype in (int, float, bool):
                try:
                    value = dtype(value)
                    break
                except:
                    pass
            desc_dict[key] = value
    return desc_dict


def map_dict(metadata0: dict, mappings: list) -> dict:
    metadata = {}
    for mapping in mappings:
        dest_key = mapping[0]
        source_key = mapping[1]
        def_value = mapping[2] if len(mapping) > 2 else None
        value = get_dict_value(metadata0, source_key, def_value)
        if len(mapping) > 3:
            reformat = mapping[3]
            if 'date' in dest_key.lower() or 'time' in dest_key.lower():
                try:
                    value = datetime.fromtimestamp(float(value))
                except ValueError:
                    pass
            if '%' in reformat:
                reformat.format(value)
            else:
                value = eval(f'value.{reformat}')
        if value is not None:
            metadata[dest_key] = value
    return unpack_dict(metadata)


def get_dict_value(dct: dict, target_key: str, def_value=None) -> dict:
    keys = target_key.split('.')
    key = keys[0]
    value = dct.get(key, def_value)
    if value is not None and len(keys) > 1:
        return get_dict_value(value, '.'.join(keys[1:]), def_value)
    else:
        return value


def unpack_dict(dct: dict) -> dict:
    # Convert flat dict with . separator to nested dict
    new_dct = {}
    for key, value in dct.items():
        if '.' in key:
            keys = key.split('.')
            sub_dct = new_dct
            for sub_key in keys[:-1]:
                if sub_key not in sub_dct:
                    sub_dct[sub_key] = {}
                sub_dct = sub_dct[sub_key]
            sub_dct[keys[-1]] = value
        else:
            new_dct[key] = value
    return new_dct


def print_dict(dct: dict, indent: int = 0) -> str:
    s = ''
    if isinstance(dct, dict):
        for key, value in dct.items():
            s += '\n'
            if not isinstance(value, list):
                s += '\t' * indent + str(key) + ': '
            if isinstance(value, dict):
                s += print_dict(value, indent=indent + 1)
            elif isinstance(value, list):
                for v in value:
                    s += print_dict(v)
            else:
                s += str(value)
    else:
        s += str(dct)
    return s


def print_hbytes(nbytes: int) -> str:
    exps = ['', 'K', 'M', 'G', 'T']
    div = 1024
    exp = 0

    while nbytes > div:
        nbytes /= div
        exp += 1
    return f'{nbytes:.1f}{exps[exp]}B'


def check_round_significants(a: float, significant_digits: int) -> float:
    rounded = round_significants(a, significant_digits)
    if a != 0:
        dif = 1 - rounded / a
    else:
        dif = rounded - a
    if abs(dif) < 10 ** -significant_digits:
        return rounded
    return a


def round_significants(a: float, significant_digits: int) -> float:
    if a != 0:
        round_decimals = significant_digits - int(np.floor(np.log10(abs(a)))) - 1
        return round(a, round_decimals)
    return a


def get_filetitle(filename: str) -> str:
    filebase = os.path.basename(filename)
    title = os.path.splitext(filebase)[0].rstrip('.ome')
    return title


def split_num_text(text: str) -> list:
    num_texts = []
    block = ''
    is_num0 = None
    if text is None:
        return None

    for c in text:
        is_num = (c.isnumeric() or c == '.')
        if is_num0 is not None and is_num != is_num0:
            num_texts.append(block)
            block = ''
        block += c
        is_num0 = is_num
    if block != '':
        num_texts.append(block)

    num_texts2 = []
    for block in num_texts:
        block = block.strip()
        try:
            block = float(block)
        except:
            pass
        if block not in [' ', ',', '|']:
            num_texts2.append(block)
    return num_texts2


def split_value_unit_list(text: str) -> list:
    value_units = []
    if text is None:
        return None

    items = split_num_text(text)
    if isinstance(items[-1], str):
        def_unit = items[-1]
    else:
        def_unit = ''

    i = 0
    while i < len(items):
        value = items[i]
        if i + 1 < len(items):
            unit = items[i + 1]
        else:
            unit = ''
        if not isinstance(value, str):
            if isinstance(unit, str):
                i += 1
            else:
                unit = def_unit
            value_units.append((value, unit))
        i += 1
    return value_units


def get_value_units_micrometer(value_units0: list) -> list:
    conversions = {
        'nm': 1e-3,
        'µm': 1, 'um': 1, 'micrometer': 1,
        'mm': 1e3, 'millimeter': 1e3,
        'cm': 1e4, 'centimeter': 1e4,
        'm': 1e6, 'meter': 1e6
    }
    if value_units0 is None:
        return None

    values_um = []
    for value_unit in value_units0:
        if isinstance(value_unit, (list, tuple)):
            value_um = value_unit[0] * conversions.get(value_unit[1], 1)
        else:
            value_um = value_unit
        values_um.append(value_um)
    return values_um


def convert_rational_value(value) -> float:
    if value is not None and isinstance(value, tuple):
        if value[0] == value[1]:
            value = value[0]
        else:
            value = value[0] / value[1]
    return value


def tile_to_chunk_size(tile_size, ndims):
    if isinstance(tile_size, int):
        chunk_size = [tile_size] * 2
    else:
        chunk_size = list(reversed(tile_size))
    while len(chunk_size) < ndims:
        chunk_size = [1] + chunk_size
    return chunk_size
