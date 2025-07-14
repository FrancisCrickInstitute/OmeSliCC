from OmeSliCC.XmlDict import XmlDict, XmlList


def dict_to_xmldict(values: dict) -> XmlDict:
    # convert dictionary to XmlDict, ensuring sub-dictionaries are also XmlDict
    if isinstance(values, dict):
        new_values = XmlDict()
        for key, value in values.items():
            new_values[key] = dict_to_xmldict(value)
    elif isinstance(values, list):
        new_values = XmlList([dict_to_xmldict(item) for item in values])
    else:
        new_values = values
    return new_values


def ensure_ome_id(element, target_key, id_label=None):
    element1 = element
    key = target_key
    keys = target_key.split('.')
    for key in keys:
        if key in element1:
            element1 = element1[key]
        else:
            return
    if id_label is None:
        id_label = key
    element1['@ID'] = f'{id_label}:0'
