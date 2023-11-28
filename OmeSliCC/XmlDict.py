from __future__ import annotations
import xmltodict


class XmlDict(dict):
    """xmltodict type dictionary providing and propagating access to keys without @ sign"""

    def __getitem__(self, key: str):
        at_key = '@' + key
        if key not in self and at_key in self:
            key = at_key
        value = dict.__getitem__(self, key)
        if isinstance(value, dict):
            value = XmlDict(value)
        if isinstance(value, list):
            value = XmlList(value)
        return value

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def copy(self) -> XmlDict:
        return XmlDict(dict.copy(self))


class XmlList(list):
    """xmltodict type list propagating access to keys without @ sign"""

    def __getitem__(self, index: int):
        value = list.__getitem__(self, index)
        if isinstance(value, dict):
            value = XmlDict(value)
        if isinstance(value, list):
            value = XmlList(value)
        return value

    def __iter__(self) -> XmlList:
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self):
            item = self[self.i]
            self.i += 1
            return item
        else:
            raise StopIteration


def xml2dict(xml_metadata: str) -> dict:
    return XmlDict(xmltodict.parse(xml_metadata))


def dict2xml(dct: dict) -> str:
    return xmltodict.unparse(dct, short_empty_elements=True, pretty=True)


if __name__ == '__main__':
    dct = {'@a': 1, 'b': {'@aa': 2, 'bb': 3}}
    test = XmlDict(dct)
    print(test.get('a', 11))
    print(test['a'])
    print(test.get('b').get('aa'))
    print(test['b']['aa'])

    xml = xmltodict.unparse({'TEST': test}, short_empty_elements=True, pretty=True)
    print(xml)
