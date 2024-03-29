from enum import Enum


class ChannelOperation(Enum):
    NONE = 0
    COMBINE = 1
    SPLIT = 2


RESOURCE_DIR = 'resources/'

PARAMETER_FILE = RESOURCE_DIR + 'params.yml'

IMAGE_DIR = RESOURCE_DIR + 'images/'
