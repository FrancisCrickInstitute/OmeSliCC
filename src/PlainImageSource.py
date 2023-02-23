import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from src.OmeSource import OmeSource
from src.image_util import pilmode_to_pixelinfo, get_pil_metadata

Image.MAX_IMAGE_PIXELS = None   # avoid DecompressionBombError (which prevents loading large images)


class PlainImageSource(OmeSource):
    """Plain common format image source"""

    filename: str
    """original filename"""
    loaded: bool
    """if image data is loaded"""
    arrays: list
    """list of all image arrays for different sizes"""

    def __init__(self,
                 filename: str,
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False,
                 executor: ThreadPoolExecutor = None):

        super().__init__()
        self.loaded = False
        self.arrays = []

        if executor is not None:
            self.executor = executor
        else:
            max_workers = (os.cpu_count() or 1) + 4
            self.executor = ThreadPoolExecutor(max_workers)

        self.image = Image.open(filename)
        self.metadata = get_pil_metadata(self.image)
        size = (self.image.width, self.image.height)
        self.sizes = [size]
        size_xyzct = (self.image.width, self.image.height, self.image.n_frames, len(self.image.getbands()), 1)
        self.sizes_xyzct = [size_xyzct]
        pixelinfo = pilmode_to_pixelinfo(self.image.mode)
        self.pixel_types = [pixelinfo[0]]
        self.pixel_nbits = [pixelinfo[1]]

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

    def _find_metadata(self):
        self.pixel_size = []
        pixel_size_unit = self.metadata.get('unit', '')
        res0 = self.metadata.get('XResolution', 1)
        if isinstance(res0, tuple):
            res0 = res0[0] / res0[1]
        self.pixel_size.append((1 / res0, pixel_size_unit))
        res0 = self.metadata.get('YResolution', 1)
        if isinstance(res0, tuple):
            res0 = res0[0] / res0[1]
        self.pixel_size.append((1 / res0, pixel_size_unit))
        self.channel_info = []
        channels = self.image.getbands()
        nchannels = len(channels)
        for channel in channels:
            self.channel_info.append((channel, self.pixel_nbits[0] // 8 // nchannels))
        self.source_mag = self.metadata.get('Mag', 0)

    def load(self):
        self.unload()
        self.arrays.append(np.array(self.image))
        self.loaded = True

    def unload(self):
        for array in self.arrays:
            del array
        self.arrays = []
        self.loaded = False

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        if x1 < 0 or y1 < 0:
            x1, y1 = self.sizes[level]

        if self.loaded:
            array = self.arrays[level]
        else:
            array = np.array(self.image)
        return array[y0:y1, x0:x1]
