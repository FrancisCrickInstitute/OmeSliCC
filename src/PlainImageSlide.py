import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from src.OmeSlide import OmeSlide
from src.image_util import precise_resize, pilmode_to_pixelsize, pil_resize

Image.MAX_IMAGE_PIXELS = None   # avoid DecompressionBombError (which prevents loading large images)


class PlainImageSlide(OmeSlide):
    def __init__(self, filename, source_mag=None, target_mag=None, executor=None):
        if target_mag is not None and source_mag is None:
            raise ValueError(f'Error: Provide source magnification (in parameter file) for images without meta-data')
        if executor is not None:
            self.executor = executor
        else:
            max_workers = (os.cpu_count() or 1) + 4
            self.executor = ThreadPoolExecutor(max_workers)
        self.loaded = False
        self.data = None
        self.arrays = []
        self.image = Image.open(filename)
        self.size = (self.image.width, self.image.height)
        self.sizes = [self.size]
        self.size_xyzct = (self.image.width, self.image.height, self.image.n_frames, len(self.image.getbands()), 1)
        self.sizes_xyzct = [self.size_xyzct]
        self.pixel_nbytes = [pilmode_to_pixelsize(self.image.mode)]
        self.source_mag = source_mag
        if source_mag is not None and target_mag is not None:
            self.mag_factor = source_mag / target_mag
        else:
            self.mag_factor = 1

    def load(self):
        self.unload()
        self.arrays.append(np.array(self.image))
        self.loaded = True

    def unload(self):
        for array in self.arrays:
            del array
        self.arrays = []
        self.loaded = False

    def get_size(self):
        # size at selected magnification
        return np.divide(self.size, self.mag_factor).astype(int)

    def get_thumbnail(self, target_size, precise=False):
        if precise:
            scale = target_size / self.size
            return precise_resize(np.array(self.image), scale)
        else:
            return pil_resize(self.image, target_size)


    def asarray_level(self, level, x0, y0, x1, y1):
        if self.loaded:
            array = self.arrays[level]
        else:
            array = np.array(self.image)
        return array[y0:y1, x0:x1]

    def get_max_mag(self):
        return self.source_mag
