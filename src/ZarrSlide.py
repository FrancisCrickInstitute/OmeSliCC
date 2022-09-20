import numpy as np
import zarr
from zarr.errors import GroupNotFoundError

from src.OmeSlide import OmeSlide


class ZarrSlide(OmeSlide):
    def __init__(self, filename, source_mag=None, target_mag=None):
        self.mag0 = source_mag
        self.target_mag = target_mag
        self.levels = []
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.pixel_nbytes = []

        try:
            root = zarr.open_group(filename, mode='r')
            self.metadata = root.info
            for level in root.array_keys():
                data = root.get(str(level))
                self.levels.append(data)
                xyzct = np.flip(data.shape)
                self.sizes_xyzct.append(xyzct)
                self.sizes.append((xyzct[0], xyzct[1]))
                self.pixel_types.append(data.dtype)
                self.pixel_nbytes.append(data.dtype.itemsize)
        except GroupNotFoundError:
            raise FileNotFoundError(f'File error {filename}')

    def asarray_level(self, level, x0, y0, x1, y1):
        out = self.levels[level][y0:y1, x0:x1]
        return out
