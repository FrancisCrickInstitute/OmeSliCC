import numpy as np
import zarr
from zarr.errors import GroupNotFoundError

from src.OmeSource import OmeSource


class ZarrSource(OmeSource):
    def __init__(self, filename, source_mag=None, target_mag=None, source_mag_required=False):
        self.mag0 = source_mag
        self.target_mag = target_mag
        self.levels = []
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.pixel_nbits = []

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
                self.pixel_nbits.append(data.dtype.itemsize * 8)
        except GroupNotFoundError:
            raise FileNotFoundError(f'File error {filename}')
        self.init_res_mag(filename, source_mag_required=source_mag_required)

    def asarray_level(self, level, x0, y0, x1, y1):
        out = self.levels[level][y0:y1, x0:x1]
        return out
