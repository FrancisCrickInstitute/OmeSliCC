import os
import numpy as np
from PIL import Image
import zarr
from zarr.errors import GroupNotFoundError


class ZarrSlide:
    def __init__(self, filename):
        self.levels = []
        self.level_dimensions = []
        zarr_filename = os.path.splitext(filename)[0] + '.zarr'
        try:
            root = zarr.open_group(zarr_filename, mode='r')
            for level in root.array_keys():
                data = root.get(str(level))
                self.levels.append(data)
                self.level_dimensions.append((data.shape[1], data.shape[0]))
        except GroupNotFoundError:
            raise FileNotFoundError(f'Zarr root path not found {zarr_filename}')

    def get_thumbnail(self, size):
        data = self.levels[-1]
        thumb = Image.fromarray(data[:])
        thumb.thumbnail(size, Image.ANTIALIAS)
        return thumb

    def asarray(self, level, x0, y0, x1, y1):
        return self.read_region((x0, y0), level, (x1 - x0, y1 - y0))

    def read_region(self, position, level, size):
        x, y = position
        w, h = size
        out = self.levels[level][y:y + h, x:x + w]
        if out.shape[0] < h or out.shape[1] < w:
            dx = w - out.shape[1]
            dy = h - out.shape[0]
            out = np.pad(out, ((0, dy), (0, dx), (0, 0)), 'edge')
        return out
