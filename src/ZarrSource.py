import numpy as np
import zarr
from zarr.errors import GroupNotFoundError

from src.OmeSource import OmeSource


class ZarrSource(OmeSource):
    def __init__(self, filename, source_mag=None, target_mag=None, source_mag_required=False):
        super().__init__()
        self.mag0 = source_mag
        self.target_mag = target_mag
        self.levels = []

        try:
            root = zarr.open_group(filename, mode='r')
            self.metadata = root.attrs.asdict()

            keys = []
            if 'multiscales' in self.metadata:
                for scale in self.metadata.get('multiscales', []):
                    for index, dataset in enumerate(scale.get('datasets', [])):
                        keys.append(dataset.get('path', index))
            else:
                keys = root.array_keys()

            for key in keys:
                data = root.get(str(key))
                self.levels.append(data)
                xyzct = np.flip(data.shape)
                self.sizes_xyzct.append(xyzct)
                self.sizes.append((xyzct[0], xyzct[1]))
                self.pixel_types.append(data.dtype)
                self.pixel_nbits.append(data.dtype.itemsize * 8)
        except GroupNotFoundError:
            raise FileNotFoundError(f'File error {filename}')
        self.init_metadata(filename, source_mag=source_mag, source_mag_required=source_mag_required)

    def find_metadata(self):
        if 'multiscales' in self.metadata:
            for scale in self.metadata.get('multiscales', []):
                axes = ''.join([axis.get('name', '') for axis in scale.get('axes', [])])
        self.pixel_size = []
        self.channel_info = []
        self.mag0 = 0

    def asarray_level(self, level, x0, y0, x1, y1):
        # move channels to back (tczyx -> yxc)
        out = self.levels[level][0, :, 0, y0:y1, x0:x1]
        if out.shape[0] > 1:
            return np.moveaxis(out, 0, -1)  # move axis 0 (channel) to end
        else:
            return out
