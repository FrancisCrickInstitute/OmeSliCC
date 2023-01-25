import math
import numpy as np
import zarr
from zarr.errors import GroupNotFoundError

from src.OmeSource import OmeSource


class ZarrSource(OmeSource):
    """Zarr-compatible image source"""

    filename: str
    """original filename / URL"""
    levels: list
    """list of all image arrays for different sizes"""

    def __init__(self, filename: str, source_mag: float = None, target_mag: float = None, source_mag_required: bool = False):
        super().__init__()
        self.filename = filename
        self.mag0 = source_mag
        self.target_mag = target_mag

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

            self.levels = []
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
        self._init_metadata(filename, source_mag=source_mag, source_mag_required=source_mag_required)

    def _find_metadata(self):
        pixel_size = []
        channel_info = []
        size_xyzct = self.sizes_xyzct[0]
        for scale in self.metadata.get('multiscales', []):
            axes = ''.join([axis.get('name', '') for axis in scale.get('axes', [])])
            units = [axis.get('unit', '') for axis in scale.get('axes', [])]
            scale1 = [0, 0, 0, 0, 0]
            datasets = scale.get('datasets')
            if datasets is not None:
                coordinateTransformations = datasets[0].get('coordinateTransformations')
                if coordinateTransformations is not None:
                    scale1 = coordinateTransformations[0].get('scale', [0, 0, 0, 0, 0])
            if 'z' in axes:
                pixel_size = [
                    (scale1[axes.index('x')] / size_xyzct[0], units[axes.index('x')]),
                    (scale1[axes.index('y')] / size_xyzct[1], units[axes.index('y')]),
                    (scale1[axes.index('z')] / size_xyzct[2], units[axes.index('z')])]
            else:
                pixel_size = [(1, ''), (1, ''), (1, '')]
        for data in self.metadata.values():
            if isinstance(data, dict):
                for channel in data.get('channels', []):
                    max_val = channel.get('window', {}).get('max', 0)
                    samples_per_pixel = int(math.log(max_val + 1, 256))
                    if samples_per_pixel < 1:
                        samples_per_pixel = 1
                    channel_info.append((channel.get('label', ''), samples_per_pixel))
        self.pixel_size = pixel_size
        self.channel_info = channel_info
        self.mag0 = 0

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        # move channels to back (tczyx -> yxc)
        if x1 < 0 or y1 < 0:
            x1, y1 = self.sizes[level]
        out = self.levels[level][0, :, 0, y0:y1, x0:x1]
        if len(out.shape) > 2:
            return np.moveaxis(out, 0, -1)  # move axis 0 (channel) to end
        else:
            return out
