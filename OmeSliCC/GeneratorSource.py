import numpy as np

from OmeSliCC.OmeSource import OmeSource
from OmeSliCC.OmeZarr import OmeZarr
from OmeSliCC.conversion import save_tiff
from OmeSliCC.image_util import get_numpy_slicing


class GeneratorSource(OmeSource):
    def __init__(self, size, tile_size, dtype=np.uint8, source_pixel_size: list = None, seed=None):
        super().__init__()

        size_xyzct = list(size)
        tile_shape = list(np.flip(tile_size))
        if len(size_xyzct) < 3:
            size_xyzct += [1]
        if len(tile_shape) < 3:
            tile_shape += [1]
        size_xyzct += [3]
        tile_shape += [3]
        if len(size_xyzct) < 5:
            size_xyzct += [1]
        if len(tile_shape) < 5:
            tile_shape += [1]
        self.size = size
        self.sizes = [size[:2]]
        self.sizes_xyzct = [size_xyzct]
        self.tile_shape = tile_shape
        dtype = np.dtype(dtype)
        self.dtype = dtype
        self.pixel_types = [dtype]
        self.pixel_nbits.append(dtype.itemsize * 8)

        self._init_metadata('generator', source_pixel_size=source_pixel_size)

        if np.dtype(dtype).kind != 'f':
            self.max_val = 2 ** (8 * np.dtype(dtype).itemsize) - 1
        else:
            self.max_val = 1

        self.is_rgb = True

        if seed is not None:
            np.random.seed(seed)

        self.color_value_table = [np.sin(np.divide(range(dim), dim, dtype=np.float32) * np.pi)
                                  for dim in np.flip(size)]

    def _find_metadata(self):
        self._get_ome_metadate()

    def calc_color(self, *args, **kwargs):
        channels = []
        channel = None
        range0 = kwargs['range0']
        for index, value in enumerate(reversed(args)):
            #channel = np.sin((value + range0[index]) / self.size[index] * np.pi)
            channel = self.color_value_table[index][value + range0[index]]
            channels.append(channel)
        while len(channels) < 3:
            channels.append(channel)
        return np.stack(channels, axis=-1)

    def get_tile(self, indices, tile_size=None):
        # indices / tile size in x,y(,z,...)
        if not tile_size:
            tile_size = np.flip(self.tile_shape)
        range0 = indices
        range1 = np.min([np.array(indices) + np.array(tile_size), self.size], 0)
        shape = list(reversed(range1 - range0))
        tile = np.fromfunction(self.calc_color, shape, dtype=int, range0=range0)
        # apply noise to each channel separately
        for channeli in range(3):
            noise = np.random.random(size=shape) - 0.5
            tile[..., channeli] = np.clip(tile[..., channeli] + noise, 0, 1)
        if self.dtype.kind != 'f':
            tile *= self.max_val
        tile = tile.astype(self.dtype)
        return tile

    def _asarray_level(self, level: int = None, **slicing) -> np.ndarray:
        # ignore level and c
        slices = get_numpy_slicing('xyzt', **slicing)
        indices, tile_size = [], []
        for slice1, axis_size in zip(slices, self.size):
            if axis_size > 0:
                if isinstance(slice1, slice):
                    indices.append(slice1.start)
                    tile_size.append(slice1.stop - slice1.start)
                else:
                    indices.append(slice1)
                    tile_size.append(1)
        data = self.get_tile(indices, tile_size)
        if data.ndim < 4:
            data = np.expand_dims(data, 0)
        data = np.moveaxis(data, -1, 0)
        if data.ndim < 5:
            data = np.expand_dims(data, 0)
        return data


if __name__ == '__main__':
    # (tile) size in x,y(,z,...)
    size = 256, 256, 256
    tile_size = 256, 256, 1
    dtype = np.uint8
    pixel_size = [(1, 'um')]
    seed = 0

    shape = list(reversed(size)) + [3]
    tile_shape = list(reversed(tile_size[:2]))

    print('init')
    source = GeneratorSource(size, tile_size, dtype, pixel_size, seed)
    print('init done')

    data = source.asdask(tile_size)
    print('create data done')
    save_tiff('D:/slides/test.ome.tiff', data, dimension_order=source.get_dimension_order(),
              tile_size=tile_shape, compression='LZW')
    #zarr = OmeZarr('D:/slides/test.ome.zarr')
    #zarr.write(data, source, tile_size=tile_size, npyramid_add=3, pyramid_downsample=2)

    print('done')
