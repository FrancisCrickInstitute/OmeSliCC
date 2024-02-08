import dask
import dask.array as da
import numpy as np
from tqdm import tqdm

from OmeSliCC.OmeSource import OmeSource
from OmeSliCC.OmeZarr import OmeZarr


class GeneratorSource(OmeSource):
    def __init__(self, size, tile_size, dtype=np.uint8, source_pixel_size: list = None, seed=None):
        super().__init__()

        size_xyzct = size
        while len(size_xyzct) < 5:
            size_xyzct = list(size_xyzct) + [1]
        while len(tile_size) < len(size):
            tile_size = list(tile_size) + [1]
        self.size = size
        self.sizes = [size[:2]]
        self.sizes_xyzct = [size_xyzct]
        self.tile_size = tile_size
        dtype = np.dtype(dtype)
        self.dtype = dtype
        self.pixel_types = [dtype]
        self.pixel_nbits.append(dtype.itemsize * 8)

        self._init_metadata('generator', source_pixel_size=source_pixel_size)

        if np.dtype(dtype).kind != 'f':
            self.max_val = 2 ** (8 * np.dtype(dtype).itemsize) - 1
        else:
            self.max_val = 1

        ranges = np.flip(np.ceil(np.divide(size, tile_size)).astype(int))
        self.tile_indices = list(np.ndindex(tuple(ranges)))

        if seed is not None:
            np.random.seed(seed)

        self.color_value_table = [np.sin(np.divide(range(dim), dim, dtype=np.float32) * np.pi)
                                  for dim in np.flip(size)]

        # self.noise = np.random.random(size=shape) - 0.5     # uniform
        self.noise = np.random.normal(loc=0, scale=0.1, size=np.flip(tile_size))  # gaussian

    def _find_metadata(self):
        self._get_ome_metadate()

    def calc_color(self, *args):
        channels = []
        channel = None
        for index, value in enumerate(reversed(args)):
            #channel = np.sin((value + self.range0[index]) / self.size[index] * np.pi)
            channel = self.color_value_table[index][value + self.range0[index]]
            channels.append(channel)
        while len(channels) < 3:
            channels.append(channel)
        return np.stack(channels, axis=-1)

    def get_tile(self, indices, channels_last=False):
        # tile in (z,),y,x,c
        self.range0 = np.flip(indices[1:]) * self.tile_size
        self.range1 = np.min([self.range0 + self.tile_size, self.size], 0)
        shape = list(reversed(self.range1 - self.range0))
        tile = np.fromfunction(self.calc_color, shape, dtype=int)
        # apply noise to each channel separately
        for channeli in range(3):
            tile[..., channeli] = np.clip(tile[..., channeli] + self.noise, 0, 1)
        if self.dtype.kind != 'f':
            tile *= self.max_val
        tile = tile.astype(self.dtype)
        if not channels_last:
            tile = np.moveaxis(tile, -1, 0)
        return tile

    def get_tiles(self, channels_last=False):
        for indices in tqdm(self.tile_indices):
            yield self.get_tile(indices, channels_last)

    def _get_output_dask(self):
        # TODO: fix dimensions / shapes

        #data = da.fromfunction(lambda indices: self.get_tile(indices), self.tile_indices, shape=self.tile_size, dtype=self.dtype)

        delayed_reader = dask.delayed(self.get_tile)
        tile_shape = [3] + list(np.flip(self.tile_size))
        dask_tiles = []
        for indices in self.tile_indices:
            indices_tzyx = indices
            while len(indices_tzyx) < 4:
                indices_tzyx = [0] + list(indices_tzyx)
            dask_tile = da.from_delayed(delayed_reader(indices_tzyx), shape=tile_shape, dtype=self.dtype)
            dask_tiles.append(dask_tile)
        dask_data = da.block(dask_tiles)
        return dask_data

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1,
                       c: int = None, z: int = None, t: int = None) -> np.ndarray:
        return self.get_output_dask()


if __name__ == '__main__':
    # (tile) size in x,y(,z)
    size = 256, 256, 256
    tile_size = 256, 256, 1
    dtype = np.uint8
    pixel_size = ['1um']
    seed = 0

    shape = list(reversed(size)) + [3]
    tile_shape = list(reversed(tile_size[:2]))

    print('init')
    generator = GeneratorSource(size, tile_size, dtype, pixel_size, seed)
    print('init done')

    #save_tiff('D:/slides/test.ome.tiff', generator.get_tiles(channels_last=True), shape, dtype, tile_size=tile_shape)
    zarr = OmeZarr('D:/slides/test.ome.zarr')
    zarr.write(generator.get_output_dask(), generator, tile_size=tile_size, npyramid_add=3, pyramid_downsample=2)

    print('done')
