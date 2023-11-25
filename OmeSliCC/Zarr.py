import numpy as np
import pathlib
import zarr

from OmeSliCC.image_util import *
from OmeSliCC.util import *


class Zarr:

    DEFAULT_DIMENSION_ORDER = 'tczyx'

    def __init__(self, filename):
        self.filename = filename
        self.ome = ('ome' == self.filename.split('.')[1].lower())
        self.metadata = {}
        self.data = []
        self.sizes = []
        self.shapes = []
        self.dimension_order = self.DEFAULT_DIMENSION_ORDER

    def create(self, source, dimension_order=DEFAULT_DIMENSION_ORDER,
               tile_size=[1, 1, 1, 256, 256],
               npyramid_add=0, pyramid_downsample=2, compression=[]):
        # create empty dataset
        self.dimension_order = dimension_order
        self.npyramid_add = npyramid_add
        self.pyramid_downsample = pyramid_downsample
        file_url = pathlib.Path(self.filename).as_uri()
        self.zarr_root = zarr.open_group(file_url, mode='w', storage_options={'dimension_separator': '/'})
        size0 = source.get_size_xyzct()
        shape0 = list(np.flip(size0))
        self.dtype = source.pixel_types[0]
        pixel_size = source.get_pixel_size()
        compressor, compression_filters = create_compression_filter(compression)
        scale = 1
        datasets = []
        for pathi in range(1 + npyramid_add):
            shape = shape0[:-2] + np.round(np.multiply(shape0[-2:], scale)).astype(int).tolist()
            self.shapes.append(shape)
            self.sizes.append(np.flip(shape))
            self.data.append(self.zarr_root.create_dataset(str(pathi), shape=shape, chunks=tile_size, dtype=self.dtype,
                                                           compressor=compressor, filters=compression_filters))
            pixel_size_x = pixel_size[0][0] if len(pixel_size) >= 1 else 1
            pixel_size_y = pixel_size[1][0] if len(pixel_size) >= 2 else 1
            pixel_size_z = pixel_size[2][0] if len(pixel_size) >= 3 else 1
            if pixel_size_z == 0:
                pixel_size_z = 1
            datasets.append({
                'path': pathi,
                'coordinateTransformations': [{'type': 'scale', 'scale': [1, 1, pixel_size_z, pixel_size_y / scale, pixel_size_x / scale]}]
            })
            scale /= pyramid_downsample

    def get(self, level, x0=0, y0=0, x1=-1, y1=-1):
        data = self.data[level][0, :, 0, y0:y1, x0:x1].squeeze()
        data = np.moveaxis(data, 0, -1)
        return data

    def set(self, data, x0=0, y0=0, x1=0, y1=0):
        if y1 == 0:
            y1 = data.shape[0]
        if x1 == 0:
            x1 = data.shape[1]
        scale = 1
        for pathi in range(1 + self.npyramid_add):
            sx0, sy0, sx1, sy1 = np.round(np.multiply([x0, y0, x1, y1], scale)).astype(int)
            if scale != 1:
                new_size = sx1 - sx0, sy1 - sy0
                data1 = image_resize(data, new_size)
            else:
                data1 = data
            data1 = np.moveaxis(data1, -1, 0)
            self.data[pathi][0, :, 0, sy0:sy1, sx0:sx1] = data1
            scale /= self.pyramid_downsample
