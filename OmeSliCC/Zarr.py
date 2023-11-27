import numpy as np
import pathlib
import zarr

from OmeSliCC.image_util import *
from OmeSliCC.ome_zarr_util import *
from OmeSliCC.util import *


class Zarr:

    DEFAULT_DIMENSION_ORDER = 'tczyx'

    def __init__(self, filename):
        self.filename = filename
        self.ome = ('ome' == self.filename.split('.')[1].lower())
        self.data = []
        self.dimension_order = self.DEFAULT_DIMENSION_ORDER

    def create(self, source, tile_size=[],
               npyramid_add=0, pyramid_downsample=2, compression=[]):
        # create empty dataset
        dimension_order = source.get_dimension_order()
        self.dimension_order = dimension_order
        self.npyramid_add = npyramid_add
        self.pyramid_downsample = pyramid_downsample
        file_url = pathlib.Path(self.filename).as_uri()
        self.zarr_root = zarr.open_group(file_url, mode='w', storage_options={'dimension_separator': '/'})
        size0 = source.get_size_xyzct()
        shape0 = [size0['xyzct'.index(dimension)] for dimension in dimension_order]
        dtype = source.pixel_types[0]
        pixel_size_um = source.get_pixel_size_micrometer()
        compressor, compression_filters = create_compression_filter(compression)
        scale = 1
        datasets = []
        for pathi in range(1 + npyramid_add):
            shape = calc_shape_scale(shape0, dimension_order, scale)
            self.data.append(self.zarr_root.create_dataset(str(pathi), shape=shape, chunks=tile_size, dtype=dtype,
                                                           compressor=compressor, filters=compression_filters))
            datasets.append({
                'path': str(pathi),
                'coordinateTransformations': create_transformation_metadata(dimension_order, pixel_size_um, scale)
            })
            scale /= pyramid_downsample

        if self.ome:
            metadata = {
                'version': '0.4',
                'axes': create_axes_metadata(dimension_order),
                'name': get_filetitle(source.source_reference),
                'datasets': datasets,
            }

            self.zarr_root.attrs['multiscales'] = [metadata]
            self.zarr_root.attrs['omero'] = create_channel_metadata(source)

    def get(self, level, x0=0, y0=0, x1=-1, y1=-1):
        data = self.data[level][0, :, 0, y0:y1, x0:x1].squeeze()
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
            self.data[pathi][sy0:sy1, sx0:sx1, :] = data1
            scale /= self.pyramid_downsample
