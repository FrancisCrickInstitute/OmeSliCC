import numpy as np
import zarr
from zarr.errors import GroupNotFoundError

from OmeSliCC.image_util import *
from OmeSliCC.util import *


class Zarr:

    default_dimension_order = 'tczyx'

    def __init__(self, filename):
        self.filename = filename
        self.ome = ('ome' == self.filename.split('.')[1].lower())
        self.metadata = {}
        self.data = []
        self.sizes = []
        self.shapes = []
        self.dimension_order = self.default_dimension_order

    def open(self):
        try:
            self.zarr_root = zarr.open_group(self.filename, mode='r')
            self.metadata = self.zarr_root.attrs.asdict()

            paths = []
            dimension_order = self.default_dimension_order
            if 'multiscales' in self.metadata:
                for scale in self.metadata.get('multiscales', []):
                    for index, dataset in enumerate(scale.get('datasets', [])):
                        paths.append(dataset.get('path', str(index)))
                    axes = scale.get('axes', [])
                    if len(axes) > 0:
                        dimension_order = ''.join([axis.get('name') for axis in axes])
            else:
                paths = self.zarr_root.array_keys()
            self.dimension_order = dimension_order
            for path in paths:
                data1 = self.zarr_root.get(path)
                self.data.append(data1)
                shape = [1, 1, 1, 1, 1]
                for i, n in enumerate(data1.shape):
                    shape_index = self.default_dimension_order.index(dimension_order[i])
                    shape[shape_index] = n
                self.shapes.append(shape)
                self.sizes.append(np.flip(shape))
                self.dtype = data1.dtype
        except GroupNotFoundError as e:
            raise FileNotFoundError(f'Read error: {e}')

    def create(self, source, tile_size=[1, 1, 1, 256, 256], npyramid_add=0, pyramid_downsample=2):
        self.npyramid_add = npyramid_add
        self.pyramid_downsample = pyramid_downsample
        self.zarr_root = zarr.open_group(self.filename, mode='w')
        size0 = source.get_size_xyzct()
        shape0 = list(np.flip(size0))
        self.dtype = source.pixel_types[0]
        pixel_size = source.get_pixel_size()
        scale = 1
        datasets = []
        for pathi in range(1 + npyramid_add):
            shape = shape0[:-2] + np.round(np.multiply(shape0[-2:], scale)).astype(int).tolist()
            self.shapes.append(shape)
            self.sizes.append(np.flip(shape))
            self.data.append(self.zarr_root.create_dataset(str(pathi), shape=shape, chunks=tile_size, dtype=self.dtype))
            pixel_size_x = pixel_size[0][0] if len(pixel_size) >= 1 else 1
            pixel_size_y = pixel_size[1][0] if len(pixel_size) >= 2 else 1
            pixel_size_z = pixel_size[2][0] if len(pixel_size) >= 3 else 1
            datasets.append({
                'path': pathi,
                'coordinateTransformations': [{'type': 'scale', 'scale': [1, 1, pixel_size_z, pixel_size_y / scale, pixel_size_x / scale]}]
            })
            scale /= pyramid_downsample

        if self.ome:
            self.dimension_order = self.default_dimension_order
            axes = []
            for dimension in self.dimension_order:
                unit1 = None
                if dimension == 't':
                    type1 = 'time'
                    unit1 = 'millisecond'
                elif dimension == 'c':
                    type1 = 'channel'
                else:
                    type1 = 'space'
                    index = 'xyz'.index(dimension)
                    unit1 = pixel_size[index][1] if index < len(pixel_size) else ''
                axis = {'name': dimension, 'type': type1}
                if unit1 is not None and unit1 != '':
                    axis['unit'] = unit1
                axes.append(axis)

            channels = [{'label': channel.get('Name', ''), 'color': channel.get('Color', '')}
                        for channel in source.get_channels()]

            metadata = {
                'version': '0.4',
                'axes': axes,
                'name': get_filetitle(source.source_reference),
                'datasets': datasets,
            }

            self.metadata['multiscales'] = [metadata]
            self.zarr_root.attrs['multiscales'] = [metadata]
            self.set_channel_metadata(channels)

    def set_channel_metadata(self, channels):
        omero_metadata = {
            'version': '0.4',
            'channels': channels,
        }
        self.metadata['omero'] = omero_metadata
        self.zarr_root.attrs['omero'] = omero_metadata

    def get(self, level, x0=0, y0=0, x1=-1, y1=-1):
        data = self.data[level][0, :, 0, y0:y1, x0:x1].squeeze()
        data = np.moveaxis(data, 0, -1)
        return data

    def set(self, x0, y0, x1, y1, data):
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
