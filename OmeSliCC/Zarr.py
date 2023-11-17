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
               npyramid_add=0, pyramid_downsample=2):
        self.dimension_order = dimension_order
        self.npyramid_add = npyramid_add
        self.pyramid_downsample = pyramid_downsample
        file_url = pathlib.Path(self.filename).as_uri()
        self.zarr_root = zarr.open_group(file_url, mode='w', storage_options={'dimension_separator': '/'})
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
            if pixel_size_z == 0:
                pixel_size_z = 1
            datasets.append({
                'path': pathi,
                'coordinateTransformations': [{'type': 'scale', 'scale': [1, 1, pixel_size_z, pixel_size_y / scale, pixel_size_x / scale]}]
            })
            scale /= pyramid_downsample

        if self.ome:
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
