import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
import zarr

from OmeSliCC.image_util import *
from OmeSliCC.util import *


class OmeZarr:

    DEFAULT_DIMENSION_ORDER = 'tczyx'

    def __init__(self, filename):
        self.filename = filename

    def write(self, data, source, dimension_order=DEFAULT_DIMENSION_ORDER,
               tile_size=[1, 1, 1, 256, 256],
               npyramid_add=0, pyramid_downsample=2, compression=[]):
        compressor, compression_filters = create_compression_filter(compression)
        storage_options = {'dimension_separator': '/', 'chunks': tile_size}
        if compressor is not None:
            storage_options['compressor'] = compressor
        if compression_filters is not None:
            storage_options['filters'] = compression_filters

        zarr_root = zarr.group(parse_url(self.filename, mode="w").store, overwrite=True)
        pixel_size_um = []
        for size in source.get_pixel_size_micrometer():
            if size == 0:
                size = 1
            pixel_size_um.append(size)

        if dimension_order.index('c') == len(dimension_order) - 1:
            # ome-zarr doesn't support channel after space dimensions (yet)
            data = np.moveaxis(data, -1, 0)
            dimension_order = dimension_order[-1] + dimension_order[:-1]

        axes = []
        for dimension in dimension_order:
            unit1 = None
            if dimension == 't':
                type1 = 'time'
                unit1 = 'millisecond'
            elif dimension == 'c':
                type1 = 'channel'
            else:
                type1 = 'space'
                unit1 = 'micrometer'
            axis = {'name': dimension, 'type': type1}
            if unit1 is not None and unit1 != '':
                axis['unit'] = unit1
            axes.append(axis)

        pixel_size_scales = []
        scale = 1
        for i in range(npyramid_add + 1):
            pixel_size_scale = []
            for dimension in dimension_order:
                if dimension == 'z':
                    pixel_size_scale1 = pixel_size_um[2]
                elif dimension == 'y':
                    pixel_size_scale1 = pixel_size_um[1] / scale
                elif dimension == 'x':
                    pixel_size_scale1 = pixel_size_um[0] / scale
                else:
                    pixel_size_scale1 = 1
                pixel_size_scale.append(pixel_size_scale1)
            pixel_size_scales.append([{'scale': pixel_size_scale, 'type': 'scale'}])
            scale /= pyramid_downsample

        write_image(image=data, group=zarr_root, axes=axes, coordinate_transformations=pixel_size_scales,
                    scaler=Scaler(downscale=pyramid_downsample, max_layer=npyramid_add),
                    storage_options=storage_options)

        channels = []
        for channel0 in source.get_channels():
            color = channel0.get('Color', '')
            if not isinstance(color, str):
                color = hex(color)[2:].zfill(6)
            channel = {'label': channel0.get('Name', ''), 'color': color}
            channels.append(channel)

        omero_metadata = {
            'version': '0.4',
            'channels': channels,
        }
        zarr_root.attrs['omero'] = omero_metadata
