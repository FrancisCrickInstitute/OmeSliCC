import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
import zarr

from OmeSliCC.image_util import *
from OmeSliCC.ome_zarr_util import *
from OmeSliCC.util import *


class OmeZarr:
    def __init__(self, filename):
        self.filename = filename

    def write(self, data, source, tile_size=[],
               npyramid_add=0, pyramid_downsample=2, compression=[]):
        compressor, compression_filters = create_compression_filter(compression)
        storage_options = {'dimension_separator': '/', 'chunks': tile_size}
        if compressor is not None:
            storage_options['compressor'] = compressor
        if compression_filters is not None:
            storage_options['filters'] = compression_filters

        zarr_root = zarr.group(parse_url(self.filename, mode="w").store, overwrite=True)
        pixel_size_um = source.get_pixel_size_micrometer()

        dimension_order = source.get_dimension_order()
        if 'c' in dimension_order and dimension_order.index('c') == len(dimension_order) - 1:
            # ome-zarr doesn't support channel after space dimensions (yet)
            data = np.moveaxis(data, -1, 0)
            dimension_order = dimension_order[-1] + dimension_order[:-1]

        axes = create_axes_metadata(dimension_order)

        pixel_size_scales = []
        scale = 1
        for i in range(npyramid_add + 1):
            pixel_size_scales.append(create_transformation_metadata(dimension_order, pixel_size_um, scale))
            scale /= pyramid_downsample

        write_image(image=data, group=zarr_root, axes=axes, coordinate_transformations=pixel_size_scales,
                    scaler=Scaler(downscale=pyramid_downsample, max_layer=npyramid_add),
                    storage_options=storage_options)

        zarr_root.attrs['omero'] = create_channel_metadata(source)
