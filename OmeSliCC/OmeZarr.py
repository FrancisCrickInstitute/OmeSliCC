import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
import zarr

from OmeSliCC.image_util import *
from OmeSliCC.ome_zarr_util import *
from OmeSliCC.util import *


class OmeZarr:
    def __init__(self, filename, zarr_version=2, ome_version='0.4'):
        self.filename = filename
        self.zarr_version = zarr_version
        self.ome_version = ome_version

    def write(self, sources, tile_size=[], compression=[],
              npyramid_add=0, pyramid_downsample=2,
              translations=[], image_operations=[]):

        multiple_images = isinstance(sources, list)
        multi_metadata = []
        if not multiple_images:
            sources = [sources]

        chunk_size = tile_to_chunk_size(tile_size, len(sources[0].get_dimension_order()))
        compressor, compression_filters = create_compression_filter(compression)
        storage_options = {'dimension_separator': '/', 'chunks': chunk_size}
        if compressor is not None:
            storage_options['compressor'] = compressor
        if compression_filters is not None:
            storage_options['filters'] = compression_filters
        self.storage_options = storage_options

        zarr_root = zarr.open_group(store=parse_url(self.filename, mode="w").store, mode="w",
                                    zarr_version=self.zarr_version)
        root_path = ''

        omero_metadata = create_channel_metadata(sources[0], self.ome_version)

        for index, source in enumerate(sources):
            dimension_order = source.get_dimension_order()
            pixel_size_um = source.get_pixel_size_micrometer()
            data = source.asarray()
            for image_operation in image_operations:
                data = image_operation(data)
            if multiple_images:
                root_path = str(index)
                group = zarr_root.create_group(root_path)
            else:
                group = zarr_root
            if index < len(translations):
                translation_um = translations[index]
            else:
                translation_um = source.get_position_micrometer()
            self.write_dataset(group, data, dimension_order, pixel_size_um,
                               npyramid_add, pyramid_downsample, translation_um)
            if multiple_images:
                meta = group.attrs['multiscales'][0].copy()
                for dataset_meta in meta['datasets']:
                    dataset_meta['path'] = f'{root_path}/{dataset_meta["path"]}'
                multi_metadata.append(meta)
                group.attrs['omero'] = omero_metadata
        if multiple_images:
            zarr_root.attrs['multiscales'] = multi_metadata
        zarr_root.attrs['omero'] = omero_metadata

    def write_dataset(self, zarr_group, data, dimension_order, pixel_size_um,
                      npyramid_add=0, pyramid_downsample=2, translation_um=[]):

        if 'c' in dimension_order and dimension_order.index('c') == len(dimension_order) - 1:
            # ome-zarr doesn't support channel after space dimensions (yet)
            data = np.moveaxis(data, -1, 0)
            dimension_order = dimension_order[-1] + dimension_order[:-1]

        axes = create_axes_metadata(dimension_order)

        pixel_size_scales = []
        scale = 1
        for i in range(npyramid_add + 1):
            pixel_size_scales.append(create_transformation_metadata(dimension_order, pixel_size_um, scale, translation_um))
            if pyramid_downsample:
                scale /= pyramid_downsample

        write_image(image=data, group=zarr_group, axes=axes, coordinate_transformations=pixel_size_scales,
                    scaler=Scaler(downscale=pyramid_downsample, max_layer=npyramid_add),
                    storage_options=self.storage_options)
