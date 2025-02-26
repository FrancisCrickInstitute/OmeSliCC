import numpy as np
import os.path
import pathlib
import shutil
import zarr
from ome_zarr.scale import Scaler

from OmeSliCC.image_util import *
from OmeSliCC.ome_zarr_util import *
from OmeSliCC.util import *


class Zarr:
    def __init__(self, filename, ome=None, v3=False):
        self.filename = filename
        if ome is not None:
            self.ome = ome
        else:
            self.ome = ('ome' == self.filename.split('.')[1].lower())
        self.v3 = v3
        self.data = []

    def create(self, source, shapes=[], chunk_shapes=[], level_scales=[],
               tile_size=None, npyramid_add=0, pyramid_downsample=2, compression=[],
               ome_version = '0.4'):
        # create empty dataset
        dimension_order = source.get_dimension_order()
        self.dimension_order = dimension_order
        nlevels = max(1 + npyramid_add, len(shapes))
        self.npyramid_add = npyramid_add
        self.pyramid_downsample = pyramid_downsample
        self.level_scales = level_scales
        file_url = pathlib.Path(self.filename, mode='w').as_uri()
        self.zarr_root = zarr.open_group(store=file_url, mode='w', storage_options={'dimension_separator': '/'})

        xyzct = source.get_size_xyzct()
        self.scaler = Scaler(downscale=pyramid_downsample, max_layer=npyramid_add)
        shape0 = [xyzct['xyzct'.index(dimension)] for dimension in dimension_order]
        dtype = source.pixel_types[0]
        pixel_size_um = source.get_pixel_size_micrometer()
        translation_um = source.get_position_micrometer()
        scale = 1
        datasets = []
        if tile_size and len(tile_size) < 5:
            if isinstance(tile_size, int):
                tile_size = [tile_size] * 2
            elif len(tile_size) == 1:
                tile_size = tile_size * 2
            tile_size = list(np.flip(tile_size))
            while len(tile_size) < 5:
                tile_size = [1] + tile_size
        for level in range(nlevels):
            if len(shapes) > 0:
                shape = shapes[level]
            else:
                shape = scale_dimensions_xy(shape0, dimension_order, scale)
            if len(chunk_shapes) > 0:
                chunk_shape = chunk_shapes[level]
            else:
                chunk_shape = np.min([tile_size, shape], axis=0)
            if self.v3:
                import zarrita
                shape = np.array(shape).tolist()                # convert to basic int
                chunk_shape = np.array(chunk_shape).tolist()    # convert to basic int
                codecs = create_compression_codecs(compression)
                dataset = self.zarr_root.create_array(str(level), shape=shape, chunk_shape=chunk_shape, dtype=dtype,
                                                      codecs=codecs)
            else:
                chunk_shape = tile_size
                compressor, compression_filters = create_compression_filter(compression)
                dataset = self.zarr_root.create_dataset(str(level), shape=shape, chunks=chunk_shape, dtype=dtype,
                                                        compressor=compressor, filters=compression_filters)
            self.data.append(dataset)
            # used for ome metadata:
            datasets.append({
                'path': str(level),
                'coordinateTransformations': create_transformation_metadata(dimension_order, pixel_size_um, scale, translation_um)
            })
            scale /= pyramid_downsample

        if self.ome:
            multiscales = {
                'version': ome_version,
                'axes': create_axes_metadata(dimension_order),
                'name': get_filetitle(source.source_reference),
                'datasets': datasets,
            }
            metadata = {'multiscales': [multiscales], 'omero': create_channel_metadata(source, ome_version)}
            if self.v3:
                self.zarr_root.update_attributes(metadata)
            else:
                self.zarr_root.attrs = metadata

    def write(self, sources, tile_size=[], compression=[],
              npyramid_add=0, pyramid_downsample=2,
              translations=[], image_operations=[]):
        source0 = sources[0]
        dimension_order = source0.get_dimension_order()
        self.dimension_order = dimension_order
        pixel_size_um = source0.get_pixel_size_micrometer()

        compressor, compression_filters = create_compression_filter(compression)
        storage_options = {'dimension_separator': '/'}
        ome_version = '0.4'
        # Zarr V3 testing
        #storage_options = {'chunks': tile_size}
        if compressor is not None:
            storage_options['compressor'] = compressor
        if compression_filters is not None:
            storage_options['filters'] = compression_filters

        file_url = pathlib.Path(self.filename, mode='w').as_uri()
        zarr_root = zarr.open_group(store=file_url, mode='w', storage_options=storage_options)

        scale = 1
        multiscales = []
        for index, source in enumerate(sources):
            data = source.asarray()
            for image_operation in image_operations:
                data = image_operation(data)
            root_path = str(index)
            if index < len(translations):
                translation_um = translations[index]
            else:
                translation_um = source.get_position_micrometer()
            dataset = zarr_root.create_dataset(root_path, data=np.asarray(data), chunks=data.chunksize,
                                               compressor=compressor, filters=compression_filters)
            self.data.append(dataset)
            # used for ome metadata:
            datasets = [{
                'path': root_path,
                'coordinateTransformations': create_transformation_metadata(dimension_order, pixel_size_um, scale, translation_um)
            }]
            multiscale = {
                'version': ome_version,
                'axes': create_axes_metadata(dimension_order),
                'name': get_filetitle(source.source_reference),
                'datasets': datasets,
            }
            multiscales.append(multiscale)

        if self.ome:
            metadata = {'multiscales': multiscales, 'omero': create_channel_metadata(source0, ome_version)}
            if self.v3:
                zarr_root.update_attributes(metadata)
            else:
                zarr_root.attrs.update(metadata)

    def get(self, level, **slicing):
        slices = get_numpy_slicing(self.dimension_order, **slicing)
        data = self.data[level][slices]
        return data

    def set(self, data, **slicing):
        scale = 1
        for level, sized_data in enumerate(self.scaler.nearest(data)):
            resized_slicing = scale_dimensions_dict(slicing, scale)
            slices = get_numpy_slicing(self.dimension_order, **resized_slicing)
            self.data[level][slices] = np.asarray(sized_data)
            scale /= self.pyramid_downsample

    def set_level(self, level, data, **slicing):
        resized_slicing = scale_dimensions_dict(slicing, 1 / self.level_scales[level])
        slices = get_numpy_slicing(self.dimension_order, **resized_slicing)
        self.data[level][slices] = np.asarray(data)
