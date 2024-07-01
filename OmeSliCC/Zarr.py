import numpy as np
import os.path
import pathlib
import shutil
import zarr

from OmeSliCC.image_util import *
from OmeSliCC.ome_zarr_util import *
from OmeSliCC.util import *


class Zarr:
    def __init__(self, filename):
        self.filename = filename
        self.ome = ('ome' == self.filename.split('.')[1].lower())
        self.data = []

    def create(self, source, tile_size=None, npyramid_add=0, pyramid_downsample=2, compression=[],
               v3=False):
        # create empty dataset
        dimension_order = source.get_dimension_order()
        self.dimension_order = dimension_order
        self.npyramid_add = npyramid_add
        self.pyramid_downsample = pyramid_downsample
        if v3:
            import zarrita
            store_path = zarrita.store.make_store_path(self.filename)
            if os.path.exists(self.filename):
                shutil.rmtree(str(store_path.store.root))
            self.zarr_root = zarrita.Group.create(store=store_path, exists_ok=True)
        else:
            file_url = pathlib.Path(self.filename, mode='w').as_uri()
            self.zarr_root = zarr.open_group(store=file_url, mode='w', storage_options={'dimension_separator': '/'})
        xyzct = source.get_size_xyzct()
        shape0 = [xyzct['xyzct'.index(dimension)] for dimension in dimension_order]
        dtype = source.pixel_types[0]
        pixel_size_um = source.get_pixel_size_micrometer()
        scale = 1
        datasets = []
        if tile_size:
            if isinstance(tile_size, int):
                tile_size = [tile_size] * 2
            elif len(tile_size) == 1:
                tile_size = tile_size * 2
            tile_size = [1, 1, 1] + list(np.flip(tile_size))
        for pathi in range(1 + npyramid_add):
            shape = calc_shape_scale(shape0, dimension_order, scale)
            if v3:
                import zarrita
                shape = np.array(shape).tolist()    # convert to basic int
                tile_size = np.array(tile_size).tolist()  # convert to basic int
                codecs = create_compression_codecs(compression)
                dataset = self.zarr_root.create_array(str(pathi), shape=shape, chunk_shape=tile_size, dtype=dtype,
                                                      codecs=codecs)
            else:
                compressor, compression_filters = create_compression_filter(compression)
                dataset = self.zarr_root.create_dataset(str(pathi), shape=shape, chunks=tile_size, dtype=dtype,
                                                        compressor=compressor, filters=compression_filters)
            self.data.append(dataset)
            # used for ome metadata:
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
        data = self.data[level][..., y0:y1, x0:x1]
        return data

    def set(self, data, x0=0, y0=0, x1=0, y1=0):
        if y1 <= 0:
            y1 = data.shape[-2]
        if x1 <= 0:
            x1 = data.shape[-1]
        scale = 1
        for pathi in range(1 + self.npyramid_add):
            sx0, sy0, sx1, sy1 = np.round(np.multiply([x0, y0, x1, y1], scale)).astype(int)
            if scale != 1:
                new_size = sx1 - sx0, sy1 - sy0
                data1 = image_resize(data, new_size, dimension_order=self.dimension_order)
            else:
                data1 = data
            #self.data[pathi][..., sy0:sy1, sx0:sx1] = data1
            self.data[pathi] = data1
            scale /= self.pyramid_downsample
