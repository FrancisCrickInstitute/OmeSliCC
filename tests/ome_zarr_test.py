import os.path
import numpy as np
from tqdm import tqdm

from OmeSliCC.image_util import *
from OmeSliCC.OmeZarrSource import OmeZarrSource
from OmeSliCC.Zarr import Zarr


def create_zarr(source, output_filename, tile_size, npyramid_add, pyramid_downsample):
    w, h = source.get_size()
    zarr = Zarr(output_filename)
    zarr.create(source, tile_size=tile_size,
                npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample)
    #if channel_metadata is not None:
    #    print('Adding channel metadata')
    #    zarr.set_channel_metadata(channel_metadata)
    nx, ny = np.ceil(np.divide((w, h), tile_size)).astype(int)
    for yi in tqdm(range(ny)):
        for xi in range(nx):
            x0, y0 = xi * tile_size, yi * tile_size
            x1, y1 = min(x0 + tile_size, w), min(y0 + tile_size, h)
            image = source.asarray(x0, y0, x1, y1)
            zarr.set(image, x0=x0, y0=y0, x1=x1, y1=y1)


def open_omezarr_source(filename):
    source = OmeZarrSource(filename)
    image = source._asarray_level(0, x0=15000, x1=16000, y0=15000, y1=16000)
    show_image(image)
    image = source._asarray_level(4)
    show_image(image)


def convert_ome_zarr_v2_to_v3(filename):
    source = OmeZarrSource(filename)
    zarr = Zarr(output_filename, ome=True, zarr_version=3, ome_version='0.5')
    print(source.shapes, source.chunk_shapes, source.level_scales)
    zarr.create(source, shapes=source.shapes, chunk_shapes=source.chunk_shapes, level_scales=source.level_scales,
                compression=compression)
    for level, data in enumerate(source.get_source_dask()):
        zarr.set_level(level, data)


if __name__ == '__main__':
    #filename = 'E:/Personal/Crick/slides/TCGA_KIRC/0f450938-5604-4af6-8783-c385ea647569/TCGA-A3-3358-01Z-00-DX1.1bd1c720-f6db-4837-8f83-e7476dd2b0a3.svs'
    #filename = 'E:/Personal/Crick/slides/test_images/zarr test.zarr'
    filename = 'D:/slides/6001240.zarr'
    #filename = 'https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr'
    output_filename = 'D:/slides/test/' + os.path.basename(filename)
    npyramid_add = 4
    pyramid_downsample = 2
    tile_size = 16
    compression = None
    output_params = {
        'tile_size': tile_size,
        'npyramid_add': npyramid_add,
        'pyramid_downsample': pyramid_downsample,
        'compression': compression
    }
    #create_zarr(source, output_filename, tile_size, npyramid_add, pyramid_downsample)
    #open_omezarr_source(output_filename)

    convert_ome_zarr_v2_to_v3(filename)

    print('done')
