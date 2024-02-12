import numpy as np
from tqdm import tqdm

from OmeSliCC.image_util import *
from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.Zarr import Zarr
from OmeSliCC.OmeZarrSource import OmeZarrSource


def simple_zarr(source, output_filename, tile_size, npyramid_add, pyramid_downsample):
    w, h = source.get_size()
    zarr = Zarr(output_filename)
    zarr.create(source, tile_size=[1, 1, 1, tile_size, tile_size],
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
            zarr.set(x0, y0, x1, y1, image)


def open_zarr_source(filename):
    source = OmeZarrSource(filename)
    image = source._asarray_level(0, x0=15000, x1=16000, y0=15000, y1=16000)
    show_image(image)
    image = source._asarray_level(4)
    show_image(image)


if __name__ == '__main__':
    source = TiffSource('E:/Personal/Crick/slides/TCGA_KIRC/0f450938-5604-4af6-8783-c385ea647569/TCGA-A3-3358-01Z-00-DX1.1bd1c720-f6db-4837-8f83-e7476dd2b0a3.svs')
    w, h = source.get_size()
    output_filename = 'D:/slides/test/test.ome.zarr'
    npyramid_add = 4
    pyramid_downsample = 2
    tile_size = 2048

    simple_zarr(source, output_filename, tile_size, npyramid_add, pyramid_downsample)
    open_zarr_source(output_filename)
