import numpy as np
from tqdm import tqdm

from OmeSliCC.GeneratorSource import GeneratorSource
from OmeSliCC.OmeZarrSource import OmeZarrSource
from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.conversion import save_image_as_tiff, save_image_as_ome_zarr
from OmeSliCC.image_util import *


def create_source(size, tile_size, dtype, pixel_size):
    source = GeneratorSource(size, tile_size, dtype, pixel_size)
    return source


def load_as_zarr(path, x0_um, x1_um, y0_um, y1_um):
    source = TiffSource(path)
    data = source.asarray_um(x0=x0_um, x1=x1_um, y0=y0_um, y1=y1_um)
    image = np.asarray(data)
    image = source.render(image, source.get_dimension_order())
    show_image(image)


def conversion_test():
    path1 = 'D:/slides/test1.ome.tiff'
    path2 = 'D:/slides/test1.ome.zarr'
    path3 = 'D:/slides/test2.ome.tiff'
    size = [1024, 1024]
    tile_size = [256, 256]
    output_params = {'tile_size': tile_size, 'npyramid_add': 3, 'pyramid_downsample': 2}

    source = create_source(size, tile_size, np.uint8, [(1, 'um')])
    nchannels = source.get_nchannels()
    save_image_as_tiff(source, source.asdask(tile_size), path1, output_params, ome=True)

    source1 = TiffSource(path1)
    assert source1 is not None
    save_image_as_ome_zarr(source, source1.asarray(), path2, output_params)

    source2 = OmeZarrSource(path2)
    assert source2 is not None
    save_image_as_tiff(source, source2.asarray(), path3, output_params, ome=True)

    source3 = TiffSource(path3)
    assert source3 is not None

    data = source3.asarray(pixel_size=[10])
    assert data.ndim == 5
    image = redimension_data(data, source3.get_dimension_order(), 'yxc', t=0, z=0)
    assert image.ndim == 3 and image.shape[2] == nchannels
    x0, y0, x1, y1 = size[0] // 2, size[1] // 2, size[0], size[1]
    w, h = np.divide(size, 2).astype(int)
    tile = source3.asarray(x0=x0, x1=x1, y0=y0, y1=y1)
    assert tile.shape == (1, nchannels, 1, h, w)
    tile = source3.asarray(x0=x0//10, x1=x1//10, y0=y0//10, y1=y1//10, pixel_size=[10])
    assert tile.shape == (1, nchannels, 1, h//10, w//10)


def check_large_tiff_arrays(input):
    source = TiffSource(input, target_pixel_size=[(10, 'um')])
    dimension_order = source.get_dimension_order()
    image = source.render(source.asarray(), dimension_order)
    show_image(image)
    tile = source.render(source.asarray(x0=1100, x1=2100, y0=1200, y1=1400), dimension_order)
    show_image(tile)
    tile = source.render(source.asarray(x0=1100, x1=2100, y0=1200, y1=1400, pixel_size=[10]), dimension_order)
    show_image(tile)


if __name__ == '__main__':
    path = 'D:/slides/EM04573_01small.ome.tif'
    path2 = 'D:/slides/test.ome.zarr'
    path3 = 'D:/slides/test.ome.tiff'
    output_params = {'tile_size': [256, 256], 'npyramid_add': 3, 'pyramid_downsample': 2}

    load_as_zarr('D:/slides/EM04573_01/EM04573_01.ome.tif', 20400, 20900, 13000, 13500)

    conversion_test()

    check_large_tiff_arrays(path)

    progress = tqdm(range(2))
    source = TiffSource(path)
    save_image_as_ome_zarr(source, source.asarray(), path2, output_params)
    progress.update(0)

    source2 = OmeZarrSource(path2)
    save_image_as_tiff(source, source.asarray(), path3, output_params, ome=True)
    progress.update(1)

    print('Done')
