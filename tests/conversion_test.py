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


def load_as_zarr(path, x_um, y_um, w_um, h_um):
    source = TiffSource(path)
    data = source.asarray_um(x_um, y_um, w_um, h_um)
    image = np.asarray(data)
    image = source.render(image)
    show_image(image)


def save_source_as_ome_tiff(output, source, output_params):
    #data = source.asarray()
    data = source.asdask(output_params['tile_size'])
    save_image_as_tiff(source, data, output, output_params, ome=True)


def save_source_as_zarr(output, source, output_params):
    data = source.asarray()
    #data = source.asdask(output_params['tile_size'])
    save_image_as_ome_zarr(source, data, output, output_params)


def convert_tiff_to_zarr(input, output, output_params):
    print('Converting tiff to zarr')
    source = TiffSource(input)
    save_source_as_zarr(output, source, output_params)


def convert_zarr_to_tiff(input, output, output_params):
    print('Converting zarr to tiff')
    source = OmeZarrSource(input)
    save_source_as_ome_tiff(output, source, output_params)


def conversion_test():
    path1 = 'test1.ome.tiff'
    path2 = 'test1.ome.zarr'
    path3 = 'test2.ome.tiff'
    size = [1024, 1024]
    tile_size = [256, 256]
    output_params = {'tile_size': tile_size, 'npyramid_add': 3, 'pyramid_downsample': 2}

    source = create_source(size, tile_size, np.uint8, [(1, 'um')])
    nchannels = source.get_nchannels()
    save_source_as_ome_tiff(path1, source, output_params)

    source1 = TiffSource(path1)
    assert source1 is not None
    convert_tiff_to_zarr(path1, path2, output_params)

    source2 = OmeZarrSource(path2)
    assert source2 is not None
    convert_zarr_to_tiff(path2, path3, output_params)

    source3 = TiffSource(path3)
    assert source3 is not None

    data = source3.asarray(pixel_size=[10])
    assert data.ndim == 5
    image, _ = source3.get_yxc_image(data)
    assert image.ndim == 3 and image.shape[2] == nchannels
    x0, y0, x1, y1 = size[0] // 2, size[1] // 2, size[0], size[1]
    w, h = np.divide(size, 2).astype(int)
    tile = source3.asarray(x0, y0, x1, y1)
    assert tile.shape == (1, nchannels, 1, h, w)
    tile = source3.asarray(x0//10, y0//10, x1//10, y1//10, pixel_size=[10])
    assert tile.shape == (1, nchannels, 1, h//10, w//10)


def check_large_tiff_arrays(input):
    source = TiffSource(input, target_pixel_size=[(10, 'um')])
    image = source.render(source.asarray())
    show_image(image)
    tile = source.render(source.asarray(1100, 1200, 2100, 1400))
    show_image(tile)
    tile = source.render(source.asarray(1100, 1200, 2100, 1400, pixel_size=[10]))
    show_image(tile)


if __name__ == '__main__':
    path = 'D:/slides/EM04573_01small.ome.tif'
    path2 = 'D:/slides/test.ome.zarr'
    path3 = 'D:/slides/test.ome.tiff'
    output_params = {'tile_size': [256, 256], 'npyramid_add': 3, 'pyramid_downsample': 2}

    #load_as_zarr('D:/slides/EM04573_01/EM04573_01.ome.tif',
    #             20400, 13000, 500, 500)

    #conversion_test()

    #check_large_tiff_arrays(path)

    progress = tqdm(range(2))
    convert_tiff_to_zarr(path, path2, output_params)
    progress.update(0)
    convert_zarr_to_tiff(path2, path3, output_params)
    progress.update(1)

    print('Done')
