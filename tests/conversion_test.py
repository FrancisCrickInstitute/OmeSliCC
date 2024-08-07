import dask.array.image as dask_image
import numpy as np
import time

from OmeSliCC.GeneratorSource import GeneratorSource
from OmeSliCC.OmeZarrSource import OmeZarrSource
from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.conversion import save_image_as_tiff, save_image_as_ome_zarr
from OmeSliCC.image_util import *


def create_source(size, tile_size, dtype, pixel_size):
    source = GeneratorSource(size, tile_size, dtype, pixel_size)
    return source


def load_as_zarr_um(path, x0_um, x1_um, y0_um, y1_um):
    source = TiffSource(path)
    data = source.asarray_um(x0=x0_um, x1=x1_um, y0=y0_um, y1=y1_um)
    image = np.asarray(data)
    image = source.render(image)
    show_image(image)


def generated_conversion_test():
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
    image = source.render(source.asarray())
    x, y = source.get_size()
    show_image(image)
    tile = source.render(source.asarray(x0=x//2, x1=x//2+1000, y0=y//2, y1=y//2+1000))
    show_image(tile)
    tile = source.render(source.asarray(x0=x//2, x1=x//2+1000, y0=y//2, y1=y//2+1000, pixel_size=[10]))
    show_image(tile)


def check_cached_loading(path, test_speed=False):
    source = TiffSource(path)
    pixel_size = [0.1]
    x, y = np.array(source.get_size()) * source.get_pixel_size_micrometer()[:2] / pixel_size / 2
    x1, y1 = x * 1.1, y * 1.1

    print('Read as dask')
    start = time.process_time()
    source._load_as_dask()
    print(f'Load process time:', time.process_time() - start)
    data = source.render(source.asarray(x0=x, x1=x1, y0=y, y1=y1, pixel_size=pixel_size))
    show_image(data)
    if test_speed:
        random_access_test(source, n=100)

    print('Read compressed')
    start = time.process_time()
    source.load()
    print(f'Load process time:', time.process_time() - start)
    data = source.render(source.asarray(x0=x, x1=x1, y0=y, y1=y1, pixel_size=pixel_size))
    show_image(data)
    if test_speed:
        random_access_test(source, n=100)

    print('Read decompressed')
    start = time.process_time()
    source.load(decompress=True)
    print(f'Load process time:', time.process_time() - start)
    data = source.render(source.asarray(x0=x, x1=x1, y0=y, y1=y1, pixel_size=pixel_size))
    show_image(data)
    if test_speed:
        random_access_test(source, n=100)

    source.unload()


def random_access_test(source, patch_size=1000, n=1000):
    size = source.get_size()
    start = time.process_time()
    for _ in range(n):
        x, y = np.random.randint(np.array(size) - patch_size)
        patch = np.asarray(source.asarray(x0=x, x1=x+patch_size, y0=y, y1=y+patch_size))
    print(f'Read {n} patches process time:', time.process_time() - start)


def dask_load_test(filename):
    tiff = TiffFile(filename)
    #data_series_asarray = tiff.series[0].asarray()  # numpy array (loads array!)
    #data_dask_imread = dask_image.imread(filename)  # dask array uses skimage.io.imread internally, loads array anyway!
    #data_dask_imread
    data_aszarr = da.from_zarr(tiff.aszarr(level=0))  # dask array (lazy loading)
    data_aszarr


if __name__ == '__main__':
    path_large = 'D:/slides/EM04573_01/EM04573_01.ome.tif'
    #path_medium = 'D:/slides/EM04676_02/combined.ome.tiff'
    path_medium = 'E:/Personal/Crick/slides/NIH/CS_20231216_K891_V003.ome.tiff'
    #path_medium = 'E:/Personal/Crick/slides/test_images/output/K891_V003.ome.tiff'

    path_small = 'D:/slides/EM04573_01small.ome.tif'
    path2 = 'D:/slides/test.ome.zarr'
    path3 = 'D:/slides/test.ome.tiff'
    output_params = {'tile_size': [256, 256], 'npyramid_add': 3, 'pyramid_downsample': 2, 'compression': 'LZW'}

    #dask_load_test(path_large)

    #load_as_zarr_um(path_large, 20400, 20900, 13000, 13500)

    #check_large_tiff_arrays(path_large)

    check_cached_loading(path_medium, test_speed=False)

    print('Conversion start')
    start = time.process_time()
    source = TiffSource(path_small)
    save_image_as_ome_zarr(source, source.asarray(), path2, output_params)
    print('Saved zarr process time:', time.process_time() - start)

    start = time.process_time()
    source2 = OmeZarrSource(path2)
    save_image_as_tiff(source, source.asarray(), path3, output_params, ome=True)
    print('Saved tiff process time:', time.process_time() - start)

    generated_conversion_test()

    print('Done')
