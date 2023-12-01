import glob
import os
import numpy as np
import random
import zarr
from imageio.v3 import imread
from tqdm import tqdm
from timeit import default_timer as timer

from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.image_util import *


def test_load(filename: str, pixel_size: list = None, position: tuple = None, size: tuple = None) -> np.ndarray:
    source = TiffSource(filename, pixel_size)
    if position is None:
        position = (0, 0)
    if size is None:
        size = source.get_size()
    image = source.asarray(position[0], position[1], position[0] + size[0], position[1] + size[1])
    return image


def compare_image_tiles(filename1: str, filename2: str, bits_per_channel: int = 8, tile_size: tuple = (512, 512)) -> tuple:
    source1 = TiffSource(filename1)
    source2 = TiffSource(filename2)

    w, h = source1.get_size()
    tw, th = tile_size
    nx = int(np.ceil(w / tw))
    ny = int(np.ceil(h / th))

    difs_max = []
    difs_mean = []
    mses = []

    for y in tqdm(range(ny)):
        for x in range(nx):
            rx = x * tw
            ry = y * th
            rx2 = min(rx + tw, w)
            ry2 = min(ry + th, h)
            patch1 = source1.asarray(rx, ry, rx2, ry2)
            patch2 = source2.asarray(rx, ry, rx2, ry2)
            dif, dif_max, dif_mean, _ = compare_image_dist(patch1, patch2)
            mse = np.mean(dif.astype(float) ** 2)
            difs_max.append(dif_max)
            difs_mean.append(dif_mean)
            mses.append(mse)

    dif_max = np.max(difs_max)
    dif_mean = np.mean(difs_mean)
    maxval = 2 ** bits_per_channel - 1
    psnr = 20 * np.log10(maxval / np.sqrt(np.mean(mses)))    # recalculate PSNR based on mean MSE
    print(f'rgb dist max: {dif_max:.1f} mean: {dif_mean:.1f} PSNR: {psnr:.1f}')
    return dif_max, dif_mean, psnr


def test_read_source(image_filename: str, n: int = 1000):
    print('Test read source')
    print(tiff_info(image_filename))
    source = TiffSource(image_filename, 40)
    size = source.get_size()
    width = size[0]
    height = size[1]
    nx = int(np.ceil(width / patch_size[0]))
    ny = int(np.ceil(height / patch_size[1]))

    source.load()

    start = timer()
    thumb = source.get_thumbnail((nx, ny))
    elapsed = timer() - start
    print(f'thumbnail time : {elapsed:.3f}')
    show_image(thumb)

    start = timer()
    for _ in tqdm(range(n)):
        xi = random.randrange(nx)
        yi = random.randrange(ny)
        x = xi * patch_size[0]
        y = yi * patch_size[1]
        image = source.asarray(x, y, x + patch_size[0], y + patch_size[1])
        image.shape
        #show_image(image)
    elapsed = timer() - start
    print(f'time (total/step): {elapsed:.3f} / {elapsed / n:.3f}')


def load_zarr_test(image_filename: str) -> np.ndarray:
    zarr_filename = os.path.splitext(image_filename)[0] + '.zarr'
    zarr_root = zarr.open_group(zarr_filename, mode='r')
    zarr_data = zarr_root.get(str(0))

    patchx = 200
    patchy = 200
    ys = patchy * patch_size[1]
    ye = ys + patch_size[1]
    xs = patchx * patch_size[0]
    xe = xs + patch_size[0]
    tile = zarr_data[ys:ye, xs:xe]
    show_image(tile)
    return tile


def test_read_zarr(image_filename: str, n: int = 1000):
    print('Test read zarr')
    zarr_filename = os.path.splitext(image_filename)[0] + '.zarr'
    zarr_root = zarr.open_group(zarr_filename, mode='r')
    zarr_data = zarr_root.get(str(0))
    shape = zarr_data.shape
    width = shape[1]
    height = shape[0]
    nx = int(np.ceil(width / patch_size[0]))
    ny = int(np.ceil(height / patch_size[1]))

    for _ in tqdm(range(n)):
        xi = random.randrange(nx)
        yi = random.randrange(ny)
        xs = xi * patch_size[0]
        ys = yi * patch_size[1]
        xe = xs + patch_size[0]
        ye = ys + patch_size[1]
        tile = zarr_data[ys:ye, xs:xe]


def calc_images_fraction(pattern: str):
    for filename in glob.glob(pattern):
        print(f'{os.path.splitext(os.path.basename(filename))[0]}', end='\t')
        try:
            image = imread(filename)
            fraction = calc_fraction_used(image)
            print(f'fraction:{fraction:.3f}')
        except Exception as e:
            print(e)


if __name__ == '__main__':
    image_dir = 'resources/images/'
    patch_size = (256, 256)
    os.chdir('../')

    # perform test
    #test_load('D:/slides/Pharos_test_images/01-08-23_test2__33.tiff')
    test_load('D:/slides/Pharos_test_images/Testing7.tiff')
