import glob
import os
import numpy as np
import random
import zarr
from datetime import datetime
from imageio import imread
from ome_types.model.pixels import DimensionOrder
from tqdm import tqdm
from timeit import default_timer as timer

from src.TiffSlide import TiffSlide
from src.conversion import save_tiff
from src.image_util import show_image, compare_image, tiff_info, compare_image_dist, load_tiff, calc_pyramid, \
    calc_fraction_used
from src.ome import create_ome_metadata


def test_load(filename, magnification, position=None, size=None):
    slide = TiffSlide(filename, magnification)
    if position is None:
        position = (0, 0)
    if size is None:
        size = slide.get_size()
    image = slide.asarray(position[0], position[1], position[0] + size[0], position[1] + size[1])
    return image


def compare_image_tiles(filename1, filename2, bits_per_channel=8, tile_size=(512, 512)):
    slide1 = TiffSlide(filename1)
    slide2 = TiffSlide(filename2)

    w, h = slide1.get_size()
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
            patch1 = slide1.asarray(rx, ry, rx2, ry2)
            patch2 = slide2.asarray(rx, ry, rx2, ry2)
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


def test_read_slide(image_filename, n=1000):
    print('Test read slide')
    print(tiff_info(image_filename))
    slide = TiffSlide(image_filename, 40)
    size = slide.get_size()
    width = size[0]
    height = size[1]
    nx = int(np.ceil(width / patch_size[0]))
    ny = int(np.ceil(height / patch_size[1]))

    slide.load()

    start = timer()
    thumb = slide.get_thumbnail((nx, ny))
    elapsed = timer() - start
    print(f'thumbnail time : {elapsed:.3f}')
    show_image(thumb)

    start = timer()
    for _ in tqdm(range(n)):
        xi = random.randrange(nx)
        yi = random.randrange(ny)
        x = xi * patch_size[0]
        y = yi * patch_size[1]
        image = slide.asarray(x, y, x + patch_size[0], y + patch_size[1])
        image.shape
        #show_image(image)
    elapsed = timer() - start
    print(f'time (total/step): {elapsed:.3f} / {elapsed / n:.3f}')


def load_zarr_test(image_filename):
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


def test_read_zarr(image_filename, n=1000):
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


def test_create_slide(infilename, outfilename):
    image, metadata = load_tiff(infilename)
    y, x, c = image.shape
    tile_size = (256, 256)
    #compression = ('JPEG2000', 65)
    compression = ('JPEGXR_NDPI', 75)

    image_info = {'description': 'test image',
                  'size_c': c, 'size_t': 1, 'size_x': x, 'size_y': y, 'size_z': 1,
                  'physical_size_x': 1, 'physical_size_y': 1, 'physical_size_z': 1,
                  'type': str(image.dtype), 'dimension_order': DimensionOrder.XYCZT, 'acquisition_date': datetime.now()}

    channels = [{'samples_per_channel': c}]

    pyramid_sizes_add = calc_pyramid((x, y), pyramid_add=3, pyramid_downsample=4.0)

    metadata = create_ome_metadata(outfilename, image_info, channels=channels, pyramid_sizes_add=pyramid_sizes_add)
    xml_metadata = metadata.to_xml()
    xml_metadata = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_metadata

    save_tiff(outfilename, image, xml_metadata=xml_metadata, tile_size=tile_size, compression=compression,
              pyramid_sizes_add=pyramid_sizes_add)


def calc_images_fraction(pattern):
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
    os.chdir('../../')

    # perform test
