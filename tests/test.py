import glob
import imageio.v3
import numpy as np
import os
import random
import zarr
from imageio.v3 import imread
from tqdm import tqdm
from timeit import default_timer as timer

from OmeSliCC.OmeZarrSource import OmeZarrSource
from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.conversion import save_tiff
from OmeSliCC.image_util import *


def render_at_pixel_size(filename: str, source_pixel_size: list = None,
                         target_pixel_size: list = None, **indices) -> np.ndarray:
    if filename.endswith('.zarr'):
        source = OmeZarrSource(filename, source_pixel_size)
    else:
        source = TiffSource(filename, source_pixel_size)
    image0 = source.asarray(pixel_size=target_pixel_size, **indices)
    image = source.render(image0)
    return image


def test_load(filename: str, pixel_size: tuple = None, **indices) -> np.ndarray:
    source = TiffSource(filename, pixel_size)
    image = source.asarray(**indices)
    return image


def test_extract_metadata(path: str):
    is_ome = '.ome.' in path
    if not is_ome:
        print('### immeta')
        print(print_dict(imageio.v3.immeta(path)))
        print()

    print('### improps')
    print(imageio.v3.improps(path))
    print()

    print('### source metadata')
    source = TiffSource(path)
    metadata = source.metadata.copy()
    metadata.pop('StripOffsets', None)
    metadata.pop('StripByteCounts', None)
    print(print_dict(metadata))


def compare_images(filename1: str, filename2: str, tile_size: tuple = None) -> dict:
    source1 = TiffSource(filename1)
    source2 = TiffSource(filename2)
    w, h = source1.get_size()
    if tile_size is None:
        tile_size = w, h
    tw, th = tile_size
    nx = int(np.ceil(w / tw))
    ny = int(np.ceil(h / th))
    bits_per_channel = source1.pixel_nbits[0]

    difs_max = []
    difs_mean = []
    mses = []

    for y in tqdm(range(ny)):
        for x in range(nx):
            rx = x * tw
            ry = y * th
            rx2 = min(rx + tw, w)
            ry2 = min(ry + th, h)
            patch1 = source1.asarray(x0=rx, x1=rx2, y0=ry, y1=ry2)
            patch2 = source2.asarray(x0=rx, x1=rx2, y0=ry, y1=ry2)
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
    return {'dif_max': dif_max, 'dif_mean': dif_mean, 'psnr': psnr}


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
        image = source.asarray(x0=x, x1=x+patch_size[0], y0=y, y1=y+patch_size[1])
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


def create_ome_tiff(filename):
    data = np.zeros((16, 16, 3), dtype=np.uint8)
    metadata = {'annotations': [{'a': 1}]}
    save_tiff(filename, data, metadata=metadata)


if __name__ == '__main__':
    os.chdir('../')
    output_filename = 'D:/slides/metadata_test.ome.tiff'
    patch_size = (256, 256)

    #path = 'E:/Personal/Crick/slides/test_images/19629.svs'
    path = 'E:/Personal/Crick/slides/test_images/volumetric Broken_NE_cropped.tif'
    #path = 'E:/Personal/Crick/slides/test_images/H&E K130_PR003.ome.tiff'
    #path = 'D:/slides/EM04613/EM04613_04_20x_WF_Reflection-02-Stitching-01.ome.tif'
    #path = 'D:/slides/EM04613/EM04613_04_20x_WF_Reflection-02-Stitching-01.ome.zarr'
    #path = 'D:/slides/12193/data/S000/000_000_0.tiff'

    # perform test
    #print(tiff_info(path))
    #test_load(path)
    #create_ome_tiff(output_filename)
    #print(tiff_info(output_filename))
    #test_extract_metadata(path)
    source = TiffSource(path)
    print('pixel size:', source.get_pixel_size())
    print('pixel size [um]:', source.get_pixel_size_micrometer())
    #show_image(render_at_pixel_size(path, target_pixel_size=[(10, 10)]))
