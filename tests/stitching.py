import glob
import numpy as np
import os

import tifffile
from tifffile import xml2dict
from tqdm import tqdm

from OmeSliCC.conversion import store_tiles
from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.image_util import *


def get_normalisation_images(composition_metadata, quantiles, scale=1.0):
    images = []
    for quantile in quantiles:
        filename = os.path.join(os.path.dirname(output_filename), f'tile_quantile_{quantile}.tiff')
        if os.path.exists(filename):
            images.append(tifffile.imread(filename))
    if len(images) < len(quantiles):
        images = create_normalisation_images(composition_metadata, quantiles, scale)
        for image, quantile in zip(images, quantiles):
            filename = os.path.join(os.path.dirname(output_filename), f'tile_quantile_{quantile}.tiff')
            tifffile.imwrite(filename, image, compression='LZW')
            image = blur_image(image, 10)
            filename = os.path.join(os.path.dirname(output_filename), f'tile_quantile_{quantile}_smooth.tiff')
            tifffile.imwrite(filename, image, compression='LZW')
    return images


def create_normalisation_images(composition_metadata, quantiles, scale=1.0):
    source0 = TiffSource(tile_filenames[0])
    size0 = source0.get_size()
    new_size = tuple(np.round(np.multiply(size0, scale)).astype(int))
    nchannels = len(source0.get_channels())
    channel_images = []
    for channeli in range(nchannels):
        # filter edge tiles
        positions = [(metadata['Bounds']['StartX'], metadata['Bounds']['StartY']) for metadata in composition_metadata]
        edges = np.array((np.min(positions, 0), np.max(positions, 0)))
        x_edges, y_edges = edges[:, 0], edges[:, 1]
        filtered_tiles = []
        for metadata in composition_metadata:
            if metadata['Bounds']['StartX'] not in x_edges and metadata['Bounds']['StartY'] not in y_edges:
                filename0 = metadata['Filename']
                for tile_filename in tile_filenames:
                    if filename0 in tile_filename:
                        filtered_tiles.append(tile_filename)

        images = []
        print('Loading tiles')
        for tile_filename in tqdm(tile_filenames):
            image = TiffSource(tile_filename).asarray()[..., channeli]
            if scale != 1:
                image = image_resize(image, new_size)
            images.append(image)

        # filter non-empty tiles
        median_image = calc_tiles_median(images)
        print('Filtering tiles with signal')
        difs = [np.mean(np.abs(image.astype(np.float32) - median_image.astype(np.float32)), (0, 1)) for image in images]
        threshold = np.mean(difs, 0)
        images = [image for image, dif in zip(images, difs) if np.all(dif < threshold)]

        norm_images0 = calc_tiles_quantiles(images, quantiles)
        norm_images = []
        for image in norm_images0:
            if scale != 1:
                image = image_resize(image, size0)
            norm_images.append(image)
        channel_images.append(norm_images)

    quatile_images = []
    for quatilei in range(len(quantiles)):
        quatile_image = None
        for channel_image in channel_images:
            image = channel_image[quatilei]
            if quatile_image is None:
                quatile_image = image
            else:
                quatile_image = cv.merge(list(cv.split(quatile_image)) + [image])
        quatile_images.append(quatile_image)
    return quatile_images


def flatfield_correction(image0, dark=0, bright=1, clip=True):
    # https://imagej.net/plugins/bigstitcher/flatfield-correction
    mean_bright_dark = np.mean(bright - dark, (0, 1))
    image = (image0 - dark) * mean_bright_dark / (bright - dark)
    if clip:
        image = np.clip(image, 0, 1)
    return image


def do_flatfield_correction(image):
    dtype = image.dtype
    image = float2int_image(flatfield_correction(int2float_image(image), dark), dtype)
    return image


def normalise(image):
    min, max = get_image_quantile(image, 0.05), get_image_quantile(image, 0.95)
    return normalise_values(image, min, max)


if __name__ == '__main__':
    tile_path = 'D:/slides/EM04573_01t/Multichannel tiles/*.tif*'
    tile_filenames = glob.glob(tile_path)
    metadata_filename = 'D:/slides/EM04573_01t/EM04573_01_20x_beads-07_info_ome_tiff.xml'
    composition_metadata = xml2dict(open(metadata_filename, encoding='utf8').read())['ExportDocument']['Image']

    #tile_filenames=tile_filenames[331:333]

    params = {'output': {'tile_size': [1024, 1024]}}#, 'npyramid_add': 4, 'pyramid_downsample': 2}}
    output_filename = 'D:/slides/EM04573_01t/tiles_small.ome.zarr'

    #flatfield_quantiles = [0.05]
    #flatfield_scale = 1.0
    #norm_images = get_normalisation_images(composition_metadata, flatfield_quantiles, scale=flatfield_scale)
    #dark = norm_images[0]

    sources = [TiffSource(tile_filename) for tile_filename in tqdm(tile_filenames)]

    #store_tiles(sources, output_filename, params, composition_metadata, image_operations=[do_flatfield_correction])
    store_tiles(sources, output_filename, params, composition_metadata)
