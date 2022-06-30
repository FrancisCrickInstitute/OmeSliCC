import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tifffile
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray
from imagecodecs import jpeg2k_encode, jpeg2k_decode
from tifffile import TiffFile, TiffPage

from src.util import tags_to_dict, print_dict, print_hbytes


def show_image(image):
    plt.imshow(image)
    plt.show()


def show_image_gray(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def get_best_mag(mags, target_mag):
    # find smallest mag larger/equal to target mag
    best_mag = None
    best_index = -1
    best_scale = 0
    for index, mag in enumerate(mags):
        scale = target_mag / mag
        if 1 >= scale > best_scale:
            best_index = index
            best_mag = mag
            best_scale = scale
    return best_mag, best_index


def get_best_size(sizes, target_size):
    # find largest scale but smaller to 1
    best_index = -1
    best_scale = 0
    for index, size in enumerate(sizes):
        scale = np.mean(np.divide(target_size, size))
        if 1 >= scale > best_scale:
            best_index = index
            best_scale = scale
    return sizes[best_index], best_index


def calc_pyramid(size, pyramid_add=0, pyramid_downsample=4.0):
    width, height = size
    sizes_add = []
    scale = 1
    for _ in range(pyramid_add):
        scale /= pyramid_downsample
        sizes_add.append((int(round(width * scale)), int(round(height * scale))))
    return sizes_add


def find_ome_magnification(metadata0):
    mag = 0
    if 'OME' in metadata0:
        metadata = metadata0['OME']
    else:
        metadata = metadata0
    try:
        mag = metadata['Instrument']['Objective']['NominalMagnification']
    except:
        pass
    return mag


def scale_image(image, new_size):
    return cv.resize(image, new_size)


def precise_resize(image, scale):
    h, w = np.ceil(image.shape[0:2] * scale).astype(int)
    new_image = np.zeros((h, w, image.shape[2]), dtype=np.float32)
    step_size = 1 / scale
    totn = np.round(step_size[0]) * np.round(step_size[1])
    for y in range(h):
        for x in range(w):
            y0, y1 = np.round([y * step_size[1], (y + 1) * step_size[1]]).astype(int)
            x0, x1 = np.round([x * step_size[0], (x + 1) * step_size[0]]).astype(int)
            value = np.sum(image[y0:y1, x0:x1], axis=(0, 1)) / totn
            new_image[y, x] = value
    return new_image


def load_tiff(filename, only_tiled=True):
    tiff = TiffFile(filename)
    pages = get_tiff_pages(tiff, only_tiled=only_tiled)
    page0 = pages[0]
    image = page0.asarray()
    if tiff.ome_metadata:
        metadata = tifffile.xml2dict(tiff.ome_metadata)
    elif tiff.metaseries_metadata:
        metadata = tiff.metaseries_metadata
    elif tiff.imagej_metadata:
        metadata = tiff.imagej_metadata
    else:
        metadata = tags_to_dict(page0.tags)
    return image, metadata


def get_tiff_pages(tiff, only_tiled=False):
    pages = []
    for page in tiff.pages:
        if isinstance(page, TiffPage):
            page_added = False
            for serie in tiff.series:
                # has series
                serie_added = False
                for level in serie.levels:
                    # has levels
                    pages.append(level.keyframe)
                    serie_added = True
                    page_added = True
                if not serie_added:
                    pages.append(serie.keyframe)
                    page_added = True
            if not page_added:
                pages.append(page)

    if only_tiled:
        tiled_pages = []
        for page in pages:
            if page.is_tiled:
                tiled_pages.append(page)
        pages = tiled_pages

    return pages


def tiff_info(filename):
    s = ''
    nom_size = 0
    tiff = TiffFile(filename)
    real_size = tiff.fstat.st_size
    s += str(tiff) + '\n'
    if tiff.ome_metadata:
        print(tiff.ome_metadata)
        s += f'OME: {print_dict(tifffile.xml2dict(tiff.ome_metadata))}\n'
    if tiff.metaseries_metadata:
        s += f'Series: {tiff.metaseries_metadata}\n'
    if tiff.imagej_metadata:
        s += f'ImageJ: {tiff.imagej_metadata}\n'

    for page in get_tiff_pages(tiff):
        s += str(page) + '\n'
        s += f'Size: {page.shape[1]} {page.shape[0]} {page.shape[2]} ({print_hbytes(page.size)})\n'
        if page.is_tiled:
            s += f'Tiling: {page.tilewidth} {page.tilelength} {page.tiledepth}\n'
        s += f'Compression: {str(page.compression)} jpegtables: {page.jpegtables is not None}\n'
        tag_dict = tags_to_dict(page.tags)
        if 'TileOffsets' in tag_dict:
            tag_dict.pop('TileOffsets')
        if 'TileByteCounts' in tag_dict:
            tag_dict.pop('TileByteCounts')
        if 'ImageDescription' in tag_dict and tag_dict['ImageDescription'].startswith('<?xml'):
            # redundant
            tag_dict.pop('ImageDescription')
        s += print_dict(tag_dict, compact=True) + '\n\n'
        nom_size += page.size

    s += f'Overall compression: 1:{nom_size / real_size:.1f}'
    return s


def tiff_info_short(filename):
    nom_size = 0
    tiff = TiffFile(filename)
    s = str(filename)
    real_size = tiff.fstat.st_size
    for page in tiff.pages:
        s += ' ' + str(page)
        nom_size += page.size
    s += f' Image size:{nom_size} File size:{real_size} Overall compression: 1:{nom_size / real_size:.1f}'
    return s


class JPEG2000(Codec):
    codec_id = "JPEG2000"

    def __init__(self, level=None):
        self.level = level

    def encode(self, buf):
        if self.level is not None:
            return jpeg2k_encode(ensure_ndarray(buf), level=self.level)
        else:
            return jpeg2k_encode(ensure_ndarray(buf))

    def decode(self, buf):
        return jpeg2k_decode(ensure_ndarray(buf))


def compare_image(image0, image1, show=False):
    dif, dif_max, dif_mean, psnr = compare_image_dist(image0, image1)
    print(f'rgb dist max: {dif_max:.1f} mean: {dif_mean:.1f} PSNR: {psnr:.1f}')
    if show:
        show_image(dif)
        show_image((dif * 10).astype(np.uint8))
    return dif


def compare_image_dist(image0, image1):
    dif = cv.absdiff(image0, image1)
    psnr = cv.PSNR(image0, image1)
    if dif.size > 1000000000:
        # split very large array
        rgb_maxs = []
        rgb_means = []
        for dif1 in np.array_split(dif, 16):
            rgb_dif = np.linalg.norm(dif1, axis=2)
            rgb_maxs.append(np.max(rgb_dif))
            rgb_means.append(np.mean(rgb_dif))
        rgb_max = np.max(rgb_maxs)
        rgb_mean = np.mean(rgb_means)
    else:
        rgb_dif = np.linalg.norm(dif, axis=2)
        rgb_max = np.max(rgb_dif)
        rgb_mean = np.mean(rgb_dif)
    return dif, rgb_max, rgb_mean, psnr


def calc_fraction_used(image, threshold=0.1):
    low = int(round(threshold * 255))
    high = int(round((1 - threshold) * 255))
    shape = image.shape
    total = shape[0] * shape[1]
    good = 0
    for y in range(shape[0]):
        for x in range(shape[1]):
            pixel = image[y, x]
            if low <= pixel[0] < high and low <= pixel[1] < high and low <= pixel[2] < high:
                good += 1
    fraction = good / total
    return fraction
