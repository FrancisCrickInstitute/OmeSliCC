import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tifffile
from PIL.ExifTags import TAGS
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray
import imagecodecs
from imagecodecs import jpeg2k_encode, jpeg2k_decode
from tifffile import TiffFile, TiffPage, TiffFrame

from src.util import tags_to_dict, print_dict, print_hbytes


def check_versions():
    print(f'tifffile {tifffile.__version__}')
    print(imagecodecs.version())


def show_image(image):
    plt.imshow(image)
    plt.show()


def show_image_gray(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def get_image_size_info(xyzct, pixel_nbytes, pixel_type, channel_info):
    w, h, zs, cs, ts = xyzct
    size = print_hbytes(w * h * zs * cs * ts * pixel_nbytes)
    if (len(channel_info) == 1 and channel_info[0][1] == 3) or len(channel_info) == 3:
        channel_infos = 'rgb'
    else:
        channel_infos = ','.join([channel[0] for channel in channel_info])
    image_size_info = f'Size: {w} x {h} x {zs} C: {cs} T: {ts}\nUncompressed: {size} Pixel type: {pixel_type}'
    if channel_infos != '':
        image_size_info += f' Channels: {channel_infos}'
    return image_size_info


def pilmode_to_pixelinfo(mode):
    pixelinfo = (np.uint8, 8)
    mode_types = {'I': (np.uint32, 32), 'F': (np.float32, 32)}
    if mode in mode_types:
        pixelinfo = mode_types[mode]
    pixelinfo = (np.dtype(pixelinfo[0]), pixelinfo[1])
    return pixelinfo


def calc_pyramid(size, npyramid_add=0, pyramid_downsample=4.0):
    width, height = size
    sizes_add = []
    scale = 1
    for _ in range(npyramid_add):
        scale /= pyramid_downsample
        sizes_add.append((int(round(width * scale)), int(round(height * scale))))
    return sizes_add


def image_resize_fast(image, target_size):
    return cv.resize(image, target_size, interpolation=cv.INTER_AREA)


def image_resize(image, target_size0):
    if not isinstance(image, np.ndarray):
        image = image.asarray()
    if image.dtype == np.int8:
        image = image.astype(np.uint8)
    target_size = np.clip(np.int0(np.round(target_size0)), 1, None)
    new_image = cv.resize(image, tuple(target_size), interpolation=cv.INTER_AREA)
    return new_image


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
        if isinstance(page, TiffPage) or isinstance(page, TiffFrame):
            for serie in tiff.series:
                # has series
                for level in serie.levels:
                    # has levels
                    if not level.keyframe in pages:
                        pages.append(level.keyframe)
                if not serie.keyframe in pages:
                    pages.append(serie.keyframe)
            if not page in pages:
                pages.append(page)

    if only_tiled:
        tiled_pages = []
        for page in pages:
            if isinstance(page, TiffPage) and page.is_tiled:
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


def get_pil_metadata(image):
    metadata = {}
    exifdata = image.getexif()
    for tag_id in exifdata:
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        if isinstance(data, bytes):
            data = data.decode()
        metadata[tag] = data
    if metadata == {}:
        metadata = image.info
    return metadata


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
