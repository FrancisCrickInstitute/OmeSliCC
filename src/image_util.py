import PIL.Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tifffile
from PIL.ExifTags import TAGS
import imagecodecs
from tifffile import TiffFile

from src.util import tags_to_dict, print_dict, print_hbytes


def check_versions():
    print(f'tifffile {tifffile.__version__}')
    print(imagecodecs.version())


def show_image(image: np.ndarray):
    plt.imshow(image)
    plt.show()


def show_image_gray(image: np.ndarray):
    plt.imshow(image, cmap='gray')
    plt.show()


def ensure_unsigned_type(dtype: np.dtype) -> np.dtype:
    if dtype.kind == 'i':
        unsigned_type = np.dtype(f'u{dtype.itemsize}')
        return unsigned_type
    else:
        return dtype


def ensure_unsigned_image(image0: np.ndarray) -> np.ndarray:
    dtype0 = image0.dtype
    dtype = ensure_unsigned_type(dtype0)
    if dtype != dtype0:
        # conversion without overhead
        offset = 2 ** (8 * dtype.itemsize - 1)
        image = image0.astype(dtype) + offset
    else:
        image = image0
    return image


def convert_image_sign_type(image0: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if image0.dtype.kind == dtype.kind:
        image = image0
    elif image0.dtype.kind == 'i':
        image = ensure_unsigned_image(image0)
    else:
        # conversion without overhead
        offset = 2 ** (8 * dtype.itemsize - 1)
        image = (image0 - offset).astype(dtype)
    return image


def get_image_size_info(xyzct: tuple, pixel_nbytes: int, pixel_type: np.dtype, channel_info: list) -> str:
    w, h, zs, cs, ts = xyzct
    size = print_hbytes(np.int64(pixel_nbytes) * w * h * zs * cs * ts)
    if (len(channel_info) == 1 and channel_info[0][1] == 3) or len(channel_info) == 3:
        channel_infos = 'rgb'
    else:
        channel_infos = ','.join([channel[0] for channel in channel_info])
    image_size_info = f'Size: {w} x {h} x {zs} C: {cs} T: {ts}\nUncompressed: {size} Pixel type: {pixel_type}'
    if channel_infos != '':
        image_size_info += f' Channels: {channel_infos}'
    return image_size_info


def pilmode_to_pixelinfo(mode: str) -> tuple:
    pixelinfo = (np.uint8, 8)
    mode_types = {'I': (np.uint32, 32), 'F': (np.float32, 32)}
    if mode in mode_types:
        pixelinfo = mode_types[mode]
    pixelinfo = (np.dtype(pixelinfo[0]), pixelinfo[1])
    return pixelinfo


def calc_pyramid(xyzct: tuple, npyramid_add: int = 0, pyramid_downsample: float = 4.0,
                 volumetric_resize: bool = False) -> list:
    x, y, z, c, t = xyzct
    if volumetric_resize and z > 1:
        size = (x, y, z)
    else:
        size = (x, y)
    sizes_add = []
    scale = 1
    for _ in range(npyramid_add):
        scale /= pyramid_downsample
        sizes_add.append(np.round(np.multiply(size, scale)).astype(int))
    return sizes_add


def image_resize_fast(image: np.ndarray, target_size: tuple) -> np.ndarray:
    return cv.resize(image, target_size, interpolation=cv.INTER_AREA)


def image_resize(image: np.ndarray, target_size0: tuple) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        image = image.asarray()
    dtype0 = image.dtype
    image = ensure_unsigned_image(image)
    target_size = tuple(np.clip(np.int0(np.round(target_size0)), 1, None))
    if len(image.shape) > 2 and image.shape[2] > 4:
        # volumetric
        target_shape = (target_size[1], target_size[0], image.shape[2])
        new_image = np.zeros(target_shape, dtype=image.dtype)
        for z in range(target_shape[2]):
            new_image[..., z] = cv.resize(image[..., z], target_size, interpolation=cv.INTER_AREA)
    else:
        new_image = cv.resize(image, target_size, interpolation=cv.INTER_AREA)
    new_image = convert_image_sign_type(new_image, dtype0)
    return new_image


def precise_resize(image: np.ndarray, scale: np.ndarray) -> np.ndarray:
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


def get_tiff_pages(tiff: TiffFile) -> list:
    pages = []
    found = False
    for serie in tiff.series:
        # has series
        for level in serie.levels:
            # has levels
            level_pages = []
            for page in level.pages:
                found = True
                level_pages.append(page)
            if len(level_pages) > 0:
                pages.append(level_pages)

    if not found:
        for page in tiff.pages:
            pages.append(page)
    return pages


def get_resolution_from_pixel_size(pixel_size: list, scale_factor: float = 1) -> tuple:
    conversions = {
        'cm': (1, 'centimeter'),
        'mm': (1, 'millimeter'),
        'µm': (1, 'micrometer'),
        'nm': (1000, 'micrometer'),
        'nanometer': (1000, 'micrometer'),
    }
    resolutions = []
    resolutions_unit = None
    if len(pixel_size) > 0:
        units = []
        for size, unit in pixel_size:
            if size != 0 and size != 1:
                resolution = 1 / (size * scale_factor)
                resolutions.append(resolution)
                if unit != '':
                    units.append(unit)
        if len(units) > 0:
            resolutions_unit = units[0]
            if resolutions_unit in conversions:
                conversion = conversions[resolutions_unit]
                resolutions = list(np.multiply(resolutions, conversion[0]))
                resolutions_unit = conversion[1]
    if len(resolutions) == 0:
        resolutions = None
    return resolutions, resolutions_unit


def tiff_info(filename: str) -> str:
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


def tiff_info_short(filename: str) -> str:
    nom_size = 0
    tiff = TiffFile(filename)
    s = str(filename)
    real_size = tiff.fstat.st_size
    for page in tiff.pages:
        s += ' ' + str(page)
        nom_size += page.size
    s += f' Image size:{nom_size} File size:{real_size} Overall compression: 1:{nom_size / real_size:.1f}'
    return s


def get_pil_metadata(image: PIL.Image) -> dict:
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


def compare_image(image0, image1, show=False) -> float:
    dif, dif_max, dif_mean, psnr = compare_image_dist(image0, image1)
    print(f'rgb dist max: {dif_max:.1f} mean: {dif_mean:.1f} PSNR: {psnr:.1f}')
    if show:
        show_image(dif)
        show_image((dif * 10).astype(np.uint8))
    return dif


def compare_image_dist(image0: np.ndarray, image1: np.ndarray) -> tuple:
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


def calc_fraction_used(image: np.ndarray, threshold: float = 0.1) -> float:
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


def reverse_last_axis(image, reverse=True):
    if reverse and len(image.shape) > 2 and image.shape[-1] > 1:
        return np.moveaxis(image, -1, 0)
    else:
        return image
