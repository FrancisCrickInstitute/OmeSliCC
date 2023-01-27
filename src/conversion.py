# https://pyquestions.com/how-to-save-a-very-large-numpy-array-as-an-image-loading-as-little-as-possible-into-memory
# * TODO: fix Zarr support, extend to Ome.Zarr
# * TODO: Add JPEGXR support for Zarr


import os
import numpy as np
import zarr
import cv2 as cv
from PIL import Image
from imagecodecs.numcodecs import Jpeg2k, JpegXr
from numcodecs import register_codec
from numcodecs.blosc import Blosc
from tifffile import TIFF, TiffWriter

from src.BioSource import BioSource
from src.OmeSource import OmeSource
from src.PlainImageSource import PlainImageSource
from src.TiffSource import TiffSource
from src.ZarrSource import ZarrSource
from src.image_util import image_resize, get_image_size_info, calc_pyramid, ensure_signed_image, reverse_color_axis, \
    get_resolution_from_pixel_size
from src.util import get_filetitle

register_codec(Jpeg2k)
register_codec(JpegXr)


def load_source(filename: str, params: dict) -> OmeSource:
    source_mag = params['input'].get('mag')
    target_mag = params['output'].get('mag')
    ext = os.path.splitext(filename)[1].lower()
    if 'zarr' in ext:
        source = ZarrSource(filename, source_mag=source_mag, target_mag=target_mag)
    elif ext.lstrip('.') in TIFF.FILE_EXTENSIONS:
        source = TiffSource(filename, source_mag=source_mag, target_mag=target_mag)
    elif ext in Image.registered_extensions().keys():
        source = PlainImageSource(filename, source_mag=source_mag, target_mag=target_mag)
    else:
        source = BioSource(filename, target_mag=target_mag)
    return source


def get_image_info(filename: str, params: dict) -> str:
    source = load_source(filename, params)
    xyzct = source.get_size_xyzct()
    pixel_nbytes = source.get_pixel_nbytes()
    pixel_type = source.get_pixel_type()
    channel_info = source.get_channel_info()
    image_info = os.path.basename(filename) + '\n'
    image_info += get_image_size_info(xyzct, pixel_nbytes, pixel_type, channel_info)
    sizes = source.get_actual_size()
    if len(sizes) > 0:
        image_info += '\nActual size:'
        infos = []
        for size in sizes:
            infos.append(f' {size[0]:.3f} {size[1]}')
        image_info += ' x'.join(infos)
    return image_info


def extract_thumbnail(filename: str, params: dict):
    output_params = params['output']
    output_folder = output_params['folder']
    target_size = output_params.get('thumbnail_size', 1000)
    overwrite = output_params.get('overwrite', True)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = os.path.join(output_folder, f'{get_filetitle(filename)}_thumb.tiff')
    output_filename0 = os.path.join(output_folder, f'{get_filetitle(filename)}_channel0_thumb.tiff')
    if overwrite or (not os.path.exists(output_filename) and not os.path.exists(output_filename0)):
        source = load_source(filename, params)
        size = source.sizes[0]

        if target_size < 1:
            factor = target_size
        else:
            factor = np.max(np.divide(size, target_size))
        thumb_size = np.round(np.divide(size, factor)).astype(int)
        thumb = source.get_thumbnail(thumb_size)
        nchannels = thumb.shape[2] if len(thumb.shape) > 2 else 1
        if nchannels not in [1, 3]:
            for channeli in range(nchannels):
                output_filename = os.path.join(output_folder, f'{get_filetitle(filename)}_channel{channeli}_thumb.tiff')
                cv.imwrite(output_filename, thumb[..., channeli])
        else:
            cv.imwrite(output_filename, thumb)


def convert(filename: str, params: dict):
    output_params = params['output']
    output_folder = output_params['folder']
    output_format = output_params['format']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, get_filetitle(filename, remove_all_ext=True) + '.' + output_format)
    source = load_source(filename, params)
    if 'zar' in output_format:
        convert_to_zarr(source, output_filename, output_params)
    else:
        convert_to_tiff(source, output_filename, output_params, ome=('ome' in output_format))


def convert_to_zarr0(input_filename: str, output_folder: str, patch_size: tuple = (256, 256)):
    source = TiffSource(input_filename)
    size = source.sizes[0]
    width = size[0]
    height = size[1]
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)    # clevel=9

    output_filename = os.path.join(output_folder, get_filetitle(input_filename)) + '.zarr'
    root = zarr.open_group(output_filename, mode='a')

    nx = int(np.ceil(width / patch_size[0]))
    ny = int(np.ceil(height / patch_size[1]))

    # thumbnail
    level = 1
    label = str(level)
    if label not in root.array_keys():
        thumb = np.asarray(source.get_thumbnail((nx, ny)))
        # ensure correct size in case thumb scaled using aspect ratio
        if thumb.shape[1] < nx or thumb.shape[0] < ny:
            if thumb.shape[1] < nx:
                dx = nx - thumb.shape[1]
            else:
                dx = 0
            if thumb.shape[0] < ny:
                dy = ny - thumb.shape[0]
            else:
                dy = 0
            thumb = np.pad(thumb, ((0, dy), (0, dx), (0, 0)), 'edge')
        thumb = thumb[0:ny, 0:nx]
        root.create_dataset(label, data=thumb,
                            compressor=compressor)

    # image
    level = 0
    label = str(level)
    if label not in root.array_keys():
        data = root.create_dataset(label, shape=(height, width, 3),
                                   chunks=(patch_size[0], patch_size[1], None), dtype='uint8',
                                   compressor=compressor)
        for y in range(ny):
            ys = y * patch_size[1]
            h = patch_size[1]
            if ys + h > height:
                h = height - ys
            for x in range(nx):
                xs = x * patch_size[0]
                w = patch_size[0]
                if xs + w > width:
                    w = width - xs
                tile = source.asarray(xs, ys, xs + w, ys + h)
                data[ys:ys+h, xs:xs+w] = tile


def convert_to_zarr(source: OmeSource, output_filename: str, output_params: dict):
    shape = source.get_shape()
    dtype = source.get_pixel_type()
    tile_size = output_params['tile_size']
    compression = output_params.get('compression')
    overwrite = output_params.get('overwrite', True)

    if overwrite or not os.path.exists(output_filename):
        zarr_root = zarr.open_group(output_filename, mode='w')
        zarr_root.create_dataset(str(0), shape=shape, chunks=(tile_size[0], tile_size[1], None), dtype=dtype,
                                 compressor=None, filters=compression)


def convert_to_tiff(source: OmeSource, output_filename: str, output_params: dict, ome: bool = False):
    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    split_channel_files = output_params.get('split_channel_files')
    output_format = output_params['format']
    combine_channels = output_params.get('combine_channels', False)
    overwrite = output_params.get('overwrite', True)

    npyramid_add = output_params.get('npyramid_add', 0)
    pyramid_downsample = output_params.get('pyramid_downsample')
    if npyramid_add > 0:
        pyramid_sizes_add = calc_pyramid(source.get_size(), npyramid_add, pyramid_downsample)
    else:
        pyramid_sizes_add = None

    output_filename0 = output_filename.replace(output_format, '').rstrip('.') + f'_channel0.{output_format}'
    if overwrite or (not os.path.exists(output_filename) and not os.path.exists(output_filename0)):
        image = source.clone_empty()
        # get source image chunks
        chunk_size = (10240, 10240)
        for x0, y0, x1, y1, chunk in source.produce_chunks(chunk_size):
            image[y0:y1, x0:x1] = chunk

        if ome:
            metadata = None
            xml_metadata = source.get_xml_metadata(output_filename, combine_channels=combine_channels, pyramid_sizes_add=pyramid_sizes_add)
            # TODO: fix hack:
            xml_metadata = xml_metadata.replace('Color="-1"', '')
        else:
            metadata = source.get_metadata()
            xml_metadata = None
        resolution, resolution_unit = get_resolution_from_pixel_size(source.pixel_size, source.best_factor)

        nchannels = image.shape[2] if len(image.shape) > 2 else 1
        if split_channel_files and nchannels > 1:
            for channeli in range(image.shape[2]):
                image0 = image[..., channeli]
                output_filename0 = output_filename.replace(output_format, '').rstrip('.') + f'_channel{channeli}.{output_format}'
                save_tiff(output_filename0, image0, metadata=metadata, xml_metadata=xml_metadata, resolution=resolution,
                          resolution_unit=resolution_unit, tile_size=tile_size, compression=compression,
                          combine_channels=combine_channels, npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample)
        else:
            save_tiff(output_filename, image, metadata=metadata, xml_metadata=xml_metadata, resolution=resolution,
                      resolution_unit=resolution_unit, tile_size=tile_size, compression=compression,
                      combine_channels=combine_channels, npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample)


def save_tiff(filename: str, image: np.ndarray, metadata: dict = None, xml_metadata: str = None,
              resolution: tuple = None, resolution_unit: str = None, tile_size: tuple = None, compression: [] = None,
              combine_channels=True, npyramid_add: int = 0, pyramid_downsample: float = 4.0, pyramid_sizes_add: list = None):
    # Use tiled writing (less memory needed but maybe slower):
    # writer.write(tile_iterator, shape=shape_size_at_desired_mag_pyramid_scale, tile=tile_size)

    # image = ensure_signed_image(image)   # * Compression JPEGXR_NDPI does not support signed types

    width, height = image.shape[1], image.shape[0]
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if combine_channels:
        nchannels = 1
    if resolution is not None:
        resolution = tuple(resolution[0:2])
    if pyramid_sizes_add is not None:
        npyramid_add = len(pyramid_sizes_add)
    xml_metadata_bytes = xml_metadata.encode() if xml_metadata is not None else None
    bigtiff = (image.size * image.itemsize > 2 ** 32)       # estimate size (w/o compression or pyramid)

    with TiffWriter(filename, bigtiff=bigtiff) as writer:
        for c in range(nchannels):
            if nchannels > 1:
                image1 = image[..., c]
            else:
                image1 = image
            writer.write(image1, subifds=npyramid_add, resolution=resolution, resolutionunit=resolution_unit,
                         tile=tile_size, compression=compression, metadata=metadata, description=xml_metadata_bytes)
            metadata = None
            xml_metadata_bytes = None
            scale = 1
            for i in range(npyramid_add):
                if pyramid_sizes_add is not None:
                    new_width, new_height = pyramid_sizes_add[i]
                else:
                    scale /= pyramid_downsample
                    if resolution is not None:
                        resolution = tuple(np.divide(resolution, pyramid_downsample))
                    new_width, new_height = np.int0(np.round(np.multiply([width, height], scale)))
                image1 = image_resize(image1, (new_width, new_height))
                writer.write(image1, subfiletype=1, resolution=resolution, resolutionunit=resolution_unit,
                             tile=tile_size, compression=compression)


def save_tiff_test(filename: str, image: np.ndarray, metadata: dict = None, xml_metadata: str = None,
              resolution: tuple = None, resolution_unit: str = None, tile_size: tuple = None, compression: [] = None,
              combine_channels=True, npyramid_add: int = 0, pyramid_downsample: float = 4.0, pyramid_sizes_add: list = None):
    # Use tiled writing (less memory needed but maybe slower):
    # writer.write(tile_iterator, shape=shape_size_at_desired_mag_pyramid_scale, tile=tile_size)

    #data = ensure_signed_image(data)   # * Compression JPEGXR_NDPI does not support signed types

    split_channels = not combine_channels
    width, height = image.shape[1], image.shape[0]
    scale = 1
    if resolution is not None:
        resolution = tuple(resolution[0:2])
    if pyramid_sizes_add is not None:
        npyramid_add = len(pyramid_sizes_add)

    if xml_metadata is not None:
        xml_metadata_bytes = xml_metadata.encode()
    else:
        xml_metadata_bytes = None
    bigtiff = (image.size * image.itemsize > 2 ** 32)       # estimate size (w/o compression or pyramid)
    with TiffWriter(filename, bigtiff=bigtiff) as writer:
        writer.write(reverse_color_axis(image, reverse=split_channels), subifds=npyramid_add, resolution=resolution, resolutionunit=resolution_unit,
                     tile=tile_size, compression=compression, metadata=metadata, description=xml_metadata_bytes)

        for i in range(npyramid_add):
            if pyramid_sizes_add is not None:
                new_width, new_height = pyramid_sizes_add[i]
            else:
                scale /= pyramid_downsample
                if resolution is not None:
                    resolution = tuple(np.divide(resolution, pyramid_downsample))
                new_width, new_height = np.int0(np.round(np.multiply([width, height], scale)))
            image = image_resize(image, (new_width, new_height))
            writer.write(reverse_color_axis(image, reverse=split_channels), subfiletype=1, resolution=resolution, resolutionunit=resolution_unit,
                         tile=tile_size, compression=compression)
