# https://pyquestions.com/how-to-save-a-very-large-numpy-array-as-an-image-loading-as-little-as-possible-into-memory
# * TODO: fix Zarr support, extend to Ome.Zarr
# * TODO: Add JPEGXR support for Zarr


import os
import numpy as np
import zarr
import cv2 as cv
from PIL import Image
from numcodecs import register_codec
from numcodecs.blosc import Blosc
from tifffile import TiffWriter

from src.BioSource import BioSource
from src.PlainImageSource import PlainImageSource
from src.TiffSource import TiffSource
from src.ZarrSource import ZarrSource
from src.image_util import JPEG2000, image_resize, get_image_size_info, calc_pyramid
from src.util import get_filetitle

register_codec(JPEG2000)


def load_source(filename):
    ext = os.path.splitext(filename)[1].lower()
    if 'zarr' in ext:
        source = ZarrSource(filename)
    elif 'tif' in ext or 'svs' in ext:
        source = TiffSource(filename)
    elif ext in Image.registered_extensions().keys():
        source = PlainImageSource(filename)
    else:
        source = BioSource(filename)
    return source


def get_image_info(filename):
    source = load_source(filename)
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


def extract_thumbnail(filename, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    source = load_source(filename)
    size = source.sizes[0]

    #thumbsize = np.int0(np.divide(size, 10))    # use arbitrary factor 10
    factor = np.max(np.divide(size, 1000))
    thumbsize = np.round(np.divide(size, factor)).astype(int)

    # write thumbnail to file
    thumb = source.get_thumbnail(thumbsize)
    nchannels = thumb.shape[2] if len(thumb.shape) > 2 else 1
    if nchannels == 2:
        for channeli in range(nchannels):
            output_filename = os.path.join(output_folder, f'{get_filetitle(filename)}_channel{channeli}_thumb.tiff')
            cv.imwrite(output_filename, thumb[..., channeli])
    else:
        output_filename = os.path.join(output_folder, get_filetitle(filename) + '_thumb.tiff')
        cv.imwrite(output_filename, thumb)
    return thumb


def convert(filename, output_params):
    output_folder = output_params['folder']
    output_format = output_params['format']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, get_filetitle(filename, remove_all_ext=True) + '.' + output_format)
    source = load_source(filename)
    if 'zar' in output_format:
        convert_to_zarr(source, output_filename, output_params)
    elif 'ome' in output_format:
        convert_to_tiff(source, output_filename, output_params, ome=True)
    else:
        convert_to_tiff(source, output_filename, output_params)


def convert_to_zarr0(input_filename, output_folder, patch_size=(256, 256)):
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


def convert_to_zarr(source, output_filename, output_params):
    shape = source.get_shape()
    dtype = source.get_pixel_type()
    tile_size = output_params['tile_size']
    compression = output_params.get('compression')

    zarr_root = zarr.open_group(output_filename, mode='w')
    zarr_data = zarr_root.create_dataset(str(0), shape=shape, chunks=(tile_size[0], tile_size[1], None), dtype=dtype,
                                         compressor=None, filters=compression)
    return zarr_data


def convert_to_tiff(source, output_filename, output_params, ome=False):
    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    channel_operation = output_params.get('channel_operation')
    output_format = output_params['format']

    npyramid_add = output_params.get('npyramid_add', 0)
    pyramid_downsample = output_params.get('pyramid_downsample')
    if npyramid_add > 0:
        pyramid_sizes_add = calc_pyramid(source.get_size(), npyramid_add, pyramid_downsample)
    else:
        pyramid_sizes_add = None

    image = source.clone_empty()
    chunk_size = (10240, 10240)
    for x0, y0, x1, y1, chunk in source.produce_chunks(chunk_size):
        image[y0:y1, x0:x1] = chunk

    if ome:
        metadata = None
        xml_metadata = source.get_xml_metadata(output_filename, pyramid_sizes_add=pyramid_sizes_add)
    else:
        metadata = source.get_metadata()
        xml_metadata = None

    if channel_operation == 'split' and len(image.shape) > 2 and image.shape[2] > 1:
        for channeli in range(image.shape[2]):
            image0 = image[..., channeli]
            output_filename0 = output_filename.replace(output_format, '').rstrip('.') + f'_channel{channeli}.' + output_format
            save_tiff(output_filename0, image0, metadata=metadata, xml_metadata=xml_metadata, tile_size=tile_size, compression=compression,
                      npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample)
    else:
        save_tiff(output_filename, image, metadata=metadata, xml_metadata=xml_metadata, tile_size=tile_size, compression=compression,
                  npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample)


def save_tiff(filename, data, metadata=None, xml_metadata=None, tile_size=None, compression=None,
              npyramid_add=0, pyramid_downsample=4.0, pyramid_sizes_add=None):
    # Use tiled(/slower?) writing:
    # writer.write(tile_iterator, shape=shape_size_at_desired_mag_pyramid_scale, tile=tile_size)
    if xml_metadata is not None:
        xml_metadata_bytes = xml_metadata.encode()
    else:
        xml_metadata_bytes = None
    width, height = data.shape[1], data.shape[0]
    with TiffWriter(filename, bigtiff=True) as writer:
        if pyramid_sizes_add is not None:
            npyramid_add = len(pyramid_sizes_add)

        writer.write(data, subifds=npyramid_add,
                     tile=tile_size, compression=compression, metadata=metadata, description=xml_metadata_bytes)

        scale = 1
        resized_image = data
        for i in range(npyramid_add):
            if pyramid_sizes_add is not None:
                new_width, new_height = pyramid_sizes_add[i]
            else:
                scale /= pyramid_downsample
                new_width, new_height = np.int0(np.round(np.array([width, height]) * scale))
            resized_image = image_resize(resized_image, (new_width, new_height))
            writer.write(resized_image, subfiletype=1,
                         tile=tile_size, compression=compression)
