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
from tqdm import tqdm

from src.BioSource import BioSource
from src.PlainImageSource import PlainImageSource
from src.TiffSource import TiffSource
from src.ZarrSource import ZarrSource
from src.image_util import JPEG2000, image_resize, get_image_size_info
from src.util import get_filetitle

register_codec(JPEG2000)


def load_source(filename):
    ext = os.path.splitext(filename)[1].lower()
    if 'zgroup' in ext or (os.path.isdir(filename) and os.path.exists(os.path.join(filename, '.zgroup'))):
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
    if thumb.dtype == np.uint16:
        thumb = (thumb / 256).astype('uint8')
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
    dtype = source.pixel_types[0]
    tile_size = output_params['tile_size']
    compression = output_params.get('compression')

    zarr_root = zarr.open_group(output_filename, mode='w')
    zarr_data = zarr_root.create_dataset(str(0), shape=shape, chunks=(tile_size[0], tile_size[1], None), dtype=dtype,
                                         compressor=None, filters=compression)
    return zarr_data


def convert_to_tiff(source, output_filename, output_params, ome=False):
    image = source.clone_empty()
    chunk_size = (10240, 10240)
    # TODO: remove tqdm
    total = np.prod(np.ceil(source.get_size() / chunk_size))
    for x0, y0, x1, y1, chunk in tqdm(source.produce_chunks(chunk_size), total=total):
        image[y0:y1, x0:x1] = chunk

    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    npyramid_add = output_params.get('npyramid_add')
    pyramid_downsample = output_params.get('pyramid_downsample')
    if ome:
        metadata = None
        xml_metadata = source.get_xml_metadata(output_filename)
    else:
        metadata = source.get_metadata()
        xml_metadata = None
    save_tiff(output_filename, image, metadata=metadata, xml_metadata=xml_metadata, tile_size=tile_size, compression=compression,
              npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample)


def save_tiff(filename, image, metadata=None, xml_metadata=None, tile_size=None, compression=None,
              npyramid_add=0, pyramid_downsample=4.0, pyramid_sizes_add=None):
    if xml_metadata is not None:
        xml_metadata_bytes = xml_metadata.encode()
    else:
        xml_metadata_bytes = None
    width, height = image.shape[1], image.shape[0]
    with TiffWriter(filename, bigtiff=True) as writer:
        if pyramid_sizes_add is not None:
            npyramid_add = len(pyramid_sizes_add)
            scale = 1
            resized_image = image

        writer.write(image, subifds=npyramid_add,
                     tile=tile_size, compression=compression, metadata=metadata, description=xml_metadata_bytes)

        for i in range(npyramid_add):
            if pyramid_sizes_add is not None:
                new_width, new_height = pyramid_sizes_add[i]
            else:
                scale /= pyramid_downsample
                new_width, new_height = int(round(width * scale)), int(round(height * scale))
            resized_image = image_resize(resized_image, (new_width, new_height))
            writer.write(resized_image, subfiletype=1,
                         tile=tile_size, compression=compression)
