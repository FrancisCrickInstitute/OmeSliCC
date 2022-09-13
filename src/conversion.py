# * TODO: fix Zarr support, extend to Ome.Zarr
# * TODO: Add JPEGXR support for Zarr


import logging
import os
import numpy as np
import zarr
from PIL import Image
from numcodecs import register_codec
from numcodecs.blosc import Blosc
from tifffile import tifffile, TiffWriter

from src.BioSlide import BioSlide
from src.PlainImageSlide import PlainImageSlide
from src.TiffSlide import TiffSlide
from src.image_util import JPEG2000, tags_to_dict, scale_image, get_image_size_info
from src.parameters import ChannelOperation
from src.util import get_filetitle

register_codec(JPEG2000)


def load_slide(filename):
    ext = os.path.splitext(filename)[1].lower()
    if 'tif' in ext or 'svs' in ext:
        slide = TiffSlide(filename)
    else:
        try:
            slide = PlainImageSlide(filename)
        except:
            slide = BioSlide(filename)
    return slide


def get_image_info(filename):
    slide = load_slide(filename)
    xyzct = slide.sizes_xyzct[0]
    pixel_nbytes = slide.pixel_nbytes[0]
    image_info = os.path.basename(filename) + ' ' + get_image_size_info(xyzct, pixel_nbytes)
    logging.info(image_info)
    return image_info


def extract_thumbnail(filename, output_folder):
    slide = load_slide(filename)
    size = slide.sizes[0]
    thumbsize = np.int0(np.divide(size, 10))
    # write thumbnail to file
    thumb = slide.get_thumbnail(thumbsize)
    output_filename = os.path.join(output_folder, get_filetitle(filename) + '_thumb.tiff')
    Image.fromarray(thumb).save(output_filename)
    #save_tiff(output_filename, thumb)
    return thumb


def convert_slide(filename, output_params):
    output_folder = output_params['folder']
    output_format = output_params['format']
    output_filename = os.path.join(output_folder, get_filetitle(filename, remove_all_ext=True) + '.' + output_format)
    slide = load_slide(filename)
    if 'zar' in output_format:
        convert_slide_to_zarr(slide, output_filename, output_params)
    elif 'ome' in output_format:
        convert_slide_to_tiff(slide, output_filename, output_params, ome=True)
    else:
        convert_slide_to_tiff(slide, output_filename, output_params)


def convert_slide_to_zarr0(input_filename, output_folder, patch_size=(256, 256)):
    slide = TiffSlide(input_filename)
    size = slide.sizes[0]
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
        thumb = np.asarray(slide.get_thumbnail((nx, ny)))
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

    # slide
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
                tile = slide.asarray(xs, ys, xs + w, ys + h)
                data[ys:ys+h, xs:xs+w] = tile


def convert_slide_to_zarr(slide, output_filename, output_params):
    size = slide.sizes_xyzct[0]
    shape = (size[0], size[1], size[2] * size[3])
    if slide.pixel_nbytes[0] == 2:
        dtype = 'uint16'
    else:
        dtype = 'uint8'
    tile_size = output_params['tile_size']
    compression = output_params.get('compression')

    zarr_root = zarr.open_group(output_filename, mode='w')
    zarr_data = zarr_root.create_dataset(str(0), shape=shape, chunks=(tile_size[0], tile_size[1], None), dtype=dtype,
                                         compressor=None, filters=compression)
    return zarr_data


def convert_slide_to_tiff(slide, output_filename, output_params, ome=False):
    image = slide.asarray
    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    pyramid_add = output_params.get('pyramid_add')
    pyramid_downsample = output_params.get('pyramid_downsample')
    if ome:
        metadata = None
        ome_metadata = slide.get_ome_metadata()
    else:
        metadata = slide.get_metadata()
        ome_metadata = None
    save_tiff(output_filename, image, metadata=metadata, xml_metadata=ome_metadata, tile_size=tile_size, compression=compression,
              pyramid_add=pyramid_add, pyramid_downsample=pyramid_downsample)


def save_tiff(filename, image, metadata=None, xml_metadata=None, tile_size=None, compression=None,
              pyramid_add=0, pyramid_downsample=4.0, pyramid_sizes_add=None):
    if xml_metadata is not None:
        xml_metadata_bytes = xml_metadata.encode()
    else:
        xml_metadata_bytes = None
    width, height = image.shape[1], image.shape[0]
    scale = 1
    with TiffWriter(filename, bigtiff=True) as writer:
        if pyramid_sizes_add is not None:
            pyramid_add = len(pyramid_sizes_add)
        writer.write(image, subifds=pyramid_add,
                     tile=tile_size, compression=compression, metadata=metadata, description=xml_metadata_bytes)

        for i in range(pyramid_add):
            if pyramid_sizes_add is not None:
                new_width, new_height = pyramid_sizes_add[i]
            else:
                scale /= pyramid_downsample
                new_width, new_height = int(round(width * scale)), int(round(height * scale))
            new_image = scale_image(image, (new_width, new_height))
            writer.write(new_image, subfiletype=1,
                         tile=tile_size, compression=compression)
