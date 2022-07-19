# * TODO: incorporate this experimental code into main pipeline
# * TODO: fix Zarr support, extend to Ome.Zarr
# * TODO: Add JPEGXR support for Zarr

import os
import numpy as np
import pandas as pd
import zarr
from numcodecs import register_codec
from numcodecs.blosc import Blosc
from tifffile import tifffile, TiffFile, TiffWriter
from tqdm import tqdm

from src.TiffSlide import TiffSlide
from src.image_util import JPEG2000, tags_to_dict, scale_image

register_codec(JPEG2000)


def convert_slides_to_zarr(csv_file, image_dir, patch_size):
    data = pd.read_csv(csv_file, delimiter='\t').to_dict()
    image_files = list(data['path'].values())
    nslides = len(image_files)

    for image_file in tqdm(image_files, total=nslides):
        filename = os.path.join(image_dir, image_file)
        if os.path.isfile(filename):
            convert_slide_to_zarr(filename, patch_size)


def convert_slide_to_zarr0(filename, patch_size):
    slide = TiffSlide(filename, 40)
    size = slide.sizes[0]
    width = size[0]
    height = size[1]
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)    # clevel=9

    zarr_filename = os.path.splitext(filename)[0] + '.zarr'
    root = zarr.open_group(zarr_filename, mode='a')

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


def convert_slide_to_zarr(image_filename, patch_size):
    slide = TiffSlide(image_filename, 40)
    size = slide.sizes[0]
    width = size[0]
    height = size[1]

    zarr_filename = os.path.splitext(image_filename)[0] + '.zarr'
    zarr_root = zarr.open_group(zarr_filename, mode='w')

    shape = (height, width, 3)
    #compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)    # similar: cname='zlib'
    zarr_data = zarr_root.create_dataset(str(0), shape=shape, chunks=(patch_size[0], patch_size[1], None), dtype='uint8',
                                         compressor=None, filters=[JPEG2000(50)])
    return zarr_data


def convert_slide_to_tiff(infilename, outfilename, ome=False, overwrite=False):
    if overwrite or not os.path.exists(outfilename):
        print(f'{infilename} -> {outfilename}')
        try:
            tiff = TiffFile(infilename)
            outpath = os.path.dirname(outfilename)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            with TiffWriter(outfilename, ome=ome, bigtiff=True) as writer:
                for page in tiff.pages:
                    if page.is_tiled:
                        tile_size = (page.tilelength, page.tilewidth)
                        if ome:
                            if page.is_ome:
                                metadata = tifffile.xml2dict(tiff.ome_metadata)
                            else:
                                metadata = tags_to_dict(page.tags)  # create pseudo OME metadata
                            description = None
                        else:
                            metadata = None
                            description = page.description
                        writer.write(page.asarray(), tile=tile_size, compression=['JPEG2000', 10],
                                     metadata=metadata, description=description)
        except Exception as e:
            print('file:', infilename, e)


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
        writer.write(image, photometric='RGB', subifds=pyramid_add,
                     tile=tile_size, compression=compression,
                     metadata=metadata, description=xml_metadata_bytes)

        for i in range(pyramid_add):
            if pyramid_sizes_add is not None:
                new_width, new_height = pyramid_sizes_add[i]
            else:
                scale /= pyramid_downsample
                new_width, new_height = int(round(width * scale)), int(round(height * scale))
            new_image = scale_image(image, (new_width, new_height))
            writer.write(new_image, photometric='RGB', subfiletype=1,
                         tile=tile_size, compression=compression)
