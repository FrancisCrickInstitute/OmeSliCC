# https://pyquestions.com/how-to-save-a-very-large-numpy-array-as-an-image-loading-as-little-as-possible-into-memory
# * TODO: fix Zarr support, extend to Ome.Zarr
# * TODO: Add JPEGXR support for Zarr
# TODO: use ome-zarr python package for reading/writing if supporting image compressors (Jpeg2k/xr)

import os
import numpy as np
import zarr
from PIL import Image
from imagecodecs.numcodecs import Lzw, Jpeg2k, JpegXr, JpegXl
from tifffile import TIFF, TiffWriter

from src import Omero
from src.BioSource import BioSource
from src.OmeSource import OmeSource, get_resolution_from_pixel_size
from src.OmeroSource import OmeroSource
from src.PlainImageSource import PlainImageSource
from src.TiffSource import TiffSource
from src.ZarrSource import ZarrSource
from src.image_util import image_resize, get_image_size_info, calc_pyramid, ensure_unsigned_image, reverse_last_axis, \
    save_image
from src.util import get_filetitle, split_value_unit_list, ensure_list


def create_source(source_ref: str, params: dict, omero: Omero = None) -> OmeSource:
    source_pixel_size = split_value_unit_list(params['input'].get('pixel_size'))
    target_pixel_size = split_value_unit_list(params['output'].get('pixel_size'))
    ext = os.path.splitext(source_ref)[1].lower()
    if omero is not None:
        source = OmeroSource(omero, int(source_ref), source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
    elif 'zarr' in ext:
        source = ZarrSource(source_ref, source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
    elif ext.lstrip('.') in TIFF.FILE_EXTENSIONS:
        source = TiffSource(source_ref, source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
    elif ext in Image.registered_extensions().keys():
        source = PlainImageSource(source_ref, source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
    else:
        source = BioSource(source_ref, source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
    return source


def get_image_info(source: OmeSource) -> str:
    xyzct = source.get_size_xyzct()
    pixel_nbytes = source.get_pixel_nbytes()
    pixel_type = source.get_pixel_type()
    channel_info = source.get_channel_info()
    image_info = os.path.basename(source.source_reference) + '\n'
    image_info += get_image_size_info(xyzct, pixel_nbytes, pixel_type, channel_info)
    sizes = source.get_physical_size()
    if len(sizes) > 0:
        image_info += '\nPhysical size:'
        infos = []
        for size in sizes:
            infos.append(f' {size[0]:.3f} {size[1]}')
        image_info += ' x'.join(infos)
    return image_info


def extract_thumbnail(source: OmeSource, params: dict):
    source_ref = source.source_reference
    output_params = params['output']
    output_folder = output_params['folder']
    target_size = output_params.get('thumbnail_size', 1000)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = os.path.join(output_folder, f'{get_filetitle(source_ref)}_thumb.tiff')
    size = source.get_size()
    nchannels = source.get_size_xyzct()[3]

    if target_size < 1:
        factor = target_size
    else:
        factor = np.max(np.divide(size, target_size))
    thumb_size = np.round(np.divide(size, factor)).astype(int)
    thumb = source.get_thumbnail(thumb_size)

    if nchannels not in [1, 3]:
        for channeli in range(nchannels):
            output_filename = os.path.join(output_folder, f'{get_filetitle(source_ref)}_channel{channeli}_thumb.tiff')
            save_tiff(output_filename, thumb[..., channeli])
    else:
        save_tiff(output_filename, thumb)


def convert_image(source: OmeSource, params: dict):
    source_ref = source.source_reference
    output_params = params['output']
    output_folder = output_params['folder']
    output_format = output_params['format']
    ome = ('ome' in output_format)
    overwrite = output_params.get('overwrite', True)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, get_filetitle(source_ref, remove_all_ext=True) + '.' + output_format)
    if overwrite or not os.path.exists(output_filename):
        image = get_source_image(source)
        if 'zar' in output_format:
            save_image_as_zarr(source, image, output_filename, output_params, ome=ome)
        elif 'tif' in output_format:
            save_image_as_tiff(source, image, output_filename, output_params, ome=ome)
        else:
            save_image(image, output_filename, output_params)


def get_source_image(source: OmeSource, chunk_size=(10240, 10240)):
    image = source.clone_empty()
    for x0, y0, x1, y1, chunk in source.produce_chunks(chunk_size):
        image[y0:y1, x0:x1] = chunk
    return image


def save_image_as_zarr(source: OmeSource, image: np.ndarray, output_filename: str, output_params: dict, ome: bool = False):
    # ome-zarr: https://ngff.openmicroscopy.org/latest/
    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    npyramid_add = output_params.get('npyramid_add', 0)
    pyramid_downsample = output_params.get('pyramid_downsample')
    size_xyzct = source.get_size_xyzct()

    compression = ensure_list(compression)
    compression_type = compression[0].lower()
    if len(compression) > 1:
        level = int(compression[1])
    else:
        level = None
    compressor = None
    compression_filters = None
    if 'lzw' in compression_type:
        compression_filters = [Lzw()]
    elif '2k' in compression_type or '2000' in compression_type:
        compression_filters = [Jpeg2k(level)]
    elif 'jpegxr' in compression_type:
        compression_filters = [JpegXr(level)]
    elif 'jpegxl' in compression_type:
        compression_filters = [JpegXl(level)]
    else:
        compressor = compression

    zarr_root = zarr.open_group(output_filename, mode='w')
    size = source.get_size()
    pixel_size = source.get_pixel_size()
    scale = 1
    datasets = []
    for pathi in range(1 + npyramid_add):
        if scale != 1:
            new_size = np.multiply(size, scale)
            image = image_resize(image, new_size)
        out = zarr_root.create_dataset(str(pathi), shape=image.shape, chunks=tile_size,
                                       dtype=image.dtype, compressor=compressor, filters=compression_filters)
        out[...] = image
        pixel_size_x = pixel_size[0][0]
        pixel_size_y = pixel_size[1][0]
        pixel_size_z = pixel_size[2][0]
        datasets.append({
            'path': pathi,
            'coordinateTransformations': [{'type': 'scale', 'scale': [1, 1, pixel_size_z, pixel_size_y / scale, pixel_size_x / scale]}]
        })
        scale /= pyramid_downsample

    if ome:
        # metadata
        axes = []
        dimension_order = 'yx'
        _, _, z, c, t = size_xyzct
        if c > 1:
            dimension_order += 'c'
        if z > 1:
            dimension_order += 'z'
        if t > 1:
            dimension_order += 't'
        for dimension in dimension_order:
            unit1 = None
            if dimension == 't':
                type1 = 'time'
                unit1 = 'millisecond'
            elif dimension == 'c':
                type1 = 'channel'
            else:
                type1 = 'space'
                unit1 = pixel_size['xyz'.index(dimension)][1]
            axis = {'name': dimension, 'type': type1}
            if unit1 is not None and unit1 != '':
                axis['unit'] = unit1
            axes.append(axis)

        zarr_root.attrs['multiscales'] = [{
            'version': '0.4',
            'axes': axes,
            'name': get_filetitle(source.source_reference),
            'datasets': datasets
        }]


def save_image_as_tiff(source: OmeSource, image: np.ndarray, output_filename: str, output_params: dict, ome: bool = False):
    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    combine_rgb = output_params.get('combine_rgb', True)
    channel_output = 'combine_rgb' if combine_rgb else ''

    npyramid_add = output_params.get('npyramid_add', 0)
    pyramid_downsample = output_params.get('pyramid_downsample')
    if npyramid_add > 0:
        pyramid_sizes_add = calc_pyramid(source.get_size_xyzct(), npyramid_add, pyramid_downsample)
    else:
        pyramid_sizes_add = None

    if ome:
        metadata = None
        xml_metadata = source.create_xml_metadata(output_filename, channel_output=channel_output,
                                                  pyramid_sizes_add=pyramid_sizes_add)
        #print(xml_metadata)
    else:
        metadata = source.get_metadata()
        xml_metadata = None
    resolution, resolution_unit = get_resolution_from_pixel_size(source.get_pixel_size())

    save_tiff(output_filename, image, metadata=metadata, xml_metadata=xml_metadata,
              resolution=resolution, resolution_unit=resolution_unit, tile_size=tile_size,
              compression=compression, combine_rgb=combine_rgb, pyramid_sizes_add=pyramid_sizes_add)


def save_tiff(filename: str, image: np.ndarray, metadata: dict = None, xml_metadata: str = None,
              resolution: tuple = None, resolution_unit: str = None, tile_size: tuple = None, compression: [] = None,
              combine_rgb=True, npyramid_add: int = 0, pyramid_downsample: float = 4.0, pyramid_sizes_add: list = None):
    # Use tiled writing (less memory needed but maybe slower):
    # writer.write(tile_iterator, shape=shape_size_at_desired_mag_pyramid_scale, tile=tile_size)

    image = ensure_unsigned_image(image)

    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    volumetric = (nchannels > 4)
    split_channels = not (combine_rgb and nchannels == 3)
    photometric = 'minisblack' if split_channels or nchannels != 3 else None
    if volumetric:
        size = image.shape[1], image.shape[0], image.shape[2]
    else:
        size = image.shape[1], image.shape[0]
    if resolution is not None:
        # tifffile only support x/y resolution
        resolution = tuple(resolution[0:2])
    if pyramid_sizes_add is not None:
        npyramid_add = len(pyramid_sizes_add)
    scale = 1
    xml_metadata_bytes = xml_metadata.encode() if xml_metadata is not None else None
    bigtiff = (image.size * image.itemsize > 2 ** 32)       # estimate size (w/o compression or pyramid)
    with TiffWriter(filename, ome=False, bigtiff=bigtiff) as writer:    # set ome=False to provide custom OME xml in description
        writer.write(reverse_last_axis(image, reverse=split_channels), photometric=photometric, subifds=npyramid_add,
                     resolution=resolution, resolutionunit=resolution_unit, tile=tile_size, compression=compression,
                     metadata=metadata, description=xml_metadata_bytes)

        for i in range(npyramid_add):
            if pyramid_sizes_add is not None:
                new_size = pyramid_sizes_add[i]
            else:
                scale /= pyramid_downsample
                if resolution is not None:
                    resolution = tuple(np.divide(resolution, pyramid_downsample))
                new_size = np.multiply(size, scale)
            image = image_resize(image, new_size)
            writer.write(reverse_last_axis(image, reverse=split_channels), photometric=photometric, subfiletype=1,
                         resolution=resolution, resolutionunit=resolution_unit, tile=tile_size, compression=compression)
