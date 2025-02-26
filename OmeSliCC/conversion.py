# https://pyquestions.com/how-to-save-a-very-large-numpy-array-as-an-image-loading-as-little-as-possible-into-memory


#import glob
import logging
import numpy as np
import os
from PIL import Image
import psutil
from tifffile import TiffWriter, TIFF, PHOTOMETRIC

from OmeSliCC.OmeSource import OmeSource, get_resolution_from_pixel_size
from OmeSliCC.OmeZarr import OmeZarr
from OmeSliCC.PlainImageSource import PlainImageSource
from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.Zarr import Zarr
from OmeSliCC.OmeZarrSource import OmeZarrSource
from OmeSliCC.image_util import *
from OmeSliCC.util import *


def create_source(source_ref: str, params: dict, omero=None) -> OmeSource:
    source_pixel_size = split_value_unit_list(params.get('input', {}).get('pixel_size'))
    target_pixel_size = split_value_unit_list(params.get('output', {}).get('pixel_size'))
    ext = os.path.splitext(source_ref)[1].lower()
    if omero is not None:
        from OmeSliCC.OmeroSource import OmeroSource
        source = OmeroSource(omero, int(source_ref), source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
    elif ext == '.zarr':
        source = OmeZarrSource(source_ref, source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
    elif ext.lstrip('.') in TIFF.FILE_EXTENSIONS:
        source = TiffSource(source_ref, source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
    elif ext in Image.registered_extensions().keys():
        source = PlainImageSource(source_ref, source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
    else:
        try:
            from OmeSliCC.BioSource import BioSource
            source = BioSource(source_ref, source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
        except ImportError:
            raise NotImplementedError('Unsupported: Bioformats not installed')
    return source


def get_image_info(source: OmeSource) -> str:
    sizes_xyzct = source.sizes_xyzct
    pixel_nbytes = source.get_pixel_nbytes()
    pixel_type = source.get_pixel_type()
    channels = source.get_channels()
    image_info = os.path.basename(source.source_reference)
    image_info += ' ' + get_image_size_info(sizes_xyzct, pixel_nbytes, pixel_type, channels)
    sizes = source.get_physical_size()
    if len(sizes) > 0:
        image_info += ' Physical size:'
        infos = []
        for size in sizes:
            if size[0] > 0:
                infos.append(f' {size[0]:.2f} {size[1]}')
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
            save_tiff(output_filename, thumb[..., channeli], dimension_order='yx')
    else:
        save_tiff(output_filename, thumb, dimension_order='yxc')


def convert_image(source: OmeSource, params: dict, load_chunked: bool = False):
    source_ref = source.source_reference
    output_params = params['output']
    output_folder = output_params['folder']
    output_format = output_params['format']
    ome = ('ome' in output_format)
    overwrite = output_params.get('overwrite', True)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filetitle = get_filetitle(source_ref).rstrip('.ome')
    output_filename = str(os.path.join(output_folder, filetitle + '.' + output_format))
    if overwrite or not os.path.exists(output_filename):
        if load_chunked:
            image = get_source_image_chunked(source)
            #image = get_source_image_dask(source)
        else:
            image = get_source_image(source)
        if 'ome.zarr' in output_format:
            save_image_as_ome_zarr(source, image, output_filename, output_params)
        elif 'zarr' in output_format:
            save_image_as_zarr(source, image, output_filename, output_params)
        elif 'tif' in output_format:
            save_image_as_tiff(source, image, output_filename, output_params, ome=ome)
        else:
            save_image(image, output_filename, output_params)
        check_image(source, image, output_filename)


def combine_images(sources: list[OmeSource], params: dict):
    source0 = sources[0]
    source_ref = source0.source_reference
    output_params = params['output']
    output_folder = output_params['folder']
    output_format = output_params['format']
    extra_metadata = output_params.get('extra_metadata', {})

    images = [get_source_image(source) for source in sources]
    image = da.concatenate(images, axis=1)

    # Experimental metadata
    #metadatas = []
    #for source in sources:
    #    image_filename = source.source_reference
    #    filepattern = os.path.splitext(image_filename)[0].rstrip('.ome') + '*'
    #    for metadata_filename in glob.glob(filepattern):
    #        if metadata_filename != image_filename:
    #            metadata = file_to_dict(metadata_filename)
    #            if metadata is not None:
    #                metadatas.append(metadata)

    new_source = OmeSource()
    ome = ('ome' in output_format)
    filetitle = get_filetitle(source_ref).rstrip('.ome')
    output_filename = str(os.path.join(output_folder, filetitle + '_combined.' + output_format))
    new_source.source_reference = output_filename
    new_source.target_pixel_size = source0.get_pixel_size()
    new_source.position = source0.get_position()
    new_source.rotation = source0.get_rotation()

    channels = extra_metadata.get('channels', [])
    if not channels:
        channels = []
        for source in sources:
            for channeli, channel in enumerate(source.get_channels()):
                label = channel.get('label', '')
                if label == '' or label in [standard_type.name.lower() for standard_type in TIFF.PHOTOMETRIC]:
                    channel = channel.copy()
                    label = get_filetitle(source.source_reference).rstrip('.ome')
                    if len(source.get_channels()) > 1:
                        label += f'#{channeli}'
                    channel['label'] = label
                channels.append(channel)
    nchannels = len(channels)

    if image.shape[1] != nchannels:
        logging.warning('#Combined image channels does not match #data channels')

    new_source.channels = channels

    new_source.sizes = [source0.get_size()]
    sizes_xyzc = list(source0.get_size_xyzct())
    sizes_xyzc[3] = nchannels
    new_source.sizes_xyzct = [tuple(sizes_xyzc)]
    new_source.pixel_types = source0.pixel_types
    new_source.pixel_nbits = source0.pixel_nbits
    new_source.best_level, new_source.best_factor, new_source.full_factor = 0, 1, 1
    new_source.source_mag = source0.source_mag
    new_source.output_dimension_order = source0.output_dimension_order

    if 'zar' in output_format:
        if 'ome.' in output_format:
            save_image_as_ome_zarr(new_source, image, output_filename, output_params)
        else:
            save_image_as_zarr(new_source, image, output_filename, output_params)
    elif 'tif' in output_format:
        save_image_as_tiff(new_source, image, output_filename, output_params, ome=ome)
    else:
        save_image(image, output_filename, output_params)


def store_tiles(sources: list[OmeSource], output_filename: str, params: dict,
                composition_metadata: list = [], image_operations: list = []):
    output_params = params['output']
    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    npyramid_add = output_params.get('npyramid_add', 0)
    pyramid_downsample = output_params.get('pyramid_downsample')

    translations = []
    pixel_size = sources[0].get_pixel_size_micrometer()
    for meta in composition_metadata:
        bounds = meta['Bounds']
        translation = bounds['StartX'], bounds['StartY']
        translation_um = np.multiply(translation, pixel_size[:2])
        translations.append(translation_um)

    #zarr = OmeZarr(output_filename)
    zarr = Zarr(output_filename)
    zarr.write(sources, tile_size=tile_size, compression=compression,
               npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample,
               translations=translations, image_operations=image_operations)


def get_source_image(source: OmeSource):
    image = source.asarray()
    image_size = image.size * image.itemsize
    if image_size < psutil.virtual_memory().total:
        image = np.asarray(image)   # pre-computing is way faster than dask saving/scaling
    return image


def get_source_image_dask(source: OmeSource, chunk_size=(10240, 10240)):
    image = source.asdask(chunk_size)
    return image


def get_source_image_chunked(source: OmeSource, chunk_size=(10240, 10240)):
    image = source.clone_empty()
    for indices, chunk in source.produce_chunks(chunk_size):
        s = indices
        e = np.array(s) + chunk.shape
        image[s[0]:e[0],
              s[1]:e[1],
              s[2]:e[2],
              s[3]:e[3],
              s[4]:e[4]] = chunk
    return image


def save_image_as_ome_zarr(source: OmeSource, data: np.ndarray, output_filename: str, output_params: dict):
    # ome-zarr: https://ngff.openmicroscopy.org/latest/
    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    npyramid_add = output_params.get('npyramid_add', 0)
    pyramid_downsample = output_params.get('pyramid_downsample')

    zarr = OmeZarr(output_filename)
    zarr.write(source, tile_size=tile_size, compression=compression,
               npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample)


def save_image_as_zarr(source: OmeSource, data: np.ndarray, output_filename: str, output_params: dict,
                       ome: bool = None, v3: bool = False):
    # ome-zarr: https://ngff.openmicroscopy.org/latest/
    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    npyramid_add = output_params.get('npyramid_add', 0)
    pyramid_downsample = output_params.get('pyramid_downsample')

    zarr = Zarr(output_filename, ome=ome, v3=v3)
    zarr.create(source, tile_size=tile_size, npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample,
                compression=compression)
    zarr.set(data)


def save_image_as_tiff(source: OmeSource, image: np.ndarray, output_filename: str, output_params: dict, ome: bool = False):
    tile_size = output_params.get('tile_size')
    compression = output_params.get('compression')
    combine_rgb = output_params.get('combine_rgb', True)

    npyramid_add = output_params.get('npyramid_add', 0)
    pyramid_downsample = output_params.get('pyramid_downsample')
    if npyramid_add > 0:
        pyramid_sizes_add = calc_pyramid(source.get_size_xyzct(), npyramid_add, pyramid_downsample)
    else:
        pyramid_sizes_add = []

    if ome:
        metadata = None
        xml_metadata = source.create_xml_metadata(output_filename, combine_rgb=combine_rgb,
                                                  pyramid_sizes_add=pyramid_sizes_add)
    else:
        metadata = source.get_metadata()
        xml_metadata = None
    resolution, resolution_unit = get_resolution_from_pixel_size(source.get_pixel_size())

    save_tiff(output_filename, image, metadata=metadata, xml_metadata=xml_metadata,
              dimension_order=source.get_dimension_order(),
              resolution=resolution, resolution_unit=resolution_unit, tile_size=tile_size,
              compression=compression, combine_rgb=combine_rgb, pyramid_sizes_add=pyramid_sizes_add)


def save_tiff(filename: str, image: np.ndarray, metadata: dict = None, xml_metadata: str = None,
              dimension_order: str = 'yxc',
              resolution: tuple = None, resolution_unit: str = None, tile_size: tuple = None, compression: [] = None,
              combine_rgb=True, pyramid_sizes_add: list = []):
    x_index = dimension_order.index('x')
    y_index = dimension_order.index('y')
    size = image.shape[x_index], image.shape[y_index]

    nchannels = 1
    if 'c' in dimension_order:
        c_index = dimension_order.index('c')
        nchannels = image.shape[c_index]
    else:
        c_index = -1

    if tile_size is not None and isinstance(tile_size, int):
        tile_size = [tile_size] * 2

    split_channels = not (combine_rgb and nchannels == 3)
    if nchannels == 3 and not split_channels:
        photometric = PHOTOMETRIC.RGB
        image = np.moveaxis(image, c_index, -1)
        dimension_order = dimension_order.replace('c', '') + 'c'
    else:
        photometric = PHOTOMETRIC.MINISBLACK

    if resolution is not None:
        # tifffile only supports x/y pyramid resolution
        resolution = tuple(resolution[0:2])

    # maximum size (w/o compression)
    max_size = image.size * image.itemsize
    base_size = np.divide(max_size, np.prod(size))
    for new_size in pyramid_sizes_add:
        max_size += np.prod(new_size) * base_size
    bigtiff = (max_size > 2 ** 32)

    #scaler = Scaler(downscale=..., max_layer=len(pyramid_sizes_add))   # use ome-zarr-py dask scaling

    if xml_metadata is not None:
        # set ome=False to provide custom OME xml in description
        xml_metadata_bytes = xml_metadata.encode()
        is_ome = False
    else:
        xml_metadata_bytes = None
        is_ome = None
    with TiffWriter(filename, ome=is_ome, bigtiff=bigtiff) as writer:
        writer.write(image, photometric=photometric, subifds=len(pyramid_sizes_add),
                     resolution=resolution, resolutionunit=resolution_unit, tile=tile_size, compression=compression,
                     metadata=metadata, description=xml_metadata_bytes)
        for new_size in pyramid_sizes_add:
            image = image_resize(image, new_size, dimension_order=dimension_order)
            #image = scaler.resize_image(image)    # significantly slower
            writer.write(image, photometric=photometric, subfiletype=1,
                         resolution=resolution, resolutionunit=resolution_unit, tile=tile_size, compression=compression)


def check_image(source, image, converted_filename):
    error_message = None
    try:
        dummy_params = {'input': {}, 'output': {}}
        converted_source = create_source(converted_filename, dummy_params)
        w, h = converted_source.get_size()
        x1, y1 = min(16, w), min(16, h)
        slicing = {'x0': 0, 'x1': x1, 'y0': 0, 'y1': y1, 'z': 0, 't': 0}
        slices = get_numpy_slicing(source.get_dimension_order(), **slicing)
        patch_original = image[slices]
        patch_converted = converted_source.asarray(**slicing)
        np.testing.assert_allclose(patch_original, patch_converted, verbose=False)
    except Exception as e:
        error_message = str(e)
    if error_message:
        raise ValueError(f'Converted image check\n{error_message}')
