import logging
import numpy as np

from OmeSliCC.color_conversion import *
from OmeSliCC.ome_metadata import create_ome_metadata
from OmeSliCC.image_util import *
from OmeSliCC.util import *


class OmeSource:
    """OME-compatible image source (base class)"""
    """Internal image format is [TCZYX]"""

    metadata: dict
    """metadata dictionary"""
    has_ome_metadata: bool
    """has ome metadata"""
    dimension_order: str
    """source dimension order"""
    output_dimension_order: str
    """data dimension order"""
    source_pixel_size: list
    """original source pixel size"""
    target_pixel_size: list
    """target pixel size"""
    sizes: list
    """x/y size pairs for all pages"""
    sizes_xyzct: list
    """xyzct size for all pages"""
    pixel_types: list
    """pixel types for all pages"""
    pixel_nbits: list
    """#bits for all pages"""
    channels: list
    """channel information for all image channels"""

    def __init__(self):
        self.metadata = {}
        self.has_ome_metadata = False
        self.dimension_order = ''
        self.output_dimension_order = ''
        self.source_pixel_size = []
        self.target_pixel_size = []
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.pixel_nbits = []
        self.channels = []

    def _init_metadata(self,
                       source_reference: str,
                       source_pixel_size: list = None,
                       target_pixel_size: list = None,
                       source_info_required: bool = False):

        self.source_reference = source_reference
        self.target_pixel_size = target_pixel_size
        self._find_metadata()
        if (len(self.source_pixel_size) == 0 or self.source_pixel_size[0][0] == 0
                or self.source_pixel_size[0][1] == '' or self.source_pixel_size[0][1] == 'inch') \
                and source_pixel_size is not None:
            # if pixel size is not set, or default/unspecified value
            self.source_pixel_size = source_pixel_size
        if len(self.source_pixel_size) == 0 or self.source_pixel_size[0][0] == 0:
            msg = f'{source_reference}: No source pixel size in metadata or provided'
            if source_info_required:
                raise ValueError(msg)
            else:
                logging.warning(msg)
        self._init_sizes()

    def _get_ome_metadate(self):
        images = ensure_list(self.metadata.get('Image', {}))[0]
        pixels = images.get('Pixels', {})
        self.source_pixel_size = []
        size = float(pixels.get('PhysicalSizeX', 0))
        if size > 0:
            self.source_pixel_size.append((size, pixels.get('PhysicalSizeXUnit', 'µm')))
        size = float(pixels.get('PhysicalSizeY', 0))
        if size > 0:
            self.source_pixel_size.append((size, pixels.get('PhysicalSizeYUnit', 'µm')))
        size = float(pixels.get('PhysicalSizeZ', 0))
        if size > 0:
            self.source_pixel_size.append((size, pixels.get('PhysicalSizeZUnit', 'µm')))

        self.source_mag = 0
        objective_id = images.get('ObjectiveSettings', {}).get('ID', '')
        for objective in ensure_list(self.metadata.get('Instrument', {}).get('Objective', [])):
            if objective.get('ID', '') == objective_id:
                self.source_mag = float(objective.get('NominalMagnification', 0))
        nchannels = self.get_nchannels()
        channels = []
        for channel0 in ensure_list(pixels.get('Channel', [])):
            channel = {'label': channel0.get('Name', '')}
            color = channel0.get('Color')
            if color:
                channel['color'] = int_to_rgba(int(color))
            channels.append(channel)
        if len(channels) == 0:
            if nchannels == 3:
                channels = [{'label': ''}]
            else:
                channels = [{'label': ''}] * nchannels
        self.channels = channels

    def _init_sizes(self):
        # normalise source pixel sizes
        standard_units = {'nano': 'nm', 'micro': 'µm', 'milli': 'mm', 'centi': 'cm'}
        pixel_size = []
        for pixel_size0 in self.source_pixel_size:
            pixel_size1 = check_round_significants(pixel_size0[0], 6)
            unit1 = pixel_size0[1]
            for standard_unit in standard_units:
                if unit1.lower().startswith(standard_unit):
                    unit1 = standard_units[standard_unit]
            pixel_size.append((pixel_size1, unit1))
        if 0 < len(pixel_size) < 2:
            pixel_size.append(pixel_size[0])
        self.source_pixel_size = pixel_size

        if self.target_pixel_size is None:
            self.target_pixel_size = self.source_pixel_size

        if 0 < len(self.target_pixel_size) < 2:
            self.target_pixel_size.append(self.target_pixel_size[0])

        # set source mags
        self.source_mags = [self.source_mag]
        for i, size in enumerate(self.sizes):
            if i > 0:
                mag = self.source_mag * np.mean(np.divide(size, self.sizes[0]))
                self.source_mags.append(check_round_significants(mag, 3))

        if self.target_pixel_size:
            self.best_level, self.best_factor, self.full_factor = get_best_level_factor(self, self.target_pixel_size)
        else:
            self.best_level = 0
            self.best_factor = 1

        if self.dimension_order == '':
            x, y, z, c, t = self.get_size_xyzct()
            if t > 1:
                self.dimension_order += 't'
            if c > 1:
                self.dimension_order += 'c'
            if z > 1:
                self.dimension_order += 'z'
            self.dimension_order += 'yx'

        self.output_dimension_order = 'tczyx'

    def get_source_dask(self):
        raise NotImplementedError('Implement method in subclass')

    def get_output_dask(self):
        data = self._get_output_dask()
        return redimension_data(data, self.dimension_order, self.get_dimension_order())

    def _get_output_dask(self):
        raise NotImplementedError('Implement method in subclass')

    def get_mag(self) -> float:
        mag = self.source_mag
        # get effective mag at target pixel size
        if self.target_pixel_size:
            mag *= np.mean(self.full_factor)
        return check_round_significants(mag, 3)

    def get_dimension_order(self) -> str:
        return self.output_dimension_order

    def get_physical_size(self) -> tuple:
        physical_size = []
        for size, pixel_size in zip(self.get_size_xyzct(), self.get_pixel_size()):
            physical_size.append((np.multiply(size, pixel_size[0]), pixel_size[1]))
        return tuple(physical_size)

    def get_pixel_type(self, level: int = 0) -> np.dtype:
        return self.pixel_types[level]

    def get_pixel_nbytes(self, level: int = 0) -> int:
        return self.pixel_nbits[level] // 8

    def get_channels(self) -> list:
        return self.channels

    def get_size(self) -> tuple:
        # size at target pixel size
        return np.round(np.multiply(self.sizes[self.best_level], self.best_factor)).astype(int)

    def get_size_xyzct(self) -> tuple:
        xyzct = list(self.sizes_xyzct[self.best_level]).copy()
        size = self.get_size()
        xyzct[0:2] = size
        return tuple(xyzct)

    def get_nchannels(self):
        return self.sizes_xyzct[0][3]

    def get_pixel_size(self) -> list:
        return self.target_pixel_size

    def get_pixel_size_micrometer(self):
        return get_value_units_micrometer(self.get_pixel_size())

    def get_shape(self, dimension_order: str = None, xyzct: tuple = None) -> tuple:
        shape = []
        if dimension_order is None:
            dimension_order = self.get_dimension_order()
        if xyzct is None:
            xyzct = self.get_size_xyzct()
        for dimension in dimension_order:
            index = 'xyzct'.index(dimension)
            shape.append(xyzct[index])
        return tuple(shape)

    def get_thumbnail(self, target_size: tuple, precise: bool = False) -> np.ndarray:
        size, index = get_best_size(self.sizes, target_size)
        scale = np.divide(target_size, self.sizes[index])
        image = self.get_yxc_image(self._asarray_level(index))
        if precise:
            thumbnail = precise_resize(image, scale)
        else:
            thumbnail = self.render(image_resize(image, target_size))
        return thumbnail

    def get_channel_window(self, channeli):
        min_quantile = 0.001
        max_quantile = 0.999

        if 'window' in self.channels[channeli]:
            return self.channels[channeli].get('window')

        dtype = self.get_pixel_type()
        if dtype.kind == 'f':
            info = np.finfo(dtype)
        else:
            info = np.iinfo(dtype)
        start, end = info.min, info.max

        nsizes = len(self.sizes)
        if nsizes > 1:
            image = self._asarray_level(nsizes - 1)
            image = image[:, channeli:channeli+1, ...]
            min, max = get_image_quantile(image, min_quantile), get_image_quantile(image, max_quantile)
        else:
            # do not query full size image
            min, max = start, end
        return {'start': start, 'end': end, 'min': min, 'max': max}

    def get_yxc_image(self, image, t=0, z=0, c=None):
        if z is not None:
            image = image[:, :, z, ...]
        if c is not None:
            image = image[:, c, ...]
        else:
            image = np.moveaxis(image, 1, -1)
        if t is not None:
            image = image[t, ...]
        return image

    def render(self, image: np.ndarray, t: int = 0, z: int = 0, channels: list = []) -> np.ndarray:
        if image.ndim == 5:
            image = self.get_yxc_image(image, t=t, z=z)
        new_image = np.zeros(list(image.shape[:2]) + [3], dtype=np.float32)
        tot_alpha = 0
        n = len(self.channels)

        is_rgb = (self.get_nchannels() == 3 and n == 1)
        do_normalisation = (image.dtype.itemsize == 2)

        if not is_rgb:
            for channeli, channel in enumerate(self.channels):
                if not channels or channeli in channels:
                    if n == 1:
                        channel_values = image
                    else:
                        channel_values = image[..., channeli]
                    if do_normalisation:
                        window = self.get_channel_window(channeli)
                        channel_values = normalise_values(channel_values, window['min'], window['max'])
                    else:
                        channel_values = int2float_image(channel_values)
                    color = channel.get('color')
                    if color:
                        rgba = color
                    else:
                        rgba = [1, 1, 1, 1]
                    color = rgba[:3]
                    alpha = rgba[3]
                    if alpha == 0:
                        alpha = 1
                    alpha_color = np.multiply(color, alpha).astype(np.float32)
                    new_image += np.atleast_3d(channel_values) * alpha_color
                    tot_alpha += alpha
            new_image = float2int_image(new_image / tot_alpha)
        elif do_normalisation:
            window = self.get_channel_window(0)
            new_image = float2int_image(normalise_values(image, window['min'], window['max']))
        else:
            new_image = image

        return new_image

    def asarray(self, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1,
                c: int = None, z: int = None, t: int = None,
                pixel_size: list = []) -> np.ndarray:
        # allow custom pixel size
        if pixel_size:
            level, factor, _ = get_best_level_factor(self, pixel_size)
            size0 = np.round(np.multiply(self.sizes[level], factor)).astype(int)
        else:
            level, factor = self.best_level, self.best_factor
            size0 = self.get_size()

        if x1 < 0 or y1 < 0:
            x1, y1 = size0
        if np.mean(factor) != 1:
            ox0, oy0 = np.round(np.divide([x0, y0], factor)).astype(int)
            ox1, oy1 = np.round(np.divide([x1, y1], factor)).astype(int)
        else:
            ox0, oy0, ox1, oy1 = x0, y0, x1, y1
        image0 = self._asarray_level(level, ox0, oy0, ox1, oy1, c, z, t)
        if np.mean(factor) != 1:
            image = image_resize(image0, size0, dimension_order=self.get_dimension_order())
        else:
            image = image0
        return image

    def clone_empty(self) -> np.ndarray:
        return np.zeros(self.get_shape(), dtype=self.get_pixel_type())

    def produce_chunks(self, chunk_size: tuple) -> tuple:
        w, h = self.get_size()
        ny = int(np.ceil(h / chunk_size[1]))
        nx = int(np.ceil(w / chunk_size[0]))
        for chunky in range(ny):
            for chunkx in range(nx):
                x0, y0 = chunkx * chunk_size[0], chunky * chunk_size[1]
                x1, y1 = min((chunkx + 1) * chunk_size[0], w), min((chunky + 1) * chunk_size[1], h)
                indices = 0, 0, 0, y0, x0
                yield indices, self.asarray(x0, y0, x1, y1)

    def get_metadata(self) -> dict:
        return self.metadata

    def create_xml_metadata(self, output_filename: str, combine_rgb: bool = True, pyramid_sizes_add: list = None) -> str:
        return create_ome_metadata(self, output_filename, combine_rgb=combine_rgb, pyramid_sizes_add=pyramid_sizes_add)

    def _find_metadata(self):
        raise NotImplementedError('Implement method in subclass')

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1,
                       c: int = None, z: int = None, t: int = None) -> np.ndarray:
        raise NotImplementedError('Implement method in subclass')

    def close(self):
        pass


def get_resolution_from_pixel_size(pixel_size: list) -> tuple:
    conversions = {
        'cm': (1, 'centimeter'),
        'mm': (1, 'millimeter'),
        'µm': (1, 'micrometer'),
        'um': (1, 'micrometer'),
        'nm': (1000, 'micrometer'),
        'nanometer': (1000, 'micrometer'),
    }
    resolutions = []
    resolutions_unit = None
    if len(pixel_size) > 0:
        units = []
        for size, unit in pixel_size:
            if size != 0 and size != 1:
                resolution = 1 / size
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


def get_best_level_factor(source: OmeSource, target_pixel_size: list) -> tuple:
    # find best pixel_size level and corresponding factor
    if source.source_pixel_size:
        target_factor = np.divide(get_value_units_micrometer(source.source_pixel_size)[:2],
                                  get_value_units_micrometer(target_pixel_size)[:2])
    else:
        target_factor = 1
    best_level = 0
    best_factor = None
    for level, size in enumerate(source.sizes):
        factor = np.divide(size, source.sizes[0])
        if (np.all(factor > target_factor)
                or np.all(np.isclose(factor, target_factor, rtol=1e-4))
                or best_factor is None):
            best_level = level
            best_factor = factor
    return best_level, target_factor / best_factor, target_factor


def get_best_size(sizes: list, target_size: tuple) -> tuple:
    # find largest scale but smaller to 1
    best_index = 0
    best_scale = 0
    for index, size in enumerate(sizes):
        scale = np.mean(np.divide(target_size, size))
        if 1 >= scale > best_scale:
            best_index = index
            best_scale = scale
    return sizes[best_index], best_index
