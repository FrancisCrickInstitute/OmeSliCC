import logging
import numpy as np

from OmeSliCC.XmlDict import XmlDict
from OmeSliCC.ome import create_ome_metadata
from OmeSliCC.image_util import *
from OmeSliCC.util import *


class OmeSource:
    """OME-compatible image source (base class)"""

    metadata: dict
    """metadata dictionary"""
    has_ome_metadata: bool
    """has ome metadata"""
    source_pixel_size: list
    """original source pixel size"""
    target_pixel_size: list
    """target pixel size"""
    target_scale: list
    """target (pixel size) scale"""
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
        self.source_pixel_size = []
        self.target_pixel_size = []
        self.target_scale = []
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
        if (len(self.source_pixel_size) == 0
                or self.source_pixel_size[0][0] == 0) \
                and source_pixel_size is not None:
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
        self.source_pixel_size = [(float(pixels.get('PhysicalSizeX', 0)), pixels.get('PhysicalSizeXUnit', 'µm')),
                                  (float(pixels.get('PhysicalSizeY', 0)), pixels.get('PhysicalSizeYUnit', 'µm')),
                                  (float(pixels.get('PhysicalSizeZ', 0)), pixels.get('PhysicalSizeZUnit', 'µm'))]
        self.source_mag = 0
        objective_id = images.get('ObjectiveSettings', {}).get('ID', '')
        for objective in ensure_list(self.metadata.get('Instrument', {}).get('Objective', [])):
            if objective.get('ID', '') == objective_id:
                self.source_mag = float(objective.get('NominalMagnification', 0))
        nchannels = self.sizes_xyzct[0][3]
        channels = []
        for channel0 in ensure_list(pixels.get('Channel', [])):
            channel = channel0.copy()
            channel['@SamplesPerPixel'] = int(channel['SamplesPerPixel'])
            channels.append(channel)
        if len(channels) == 0:
            if nchannels == 3:
                channels = [XmlDict({'@Name': '', '@SamplesPerPixel': nchannels})]
            else:
                channels = [XmlDict({'@Name': '', '@SamplesPerPixel': 1})] * nchannels
        self.channels = channels

    def _init_sizes(self):
        self.scales = [np.mean(np.divide(self.sizes[0], size)) for size in self.sizes]

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

        target_scale = []
        for source_pixel_size1, target_pixel_size1 in \
                zip(get_value_units_micrometer(self.source_pixel_size), get_value_units_micrometer(self.target_pixel_size)):
            if source_pixel_size1 != 0:
                target_scale.append(np.divide(target_pixel_size1, source_pixel_size1))
        self.target_scale = target_scale

        if len(target_scale) > 0:
            best_scale, self.best_level = get_best_scale(self.scales, float(np.mean(target_scale)))
            self.best_factor = np.divide(target_scale, best_scale)
        else:
            self.best_level = 0
            self.best_factor = [1]

    def get_mag(self) -> float:
        # get effective mag at target pixel size
        if len(self.target_scale) > 0:
            return check_round_significants(self.source_mag / np.mean(self.target_scale), 3)
        else:
            return self.source_mag

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
        return np.divide(self.sizes[self.best_level], self.best_factor[0:2]).astype(int)

    def get_size_xyzct(self) -> tuple:
        xyzct = list(self.sizes_xyzct[0])
        n_same_size = len([size for size in self.sizes_xyzct[1:] if list(size) == xyzct]) + 1
        if n_same_size > 1:
            if xyzct[2] == 1:
                xyzct[2] = n_same_size
            else:
                xyzct[-1] = n_same_size
        size = self.get_size()
        xyzct[0:2] = size
        return tuple(xyzct)

    def get_nchannels(self):
        return self.sizes_xyzct[0][3]

    def get_pixel_size(self) -> list:
        return self.target_pixel_size

    def get_pixel_size_micrometer(self):
        return get_value_units_micrometer(self.get_pixel_size())

    def get_shape(self) -> tuple:
        xyzct = self.get_size_xyzct()
        n = xyzct[2] * xyzct[3]
        if n > 1:
            shape = (xyzct[1], xyzct[0], n)
        else:
            shape = (xyzct[1], xyzct[0])
        return shape

    def clone_empty(self) -> np.ndarray:
        return np.zeros(self.get_shape(), dtype=self.get_pixel_type())

    def get_thumbnail(self, target_size: tuple, precise: bool = False) -> np.ndarray:
        size, index = get_best_size(self.sizes, target_size)
        scale = np.divide(target_size, self.sizes[index])
        image = self._asarray_level(index, 0, 0, size[0], size[1])
        if np.round(scale, 3)[0] == 1 and np.round(scale, 3)[1] == 1:
            return image
        elif precise:
            return precise_resize(image, scale)
        else:
            return image_resize(image, target_size)

    def asarray(self, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        # ensure fixed patch size
        if x1 < 0 or y1 < 0:
            x1, y1 = self.get_size()
        # ensure fixed patch size
        w0 = x1 - x0
        h0 = y1 - y0
        factor = self.best_factor
        if np.mean(factor) != 1:
            ox0, oy0 = np.round(np.multiply([x0, y0], factor)).astype(int)
            ox1, oy1 = np.round(np.multiply([x1, y1], factor)).astype(int)
        else:
            ox0, oy0, ox1, oy1 = x0, y0, x1, y1
        image0 = self._asarray_level(self.best_level, ox0, oy0, ox1, oy1)
        if np.mean(factor) != 1:
            h, w = np.round(np.divide(image0.shape[0:2], factor)).astype(int)
            image = image_resize_fast(image0, (w, h))
        else:
            image = image0
        if image.shape[0:2] != (h0, w0):
            image = image_reshape(image, (w0, h0))
        return image

    def produce_chunks(self, chunk_size: tuple) -> tuple:
        w, h = self.get_size()
        ny = int(np.ceil(h / chunk_size[1]))
        nx = int(np.ceil(w / chunk_size[0]))
        for chunky in range(ny):
            for chunkx in range(nx):
                x0, y0 = chunkx * chunk_size[0], chunky * chunk_size[1]
                x1, y1 = min((chunkx + 1) * chunk_size[0], w), min((chunky + 1) * chunk_size[1], h)
                yield x0, y0, x1, y1, self.asarray(x0, y0, x1, y1)

    def get_metadata(self) -> dict:
        return self.metadata

    def create_xml_metadata(self, output_filename: str, combine_rgb: bool = True, pyramid_sizes_add: list = None) -> str:
        return create_ome_metadata(self, output_filename, combine_rgb=combine_rgb, pyramid_sizes_add=pyramid_sizes_add)

    def _find_metadata(self):
        raise NotImplementedError('Implement method in subclass')

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        raise NotImplementedError('Implement method in subclass')

    def close(self):
        pass


def get_resolution_from_pixel_size(pixel_size: list) -> tuple:
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


def get_best_scale(scales: list, target_scale: float) -> tuple:
    # find smallest scale larger/equal to target scale
    best_factor = None
    best_scale = 1
    best_level = 0
    for level, scale in enumerate(scales):
        factor = target_scale / scale
        if best_factor is None or 1 <= factor < best_factor:
            best_factor = factor
            best_scale = scale
            best_level = level
    return best_scale, best_level


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
