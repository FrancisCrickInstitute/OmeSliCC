import logging
import numpy as np

from src.image_util import image_resize_fast, image_resize, precise_resize
from src.ome import create_ome_metadata
from src.util import check_round_significants, ensure_list


class OmeSource:
    """OME-compatible image source (base class)"""

    metadata: dict
    """metadata dictionary"""
    mag0: float
    """original source magnification"""
    target_mag: float
    """target magnification"""
    sizes: list
    """x/y size pairs for all pages"""
    sizes_xyzct: list
    """xyzct size for all pages"""
    pixel_types: list
    """pixel types for all pages"""
    pixel_nbits: list
    """#bits for all pages"""
    pixel_size: list
    """pixel sizes for all pages"""
    channel_info: list
    """channel information for all channels"""
    # TODO: make channel_info a (list of) dict

    def __init__(self):
        self.metadata = {}
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.pixel_nbits = []
        self.pixel_size = []
        self.channel_info = []

    def _init_metadata(self, source_reference: str, source_mag: float = None, source_mag_required: bool = False):
        self.source_reference = source_reference
        self._find_metadata()
        if self.mag0 == 0 and source_mag is not None:
            self.mag0 = source_mag
        if self.mag0 == 0:
            msg = f'{source_reference}: No source magnification in metadata or provided'
            if source_mag_required:
                raise ValueError(msg)
            else:
                logging.warning(msg)
        self._fix_pixelsize()
        self._set_mags()
        self._set_best_mag()

    def _get_ome_metadate(self):
        # TODO: use objective settings to get matching mag instead
        images = ensure_list(self.metadata.get('Image', {}))[0]
        pixels = images.get('Pixels', {})
        self.pixel_size = [(float(pixels.get('@PhysicalSizeX', 0)), pixels.get('@PhysicalSizeXUnit', 'µm')),
                           (float(pixels.get('@PhysicalSizeY', 0)), pixels.get('@PhysicalSizeYUnit', 'µm')),
                           (float(pixels.get('@PhysicalSizeZ', 0)), pixels.get('@PhysicalSizeZUnit', 'µm'))]
        for channel in ensure_list(pixels.get('Channel', {})):
            self.channel_info.append((channel.get('@Name', ''), int(channel.get('@SamplesPerPixel', 1))))
        self.mag0 = float(self.metadata.get('Instrument', {}).get('Objective', {}).get('@NominalMagnification', 0))

    def _fix_pixelsize(self):
        standard_units = {'nano': 'nm', 'micro': 'µm', 'milli': 'mm', 'centi': 'cm'}
        pixel_size = []
        for pixel_size0 in self.pixel_size:
            pixel_size1 = check_round_significants(pixel_size0[0], 6)
            unit1 = pixel_size0[1]
            for standard_unit in standard_units:
                if unit1.lower().startswith(standard_unit):
                    unit1 = standard_units[standard_unit]
            pixel_size.append((pixel_size1, unit1))
        self.pixel_size = pixel_size

    def _set_mags(self):
        self.source_mags = [self.mag0]
        for i, size in enumerate(self.sizes):
            if i > 0:
                mag = self.mag0 * np.mean(np.divide(size, self.sizes[0]))
                self.source_mags.append(check_round_significants(mag, 3))

    def _set_best_mag(self):
        if self.mag0 is not None and self.mag0 > 0 and self.target_mag is not None and self.target_mag > 0:
            source_mag, self.best_level = get_best_mag(self.source_mags, self.target_mag)
            self.best_factor = source_mag / self.target_mag
        else:
            self.best_level = 0
            self.best_factor = 1

    def get_mag(self) -> float:
        if self.target_mag is not None:
            return self.target_mag
        else:
            return self.source_mags[0]

    def get_max_mag(self) -> float:
        return np.max(self.source_mags)

    def get_actual_size(self) -> list:
        actual_size = []
        for size, pixel_size in zip(self.get_size_xyzct(), self.pixel_size):
            actual_size.append((np.multiply(size, pixel_size[0]), pixel_size[1]))
        return actual_size

    def get_pixel_type(self, level: int = 0) -> np.dtype:
        return self.pixel_types[level]

    def get_pixel_nbits(self, level: int = 0) -> int:
        return self.pixel_nbits[level]

    def get_pixel_nbytes(self, level: int = 0) -> int:
        return self.pixel_nbits[level] // 8

    def get_channel_info(self) -> list:
        return self.channel_info

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

    def get_size(self) -> tuple:
        # size at selected magnification
        return np.divide(self.sizes[self.best_level], self.best_factor).astype(int)

    def get_nchannels(self):
        return self.sizes_xyzct[0][3]

    def get_pixelsize(self):
        return self.pixel_size

    def get_pixelsize_micrometer(self):
        conversion = {'nm': 1e-3, 'µm': 1, 'mm': 1e3, 'cm': 1e4}
        return [pixelsize[0] * conversion.get(pixelsize[1], 1) for pixelsize in self.get_pixelsize()]

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
        if factor != 1:
            ox0, oy0 = np.round(np.multiply([x0, y0], factor)).astype(int)
            ox1, oy1 = np.round(np.multiply([x1, y1], factor)).astype(int)
        else:
            ox0, oy0, ox1, oy1 = x0, y0, x1, y1
        image0 = self._asarray_level(self.best_level, ox0, oy0, ox1, oy1)
        if factor != 1:
            h, w = np.round(np.divide(image0.shape[0:2], factor)).astype(int)
            image = image_resize_fast(image0, (w, h))
        else:
            image = image0
        w = image.shape[1]
        h = image.shape[0]
        if (h, w) != (h0, w0):
            image = np.pad(image, ((0, h0 - h), (0, w0 - w), (0, 0)), 'edge')
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

    def create_xml_metadata(self, output_filename: str, channel_output: str = '', pyramid_sizes_add: list = None) -> str:
        return create_ome_metadata(self, output_filename, channel_output=channel_output, pyramid_sizes_add=pyramid_sizes_add)

    def _find_metadata(self):
        raise NotImplementedError('Implement method in subclass')

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        raise NotImplementedError('Implement method in subclass')

    def close(self):
        pass


def get_best_mag(mags: list, target_mag: float) -> tuple:
    # find smallest mag larger/equal to target mag
    best_mag = None
    best_index = 0
    best_scale = 0
    for index, mag in enumerate(mags):
        if mag > 0:
            scale = target_mag / mag
            if 1 >= scale > best_scale or best_scale == 0:
                best_index = index
                best_mag = mag
                best_scale = scale
    return best_mag, best_index


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
