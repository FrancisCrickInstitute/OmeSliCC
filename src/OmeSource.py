import logging
import numpy as np

from src.image_util import image_resize_fast, image_resize, precise_resize
from src.ome import create_ome_metadata
from src.util import check_round_significants


class OmeSource:
    def __init__(self):
        self.metadata = {}
        self.ome_metadata = None
        self.mag0 = None
        self.target_mag = None
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.pixel_nbits = []
        self.pixel_size = []
        self.channel_info = []

    def init_metadata(self, filename, source_mag=None, source_mag_required=False):
        self.find_metadata()
        if self.mag0 == 0 and source_mag is not None:
            self.mag0 = source_mag
        if self.mag0 == 0:
            msg = f'{filename}: No source magnification in metadata or provided'
            if source_mag_required:
                raise ValueError(msg)
            else:
                logging.warning(msg)
        self.fix_pixelsize()
        self.set_mags()
        self.set_best_mag()

    def fix_pixelsize(self):
        standard_units = {'micro': 'µm', 'nano': 'nm'}
        pixel_size = []
        for pixel_size0 in self.pixel_size:
            pixel_size1 = check_round_significants(pixel_size0[0], 6)
            unit1 = pixel_size0[1]
            for standard_unit in standard_units:
                if unit1.lower().startswith(standard_unit):
                    unit1 = standard_units[standard_unit]
            pixel_size.append((pixel_size1, unit1))
        self.pixel_size = pixel_size

    def set_mags(self):
        self.source_mags = [self.mag0]
        for i, size in enumerate(self.sizes):
            if i > 0:
                mag = self.mag0 * np.mean(np.divide(size, self.sizes[0]))
                self.source_mags.append(check_round_significants(mag, 3))

    def set_best_mag(self):
        if self.target_mag is not None:
            source_mag, self.best_page = get_best_mag(self.source_mags, self.target_mag)
            self.best_factor = source_mag / self.target_mag
        else:
            self.best_page = 0
            self.best_factor = 1

    def get_max_mag(self):
        return np.max(self.source_mags)

    def get_actual_size(self):
        actual_size = []
        for size, pixel_size in zip(self.get_size_xyzct(), self.pixel_size):
            actual_size.append((np.multiply(size, pixel_size[0]), pixel_size[1]))
        return actual_size

    def get_pixel_type(self, level=0):
        return self.pixel_types[level]

    def get_pixel_nbits(self, level=0):
        return self.pixel_nbits[level]

    def get_pixel_nbytes(self, level=0):
        return self.pixel_nbits[level] // 8

    def get_channel_info(self):
        return self.channel_info

    def get_size_xyzct(self):
        xyzct = list(self.sizes_xyzct[0])
        n_same_size = len([size for size in self.sizes_xyzct[1:] if size == self.sizes_xyzct[0]]) + 1
        if n_same_size > 1:
            if xyzct[2] == 1:
                xyzct[2] = n_same_size
            else:
                xyzct[-1] = n_same_size
        return tuple(xyzct)

    def get_size(self):
        # size at selected magnification
        return np.divide(self.sizes[self.best_page], self.best_factor).astype(int)

    def get_shape(self):
        size = self.get_size()
        xyzct = self.sizes_xyzct[0]
        shape = (size[1], size[0], xyzct[2] * xyzct[3])
        return shape

    def clone_empty(self):
        return np.zeros(self.get_shape(), dtype=self.get_pixel_type())

    def get_thumbnail(self, target_size, precise=False):
        size, index = get_best_size(self.sizes, target_size)
        scale = np.divide(target_size, self.sizes[index])
        image = self.asarray_level(index, 0, 0, size[0], size[1])
        if np.round(scale, 3)[0] == 1 and np.round(scale, 3)[1] == 1:
            return image
        elif precise:
            return precise_resize(image, scale)
        else:
            return image_resize(image, target_size)

    def asarray(self, x0=0, y0=0, x1=-1, y1=-1):
        # ensure fixed patch size
        if x1 < 0 or y1 < 0:
            x1, y1 = self.get_size()
        # ensure fixed patch size
        w0 = x1 - x0
        h0 = y1 - y0
        factor = self.best_factor
        if factor != 1:
            np.multiply([x1, y1], factor)
            ox0, oy0 = np.round(np.multiply([x0, y0], factor)).astype(int)
            ox1, oy1 = np.round(np.multiply([x1, y1], factor)).astype(int)
        else:
            ox0, oy0, ox1, oy1 = x0, y0, x1, y1
        image0 = self.asarray_level(self.best_page, ox0, oy0, ox1, oy1)
        if factor != 1:
            w, h = np.round(np.divide(image0.shape[0:2], factor)).astype(int)
            image = image_resize_fast(image0, (w, h))
        else:
            image = image0
        w = image.shape[1]
        h = image.shape[0]
        if (h, w) != (h0, w0):
            image = np.pad(image, ((0, h0 - h), (0, w0 - w), (0, 0)), 'edge')
        return image

    def produce_chunks(self, chunk_size):
        w, h = self.get_size()
        ny = int(np.ceil(h / chunk_size[1]))
        nx = int(np.ceil(w / chunk_size[0]))
        for chunky in range(ny):
            for chunkx in range(nx):
                x0, y0 = chunkx * chunk_size[0], chunky * chunk_size[1]
                x1, y1 = min((chunkx + 1) * chunk_size[0], w), min((chunky + 1) * chunk_size[1], h)
                yield x0, y0, x1, y1, self.asarray(x0, y0, x1, y1)

    def get_metadata(self):
        return self.metadata

    def get_xml_metadata(self, output_filename, pyramid_sizes_add=None):
        return create_ome_metadata(self, output_filename, pyramid_sizes_add=pyramid_sizes_add).to_xml()

    def find_metadata(self):
        raise NotImplementedError('Implement method in subclass')

    def asarray_level(self, level, x0, y0, x1, y1):
        raise NotImplementedError('Implement method in subclass')


def get_best_mag(mags, target_mag):
    # find smallest mag larger/equal to target mag
    best_mag = None
    best_index = -1
    best_scale = 0
    for index, mag in enumerate(mags):
        scale = target_mag / mag
        if 1 >= scale > best_scale:
            best_index = index
            best_mag = mag
            best_scale = scale
    return best_mag, best_index


def get_best_size(sizes, target_size):
    # find largest scale but smaller to 1
    best_index = -1
    best_scale = 0
    for index, size in enumerate(sizes):
        scale = np.mean(np.divide(target_size, size))
        if 1 >= scale > best_scale:
            best_index = index
            best_scale = scale
    return sizes[best_index], best_index
