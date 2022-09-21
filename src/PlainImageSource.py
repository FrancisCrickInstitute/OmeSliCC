import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from src.OmeSource import OmeSource
from src.image_util import pilmode_to_pixelinfo, get_pil_metadata
from src.ome import create_ome_metadata

Image.MAX_IMAGE_PIXELS = None   # avoid DecompressionBombError (which prevents loading large images)


class PlainImageSource(OmeSource):
    def __init__(self, filename, source_mag=None, target_mag=None, source_mag_required=False, executor=None):
        self.filename = filename
        self.mag0 = source_mag
        self.target_mag = target_mag
        self.loaded = False
        self.data = None
        self.arrays = []

        if executor is not None:
            self.executor = executor
        else:
            max_workers = (os.cpu_count() or 1) + 4
            self.executor = ThreadPoolExecutor(max_workers)

        self.image = Image.open(filename)
        self.metadata = get_pil_metadata(self.image)
        size = (self.image.width, self.image.height)
        self.sizes = [size]
        size_xyzct = (self.image.width, self.image.height, self.image.n_frames, len(self.image.getbands()), 1)
        self.sizes_xyzct = [size_xyzct]
        pixelinfo = pilmode_to_pixelinfo(self.image.mode)
        self.pixel_types = [pixelinfo[0]]
        self.pixel_nbits = [pixelinfo[1]]
        self.init_res_mag(filename, source_mag_required=source_mag_required)

    def get_metadata(self):
        return self.metadata

    def get_xml_metadata(self, output_filename):
        size = self.get_size()
        xyzct = self.sizes_xyzct[0]
        physical_size = size / self.metadata['dpi']
        physical_size_z = 1
        image_info = {'size_x': size[0], 'size_y': size[1], 'size_z': xyzct[2], 'size_c': xyzct[3], 'size_t': xyzct[4],
                      'physical_size_x': physical_size[0], 'physical_size_y': physical_size[1], 'physical_size_z': physical_size_z,
                      'dimension_order': 'XYZCT', 'type': self.pixel_types[0].__name__}
        ome_metadata = create_ome_metadata(output_filename, image_info, [])
        return ome_metadata.to_xml()

    def load(self):
        self.unload()
        self.arrays.append(np.array(self.image))
        self.loaded = True

    def unload(self):
        for array in self.arrays:
            del array
        self.arrays = []
        self.loaded = False

    def asarray_level(self, level, x0, y0, x1, y1):
        if self.loaded:
            array = self.arrays[level]
        else:
            array = np.array(self.image)
        return array[y0:y1, x0:x1]
