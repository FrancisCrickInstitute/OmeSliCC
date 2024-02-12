import numpy as np
from PIL import Image
from tifffile import RESUNIT

from OmeSliCC.OmeSource import OmeSource
from OmeSliCC.image_util import *

Image.MAX_IMAGE_PIXELS = None   # avoid DecompressionBombError (which prevents loading large images)


class PlainImageSource(OmeSource):
    """Plain common format image source"""

    filename: str
    """original filename"""
    loaded: bool
    """if image data is loaded"""
    arrays: list
    """list of all image arrays for different sizes"""

    def __init__(self,
                 filename: str,
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False):

        super().__init__()
        self.loaded = False
        self.arrays = []

        self.image = Image.open(filename)
        self.metadata = get_pil_metadata(self.image)
        size = (self.image.width, self.image.height)
        self.sizes = [size]
        nchannels = len(self.image.getbands())
        size_xyzct = (self.image.width, self.image.height, self.image.n_frames, nchannels, 1)
        self.sizes_xyzct = [size_xyzct]
        pixelinfo = pilmode_to_pixelinfo(self.image.mode)
        self.pixel_types = [pixelinfo[0]]
        self.pixel_nbits = [pixelinfo[1]]

        dimension_order = 'yx'
        if self.image.n_frames > 1:
            dimension_order = 'z' + dimension_order
        if nchannels > 1:
            dimension_order += 'c'
        self.dimension_order = dimension_order

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

    def _find_metadata(self):
        self.source_pixel_size = []
        pixel_size_unit = None
        pixel_size_z = None

        description = self.metadata.get('ImageDescription', '')
        if description != '':
            metadata = desc_to_dict(description)
            if 'spacing' in metadata:
                pixel_size_unit = metadata.get('unit', '')
                if not isinstance(pixel_size_unit, str):
                    pixel_size_unit = 'micrometer'
                pixel_size_z = metadata['spacing']
        if not pixel_size_unit:
            pixel_size_unit = self.metadata.get('ResolutionUnit')
            if pixel_size_unit is not None:
                pixel_size_unit = str(RESUNIT(pixel_size_unit).name).lower()
                if pixel_size_unit == 'none':
                    pixel_size_unit = ''
        res0 = self.metadata.get('XResolution')
        if res0 is not None:
            self.source_pixel_size.append((1 / float(res0), pixel_size_unit))
        res0 = self.metadata.get('YResolution')
        if res0 is not None:
            self.source_pixel_size.append((1 / float(res0), pixel_size_unit))
        if pixel_size_z is not None:
            self.source_pixel_size.append((pixel_size_z, pixel_size_unit))
        self.source_mag = self.metadata.get('Mag', 0)
        self.channels = [{'label': ''}]

    def load(self):
        self.unload()
        for level in range(len(self.sizes)):
            self.arrays.append(self._asarray_level(level))
        self.loaded = True

    def unload(self):
        for array in self.arrays:
            del array
        self.arrays = []
        self.loaded = False

    def _asarray_level(self, level: int, **slicing) -> np.ndarray:
        nframes = self.image.n_frames
        if self.loaded:
            image = self.arrays[level]
        elif nframes > 1:
            shape = [nframes] + list(np.array(self.image).shape)
            image = np.zeros(shape, dtype=self.pixel_types[level])
            for framei in range(nframes):
                self.image.seek(framei)
                image[framei] = np.array(self.image)
        else:
            image = np.array(self.image)

        dimension_order = self.dimension_order
        slicing = get_numpy_slicing(dimension_order, **slicing)
        out = redimension_data(image[slicing], dimension_order, self.get_dimension_order())
        return out
