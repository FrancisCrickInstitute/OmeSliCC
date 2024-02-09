# https://pypi.org/project/tifffile/


import dask.array as da
from enum import Enum
import numpy as np
import os
from tifffile import TiffFile, TiffPage

from OmeSliCC import XmlDict
from OmeSliCC.OmeSource import OmeSource
from OmeSliCC.image_util import *
from OmeSliCC.util import *


class TiffSource(OmeSource):
    """Tiff-compatible image source"""

    filename: str
    """original filename"""
    loaded: bool
    """if image data is loaded"""
    decompressed: bool
    """if image data is decompressed"""
    pages: list
    """list of all relevant TiffPages"""
    data: bytes
    """raw un-decoded image byte data"""
    arrays: list
    """list of all image arrays for different sizes"""

    def __init__(self,
                 filename: str,
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False):

        super().__init__()
        self.loaded = False
        self.decompressed = False
        self.data = bytes()
        self.arrays = []

        tiff = TiffFile(filename)
        self.tiff = tiff
        self.first_page = tiff.pages.first
        if tiff.is_ome and tiff.ome_metadata is not None:
            xml_metadata = tiff.ome_metadata
            self.metadata = XmlDict.xml2dict(xml_metadata)
            if 'OME' in self.metadata:
                self.metadata = self.metadata['OME']
                self.has_ome_metadata = True
        elif tiff.is_imagej:
            self.metadata = tiff.imagej_metadata
        elif self.first_page.description:
            self.metadata = desc_to_dict(self.first_page.description)
        else:
            self.metadata = tags_to_dict(self.first_page.tags)
            if 'FEI_TITAN' in self.metadata:
                metadata = tifffile.xml2dict(self.metadata.pop('FEI_TITAN'))
                if 'FeiImage' in metadata:
                    metadata = metadata['FeiImage']
                self.metadata.update(metadata)

        if tiff.series:
            self.dimension_order = tiff.series[0].axes
        self.pages = get_tiff_pages(tiff)
        for page0 in self.pages:
            npages = len(page0)
            self.npages = npages
            if isinstance(page0, list):
                page = page0[0]
            else:
                page = page0
            if not self.dimension_order:
                self.dimension_order = page.axes
            shape = page.shape
            nchannels = shape[2] if len(shape) > 2 else 1
            nt = 1
            if isinstance(page, TiffPage):
                width = page.imagewidth
                height = page.imagelength
                self.depth = page.imagedepth
                depth = self.depth * npages
                bitspersample = page.bitspersample
            else:
                width = shape[1]
                height = shape[0]
                depth = npages
                if len(shape) > 2:
                    self.depth = shape[2]
                    depth *= self.depth
                bitspersample = page.dtype.itemsize * 8
            if tiff.is_ome:
                pixels = self.metadata.get('Image', {}).get('Pixels', {})
                depth = int(pixels.get('SizeZ', depth))
                nchannels = int(pixels.get('SizeC', nchannels))
                nt = int(pixels.get('SizeT', nt))
            self.sizes.append((width, height))
            self.sizes_xyzct.append((width, height, depth, nchannels, nt))
            self.pixel_types.append(page.dtype)
            self.pixel_nbits.append(bitspersample)

        self.fh = tiff.filehandle
        self.dimension_order = self.dimension_order.lower().replace('s', 'c')

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

    def _find_metadata(self):
        pixel_size = []
        pixel_size_unit = ''
        page = self.first_page
        # from OME metadata
        if page.is_ome:
            self._get_ome_metadate()
            return

        # from imageJ metadata
        pixel_size_z = None
        if len(pixel_size) == 0 and self.metadata is not None and 'spacing' in self.metadata:
            pixel_size_unit = self.metadata.get('unit', '')
            pixel_size_z = (self.metadata['spacing'], pixel_size_unit)
        # from description
        if len(pixel_size) < 2 and 'pixelWidth' in self.metadata:
            pixel_info = self.metadata['pixelWidth']
            pixel_size.append((pixel_info['value'], pixel_info['unit']))
            pixel_info = self.metadata['pixelHeight']
            pixel_size.append((pixel_info['value'], pixel_info['unit']))
        if len(pixel_size) < 2 and 'MPP' in self.metadata:
            pixel_size.append((self.metadata['MPP'], 'µm'))
            pixel_size.append((self.metadata['MPP'], 'µm'))
        # from page TAGS
        if len(pixel_size) < 2:
            if pixel_size_unit == '':
                pixel_size_unit = self.metadata.get('ResolutionUnit', '')
                if isinstance(pixel_size_unit, Enum):
                    pixel_size_unit = pixel_size_unit.name
                pixel_size_unit = pixel_size_unit.lower()
                if pixel_size_unit == 'none':
                    pixel_size_unit = ''
            res0 = convert_rational_value(self.metadata.get('XResolution'))
            if res0 is not None and res0 != 0:
                pixel_size.append((1 / res0, pixel_size_unit))
            res0 = convert_rational_value(self.metadata.get('YResolution'))
            if res0 is not None and res0 != 0:
                pixel_size.append((1 / res0, pixel_size_unit))

        xpos = convert_rational_value(self.metadata.get('XPosition'))
        ypos = convert_rational_value(self.metadata.get('YPosition'))
        if xpos is not None and ypos is not None:
            position = [(xpos, pixel_size_unit), (ypos, pixel_size_unit)]

        if pixel_size_z is not None and len(pixel_size) == 2:
            pixel_size.append(pixel_size_z)

        mag = self.metadata.get('Mag', self.metadata.get('AppMag', 0))

        nchannels = self.get_nchannels()
        photometric = str(self.metadata.get('PhotometricInterpretation', '')).lower().split('.')[-1]
        if nchannels == 3:
            channels = [{'label': photometric}]
        else:
            channels = [{'label': photometric}] * nchannels

        self.source_pixel_size = pixel_size
        self.source_mag = mag
        self.channels = channels

    def get_source_dask(self):
        return [da.from_zarr(self.tiff.aszarr(level=level)) for level in range(len(self.sizes))]

    def load(self, decompress: bool = False):
        self.fh.seek(0)
        self.data = self.fh.read()
        self.loaded = True
        if decompress:
            self.decompress()

    def unload(self):
        self.loaded = False
        del self.data
        self.clear_decompress()

    def decompress(self):
        self.clear_decompress()
        for page in self.pages:
            if isinstance(page, list):
                array = []
                for page1 in page:
                    array.append(page1.asarray())
                array = np.asarray(array)
            else:
                array = page.asarray()
            self.arrays.append(array)
        self.decompressed = True

    def clear_decompress(self):
        self.decompressed = False
        for array in self.arrays:
            del array
        self.arrays = []

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1,
                       c: int = None, z: int = None, t: int = None) -> np.ndarray:
        if self.decompressed:
            array = self.arrays[level]
            return array[y0:y1, x0:x1]

        data = da.from_zarr(self.tiff.aszarr(level=level))
        x0, x1, y0, y1 = np.round([x0, x1, y0, y1]).astype(int)
        slices = []
        for axis in self.dimension_order:
            if axis == 'x':
                slices.append(slice(x0, x1))
            elif axis == 'y':
                slices.append(slice(y0, y1))
            else:
                slices.append(slice(None))
        out = redimension_data(data[tuple(slices)], self.dimension_order, self.get_dimension_order())
        return out
