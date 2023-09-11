# https://pypi.org/project/tifffile/


import os
from enum import Enum
import numpy as np
from tifffile import TiffFile, TiffPage
from concurrent.futures import ThreadPoolExecutor

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
    executor: ThreadPoolExecutor
    """ThreadPoolExecutor to be used for threaded operations"""

    def __init__(self,
                 filename: str,
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False,
                 executor: ThreadPoolExecutor = None):

        super().__init__()
        self.loaded = False
        self.decompressed = False
        self.data = bytes()
        self.arrays = []

        if executor is not None:
            self.executor = executor
        else:
            max_workers = (os.cpu_count() or 1) + 4
            self.executor = ThreadPoolExecutor(max_workers)

        tiff = TiffFile(filename)
        if tiff.is_ome and tiff.ome_metadata is not None:
            xml_metadata = tiff.ome_metadata
            self.metadata = XmlDict.xml2dict(xml_metadata)
            if 'OME' in self.metadata:
                self.metadata = self.metadata['OME']
                self.has_ome_metadata = True
        elif tiff.is_imagej:
            self.metadata = tiff.imagej_metadata

        self.pages = get_tiff_pages(tiff)
        for page0 in self.pages:
            npages = len(page0)
            if isinstance(page0, list):
                page = page0[0]
            else:
                page = page0
            shape = page.shape
            if isinstance(page, TiffPage):
                width = page.imagewidth
                height = page.imagelength
                depth = page.imagedepth * npages
                bitspersample = page.bitspersample
            else:
                width = shape[1]
                height = shape[0]
                depth = npages
                bitspersample = page.dtype.itemsize * 8
            nchannels = shape[2] if len(shape) > 2 else 1
            nt = 1
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

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

    def _find_metadata(self):
        pixel_size = []
        pixel_size_unit = ''
        channels = []
        mag = 0
        page = self.pages[0]
        if isinstance(page, list):
            page = page[0]
        # from OME metadata
        if page.is_ome:
            self._get_ome_metadate()
            return

        # from imageJ metadata
        pixel_size_z = None
        if len(pixel_size) == 0 and self.metadata is not None and 'spacing' in self.metadata:
            pixel_size_unit = self.metadata.get('unit', '')
            pixel_size_z = (self.metadata['spacing'], pixel_size_unit)
        if mag == 0 and self.metadata is not None:
            mag = self.metadata.get('Mag', 0)
        # from page TAGS
        metadata = tags_to_dict(page.tags)
        if len(pixel_size) < 2:
            if pixel_size_unit == '':
                pixel_size_unit = metadata.get('ResolutionUnit', '')
                if isinstance(pixel_size_unit, Enum):
                    pixel_size_unit = pixel_size_unit.name
                pixel_size_unit = pixel_size_unit.lower()
                if pixel_size_unit == 'none':
                    pixel_size_unit = ''
            res0 = metadata.get('XResolution')
            if res0 is not None:
                if isinstance(res0, tuple):
                    res0 = res0[0] / res0[1]
                if res0 != 0:
                    pixel_size.append((1 / res0, pixel_size_unit))
            res0 = metadata.get('YResolution')
            if res0 is not None:
                if isinstance(res0, tuple):
                    res0 = res0[0] / res0[1]
                if res0 != 0:
                    pixel_size.append((1 / res0, pixel_size_unit))
        if len(channels) == 0:
            nchannels = self.sizes_xyzct[0][3]
            photometric = str(metadata.get('PhotometricInterpretation', '')).lower().split('.')[-1]
            channels = [XmlDict.XmlDict({'@Name': photometric, '@SamplesPerPixel': nchannels})]
        if mag == 0:
            mag = metadata.get('Mag', 0)
        # from description
        if not page.is_ome:
            metadata = desc_to_dict(page.description)
            if mag == 0:
                mag = metadata.get('Mag', metadata.get('AppMag', 0))
            if len(pixel_size) < 2 and 'MPP' in metadata:
                pixel_size.append((metadata['MPP'], 'µm'))
                pixel_size.append((metadata['MPP'], 'µm'))
        if pixel_size_z is not None and len(pixel_size) == 2:
            pixel_size.append(pixel_size_z)
        self.source_pixel_size = pixel_size
        self.source_mag = mag
        self.channels = channels

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
                    array.append(self._decompress_page(page1))
                array = np.asarray(array)
            else:
                array = self._decompress_page(page)
            self.arrays.append(array)
        self.decompressed = True

    def clear_decompress(self):
        self.decompressed = False
        for array in self.arrays:
            del array
        self.arrays = []

    def _decompress_page(self, page: TiffPage) -> np.ndarray:
        pw = page.shape[1]
        ph = page.shape[0]
        array = np.zeros(page.shape, page.dtype)
        tile_width = page.tilewidth
        tile_height = page.tilelength
        x, y = 0, 0
        index = 0
        for offset, bytecount in zip(page.dataoffsets, page.databytecounts):
            if bytecount > 0:
                data = self.data[offset:offset + bytecount]
                if page.jpegtables is not None:
                    decoded = page.decode(data, index, page.jpegtables)
                else:
                    decoded = page.decode(data, index)
                tile = decoded[0]
                dw = tile.shape[-2]
                dh = tile.shape[-3]
                if x + dw > pw:
                    dw = pw - x
                if y + dh > ph:
                    dh = ph - y
                array[y:y + dh, x:x + dw, :] = tile[0, 0:dh, 0:dw, :]
            x += tile_width
            if x >= page.imagewidth:
                x = 0
                y += tile_height
            index += 1
        #self.decode(page, page.dataoffsets, page.databytecounts, tile_width, tile_height, nx, array)  # numpy is not thread-safe!
        return array

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        # based on tiffile asarray
        if x1 < 0 or y1 < 0:
            x1, y1 = self.sizes[level]

        if self.decompressed:
            array = self.arrays[level]
            return array[y0:y1, x0:x1]

        dw = x1 - x0
        dh = y1 - y0
        page0 = self.pages[level]
        page = page0[0] if isinstance(page0, list) else page0
        size_xyzct = self.sizes_xyzct[level]
        n = size_xyzct[2] * size_xyzct[3]

        tile_height, tile_width = page.chunks[0], page.chunks[1]
        tile_y0, tile_x0 = y0 // tile_height, x0 // tile_width
        tile_y1, tile_x1 = np.ceil([y1 / tile_height, x1 / tile_width]).astype(int)
        w = (tile_x1 - tile_x0) * tile_width
        h = (tile_y1 - tile_y0) * tile_height
        tile_per_line = int(np.ceil(page.imagewidth / tile_width))

        out = np.zeros((h, w, n), page.dtype)

        dataoffsets = []
        databytecounts = []
        tile_locations = []
        for i, page in enumerate(page0):
            for y in range(tile_y0, tile_y1):
                for x in range(tile_x0, tile_x1):
                    index = int(y * tile_per_line + x)
                    if index < len(page.databytecounts):
                        offset = page.dataoffsets[index]
                        count = page.databytecounts[index]
                        if count > 0:
                            dataoffsets.append(offset)
                            databytecounts.append(count)
                            target_y = (y - tile_y0) * tile_height
                            target_x = (x - tile_x0) * tile_width
                            tile_locations.append((target_y, target_x, i))

            self._decode(page, dataoffsets, databytecounts, tile_locations, out)

        target_y0 = y0 - tile_y0 * tile_height
        target_x0 = x0 - tile_x0 * tile_width
        image = out[target_y0: target_y0 + dh, target_x0: target_x0 + dw]
        if n == 1:
            return image[..., 0]
        else:
            return image

    def _decode(self, page: TiffPage, dataoffsets: list, databytecounts: list, tile_locations: list, out: np.ndarray):
        def process_decoded(decoded, index, out=out):
            segment, indices, shape = decoded
            y, x, i = tile_locations[index]
            _, h, w, n = shape
            # Note: numpy is not thread-safe
            out[y: y + h, x: x + w, i: i + n] = segment[0]

        for _ in self._segments(
                func=process_decoded,
                page=page,
                dataoffsets=dataoffsets,
                databytecounts=databytecounts
        ):
            pass

    def _segments(self, func: callable, page: TiffPage, dataoffsets: list, databytecounts: list) -> tuple:
        # based on tiffile segments
        def decode(args, page=page, func=func):
            decoded = page.decode(*args, jpegtables=page.jpegtables)
            return func(decoded, args[1])

        tile_segments = []
        for index in range(len(dataoffsets)):
            offset = dataoffsets[index]
            bytecount = databytecounts[index]
            if self.loaded:
                segment = self.data[offset:offset + bytecount]
            else:
                fh = page.parent.filehandle
                fh.seek(offset)
                segment = fh.read(bytecount)
            tile_segments.append((segment, index))
        yield from self.executor.map(decode, tile_segments, timeout=10)