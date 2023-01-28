# https://pypi.org/project/tifffile/


import os
from enum import Enum
import numpy as np
import xmltodict
from ome_types import OME
from tifffile import TiffFile, TiffPage
from concurrent.futures import ThreadPoolExecutor

from src.OmeSource import OmeSource
from src.image_util import get_tiff_pages
from src.util import tags_to_dict, desc_to_dict, ensure_list


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

    def __init__(self, filename: str, source_mag: float = None, target_mag: float = None, source_mag_required: bool = False,
                 executor: ThreadPoolExecutor = None):
        super().__init__()
        self.filename = filename
        self.target_mag = target_mag
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
            self.ome_metadata = OME.from_xml(xml_metadata)
            #self.metadata = tifffile.xml2dict(xml_metadata)
            self.metadata = xmltodict.parse(xml_metadata)
            if 'OME' in self.metadata:
                self.metadata = self.metadata['OME']
        elif tiff.is_imagej:
            self.metadata = tiff.imagej_metadata

        self.pages = get_tiff_pages(tiff, only_tiled=True)
        if len(self.pages) == 0:
            self.pages = get_tiff_pages(tiff, only_tiled=False)
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
                depth = page.imagedepth
                bitspersample = page.bitspersample
            else:
                width = shape[1]
                height = shape[0]
                depth = 1
                bitspersample = page.dtype.itemsize * 8
            nchannels = 1
            if len(shape) > 2:
                nchannels = shape[2]
            if tiff.is_ome and npages == 3:
                nchannels *= npages
            else:
                depth *= npages
            self.sizes.append((width, height))
            self.sizes_xyzct.append((width, height, depth, nchannels, 1))
            self.pixel_types.append(page.dtype)
            self.pixel_nbits.append(bitspersample)

        self.fh = tiff.filehandle
        self._init_metadata(filename, source_mag=source_mag, source_mag_required=source_mag_required)

    def _find_metadata(self):
        pixel_size = []
        pixel_size_unit = ''
        channel_info = []
        mag = 0
        page = self.pages[0]
        if isinstance(page, list):
            page = page[0]
        # from OME metadata
        if page.is_ome:
            metadata = self.metadata
            for imetadata in ensure_list(metadata.get('Image', {})):
                pmetadata = imetadata.get('Pixels', {})
                pixel_size = [(pmetadata.get('PhysicalSizeX', 0), pmetadata.get('PhysicalSizeXUnit', 'µm')),
                              (pmetadata.get('PhysicalSizeY', 0), pmetadata.get('PhysicalSizeYUnit', 'µm')),
                              (pmetadata.get('PhysicalSizeZ', 0), pmetadata.get('PhysicalSizeZUnit', 'µm'))]
                for channel in ensure_list(pmetadata.get('Channel', {})):
                    channel_info.append((channel.get('Name', ''), channel.get('SamplesPerPixel', 1)))
            mag = metadata.get('Instrument', {}).get('Objective', {}).get('NominalMagnification', 0)
        # from imageJ metadata
        if len(pixel_size) == 0 and self.metadata is not None:
            pixel_size_unit = self.metadata.get('unit', '')
            pixel_size.append((self.metadata.get('spacing', 0), pixel_size_unit))
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
            res0 = metadata.get('XResolution', 1)
            if isinstance(res0, tuple):
                res0 = res0[0] / res0[1]
            pixel_size.insert(0, (1 / res0, pixel_size_unit))
            res0 = metadata.get('YResolution', 1)
            if isinstance(res0, tuple):
                res0 = res0[0] / res0[1]
            pixel_size.insert(1, (1 / res0, pixel_size_unit))
        if len(channel_info) == 0:
            channel_info = [(str(metadata.get('PhotometricInterpretation', '')).lower().split('.')[-1],
                            metadata.get('SamplesPerPixel', 1))]
        if mag == 0:
            mag = metadata.get('Mag', 0)
        # from description
        if not page.is_ome and mag == 0:
            metadata = desc_to_dict(page.description)
            mag = metadata.get('Mag', 0)
            if mag == 0:
                mag = metadata.get('AppMag', 0)
        self.pixel_size = pixel_size
        self.channel_info = channel_info
        self.mag0 = mag

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
        nchannels = self.sizes_xyzct[level][3]

        tile_height, tile_width = page.chunks[0], page.chunks[1]
        tile_y0, tile_x0 = y0 // tile_height, x0 // tile_width
        tile_y1, tile_x1 = np.ceil([y1 / tile_height, x1 / tile_width]).astype(int)
        w = (tile_x1 - tile_x0) * tile_width
        h = (tile_y1 - tile_y0) * tile_height
        tile_per_line = int(np.ceil(page.imagewidth / tile_width))

        out = np.zeros((h, w, nchannels), page.dtype)

        dataoffsets = []
        databytecounts = []
        tile_locations = []
        for c, page in enumerate(page0):
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
                            tile_locations.append((target_y, target_x, c))

            self._decode(page, dataoffsets, databytecounts, tile_locations, out)

        target_y0 = y0 - tile_y0 * tile_height
        target_x0 = x0 - tile_x0 * tile_width
        image = out[target_y0: target_y0 + dh, target_x0: target_x0 + dw]
        if nchannels == 1:
            return image[..., 0]
        else:
            return image

    def _decode(self, page: TiffPage, dataoffsets: list, databytecounts: list, tile_locations: list, out: np.ndarray):
        def process_decoded(decoded, index, out=out):
            segment, indices, shape = decoded
            y, x, c = tile_locations[index]
            _, h, w, nc = shape
            # Note: numpy is not thread-safe
            out[y: y + h, x: x + w, c: c + nc] = segment[0]

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
