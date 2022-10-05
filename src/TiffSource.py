# https://pypi.org/project/tifffile/


import os
import numpy as np
import tifffile
from ome_types import OME
from tifffile import TiffFile, TiffPage
from concurrent.futures import ThreadPoolExecutor

from src.OmeSource import OmeSource
from src.image_util import get_tiff_pages
from src.util import tags_to_dict, desc_to_dict, ensure_list


class TiffSource(OmeSource):
    def __init__(self, filename, source_mag=None, target_mag=None, source_mag_required=False, executor=None):
        super().__init__()
        self.filename = filename
        self.target_mag = target_mag
        self.loaded = False
        self.decompressed = False
        self.data = None
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
            self.metadata = tifffile.xml2dict(xml_metadata)
            if 'OME' in self.metadata:
                self.metadata = self.metadata['OME']
        elif tiff.is_imagej:
            self.metadata = tiff.imagej_metadata
        else:
            self.metadata = None

        self.pages = get_tiff_pages(tiff, only_tiled=True)
        if len(self.pages) == 0:
            self.pages = get_tiff_pages(tiff, only_tiled=False)
        for page in self.pages:
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
            self.sizes.append((width, height))
            self.sizes_xyzct.append((width, height, depth, nchannels, 1))
            self.pixel_types.append(page.dtype)
            self.pixel_nbits.append(bitspersample)

        self.fh = tiff.filehandle
        self.init_metadata(filename, source_mag=source_mag, source_mag_required=source_mag_required)

    def find_metadata(self):
        pixel_size = []
        pixel_size_unit = ''
        channel_info = []
        mag = 0
        page = self.pages[0]
        xyzct = self.sizes_xyzct[0]
        # from OME metadata
        if page.is_ome:
            metadata = self.metadata
            for imetadata in ensure_list(metadata.get('Image', {})):
                pmetadata = imetadata.get('Pixels', {})
                pixel_size = [(pmetadata.get('PhysicalSizeX', 0) / xyzct[0], pmetadata.get('PhysicalSizeXUnit', 'µm')),
                              (pmetadata.get('PhysicalSizeY', 0) / xyzct[1], pmetadata.get('PhysicalSizeYUnit', 'µm')),
                              (pmetadata.get('PhysicalSizeZ', 0) / xyzct[2], pmetadata.get('PhysicalSizeZUnit', 'µm'))]
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

    def load(self, decompress=False):
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
            self.arrays.append(self.decompress_page(page))
        self.decompressed = True

    def clear_decompress(self):
        self.decompressed = False
        for array in self.arrays:
            del array
        self.arrays = []

    def decompress_page(self, page):
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

    def get_size(self):
        # size at target magnification
        return np.divide(self.sizes[self.best_page], self.best_factor).astype(int)

    def asarray_level(self, level, x0, y0, x1, y1):
        # based on tiffile asarray
        if self.decompressed:
            array = self.arrays[level]
            return array[y0:y1, x0:x1]

        dw = x1 - x0
        dh = y1 - y0
        page = self.pages[level]
        nchannels = self.sizes_xyzct[level][3]

        if page.is_tiled:
            tile_width, tile_height = page.tilewidth, page.tilelength
        elif page.chunks is not None and page.chunks != (1, 1):
            tile_height, tile_width = page.chunks
        else:
            tile_width, tile_height = dw, dh
        tile_y0, tile_x0 = y0 // tile_height, x0 // tile_width
        tile_y1, tile_x1 = np.ceil([y1 / tile_height, x1 / tile_width]).astype(int)
        w = (tile_x1 - tile_x0) * tile_width
        h = (tile_y1 - tile_y0) * tile_height
        tile_per_line = int(np.ceil(page.imagewidth / tile_width))
        dataoffsets = []
        databytecounts = []
        tile_locations = []
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
                        tile_locations.append(
                            (slice(target_y, target_y + tile_height),
                             slice(target_x, target_x + tile_width)))

        out = np.zeros((h, w, nchannels), page.dtype)

        self.decode(page, dataoffsets, databytecounts, tile_locations, out)

        target_y0 = y0 - tile_y0 * tile_height
        target_x0 = x0 - tile_x0 * tile_width
        image = out[target_y0: target_y0 + dh, target_x0: target_x0 + dw, :]
        return image

    def decode(self, page, dataoffsets, databytecounts, tile_locations, out):
        def process_decoded(decoded, index, out=out):
            segment, indices, shape = decoded
            tile_location = tile_locations[index]
            # Note: numpy is not thread-safe
            out[tile_location] = segment[0]

        for _ in self.segments(
                func=process_decoded,
                page=page,
                dataoffsets=dataoffsets,
                databytecounts=databytecounts
        ):
            pass

    def segments(self, func, page, dataoffsets, databytecounts):
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
