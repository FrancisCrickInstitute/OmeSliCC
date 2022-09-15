# https://pypi.org/project/tifffile/
import ast
import os
import re
import numpy as np
import tifffile
from tifffile import TiffFile
from concurrent.futures import ThreadPoolExecutor

from src.OmeSlide import OmeSlide
from src.image_util import get_tiff_pages
from src.ome import create_ome_metadata
from src.util import get_filetitle, tags_to_dict


class TiffSlide(OmeSlide):
    def __init__(self, filename, source_mag=None, target_mag=None, executor=None):
        self.filename = filename
        self.target_mag = target_mag
        self.loaded = False
        self.decompressed = False
        self.data = None
        self.arrays = []
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.pixel_nbytes = []

        if executor is not None:
            self.executor = executor
        else:
            max_workers = (os.cpu_count() or 1) + 4
            self.executor = ThreadPoolExecutor(max_workers)

        tiff = TiffFile(filename)
        if tiff.is_ome and tiff.ome_metadata is not None:
            self.xml_metadata = tiff.ome_metadata
            self.ome_metadata = tifffile.xml2dict(self.xml_metadata)
        else:
            self.xml_metadata = None
            self.ome_metadata = None
        self.pages = get_tiff_pages(tiff, only_tiled=True)
        if len(self.pages) == 0:
            self.pages = get_tiff_pages(tiff, only_tiled=False)
        for page in self.pages:
            self.sizes.append((page.imagewidth, page.imagelength))
            nchannels = 1
            if len(page.shape) > 2:
                nchannels = page.shape[2]
            self.sizes_xyzct.append((page.imagewidth, page.imagelength, nchannels, page.imagedepth, 1))
            self.pixel_types.append(page.dtype)
            self.pixel_nbytes.append(page.bitspersample // 8)
        self.fh = tiff.filehandle
        self.mag0 = self.find_metadata_magnification(filename)
        if self.mag0 == 0 and source_mag is not None:
            self.mag0 = source_mag
        self.init_mags(filename)

    def find_metadata_magnification(self, filename):
        mag = 0
        page = self.pages[0]
        # from OME metadata
        if page.is_ome:
            if 'OME' in self.ome_metadata:
                metadata = self.ome_metadata['OME']
            else:
                metadata = self.ome_metadata
            try:
                mag = metadata['Instrument']['Objective']['NominalMagnification']
            except:
                pass
        if mag == 0:
            # from tiff description
            try:
                desc = page.description
                if desc.startswith('{'):
                    metadata = ast.literal_eval(desc)
                elif '|' in desc:
                    metadata = {item.split('=')[0].strip(): item.split('=')[1].strip() for item in page.description.split('|')}
                if 'AppMag' in metadata:
                    mag = float(metadata['AppMag'])
            except:
                pass
        if mag == 0:
            # from imageJ metadata
            if page.is_imagej:
                metadata = page.imagej_metadata
        if mag == 0:
            # from TAGS
            metadata = tags_to_dict(page.tags)
        if mag == 0:
            # from filename
            title = get_filetitle(filename, remove_all_ext=True)
            matches = re.findall(r'(\d+)x', title)
            if len(matches) > 0:
                mag = float(matches[-1])
        return mag

    def get_metadata(self):
        return self.ome_metadata

    def get_xml_metadata(self, output_filename):
        if self.xml_metadata is not None:
            xml_metadata = self.xml_metadata
        else:
            size = self.get_size()
            xyzct = self.sizes_xyzct[0]
            physical_size = size / self.metadata['dpi']
            physical_size_z = 1
            image_info = {'size_x': size[0], 'size_y': size[1], 'size_z': xyzct[2], 'size_c': xyzct[3],
                          'size_t': xyzct[4],
                          'physical_size_x': physical_size[0], 'physical_size_y': physical_size[1],
                          'physical_size_z': physical_size_z,
                          'dimension_order': 'XYZCT', 'type': self.pixel_types[0].__name__}
            ome_metadata = create_ome_metadata(output_filename, image_info, [])
            xml_metadata = ome_metadata.to_xml()
        return xml_metadata

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
        # size at selected magnification
        return np.divide(self.sizes[self.best_page], self.best_factor).astype(int)

    def asarray_level(self, level, x0, y0, x1, y1):
        if self.decompressed:
            array = self.arrays[level]
            return array[y0:y1, x0:x1]

        # based on tiffile asarray
        dw = x1 - x0
        dh = y1 - y0
        page = self.pages[level]

        tile_width, tile_height = page.tilewidth, page.tilelength
        tile_y0, tile_x0 = y0 // tile_height, x0 // tile_width
        tile_y1, tile_x1 = np.ceil([y1 / tile_height, x1 / tile_width]).astype(int)
        nx = tile_x1 - tile_x0
        ny = tile_y1 - tile_y0
        w = (tile_x1 - tile_x0) * tile_width
        h = (tile_y1 - tile_y0) * tile_height
        tile_per_line = int(np.ceil(page.imagewidth / tile_width))
        dataoffsets = []
        databytecounts = []
        for y in range(tile_y0, tile_y1):
            for x in range(tile_x0, tile_x1):
                index = int(y * tile_per_line + x)
                if index < len(page.databytecounts):
                    dataoffsets.append(page.dataoffsets[index])
                    databytecounts.append(page.databytecounts[index])

        out = np.zeros((h, w, 3), page.dtype)

        self.decode(page, dataoffsets, databytecounts, tile_width, tile_height, nx, out)

        im_y0 = y0 - tile_y0 * tile_height
        im_x0 = x0 - tile_x0 * tile_width
        tile = out[im_y0: im_y0 + dh, im_x0: im_x0 + dw, :]
        return tile

    def decode(self, page, dataoffsets, databytecounts, tile_width, tile_height, nx, out):
        def process_decoded(decoded, index, out=out):
            segment, indices, shape = decoded
            y = index // nx
            x = index % nx

            im_y = y * tile_height
            im_x = x * tile_width
            out[im_y: im_y + tile_height, im_x: im_x + tile_width, :] = segment[0]  # numpy is not thread-safe!

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
            if page.jpegtables is not None:
                decoded = page.decode(*args, page.jpegtables)
            else:
                decoded = page.decode(*args)
            return func(decoded, args[1])

        segments = []
        for index in range(len(dataoffsets)):
            offset = dataoffsets[index]
            bytecount = databytecounts[index]
            if bytecount > 0:
                if self.loaded:
                    segment = self.data[offset:offset + bytecount]
                else:
                    fh = page.parent.filehandle
                    fh.seek(offset)
                    segment = fh.read(bytecount)
            else:
                segment = bytearray()
            segments.append((segment, index))
            #yield decode((segment, index))
        yield from self.executor.map(decode, segments, timeout=10)
