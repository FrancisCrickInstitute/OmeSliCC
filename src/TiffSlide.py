# https://pypi.org/project/tifffile/


import os
import numpy as np
import tifffile
from tifffile import TiffFile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from src.image_util import find_ome_magnification, get_tiff_pages, get_best_mag, show_image, precise_resize, \
    get_best_size
from src.util import round_significants


class TiffSlide:
    def __init__(self, filename, target_mag=None, executor=None):
        self.target_mag = target_mag
        if executor is not None:
            self.executor = executor
        else:
            max_workers = (os.cpu_count() or 1) + 4
            self.executor = ThreadPoolExecutor(max_workers)
        self.loaded = False
        self.decompressed = False
        self.data = None
        self.arrays = []
        self.source_mags = []
        self.sizes = []
        self.main_page = -1
        self.best_page = -1
        tiff = TiffFile(filename)
        if tiff.is_ome and tiff.ome_metadata is not None:
            self.ome_metadata = tifffile.xml2dict(tiff.ome_metadata)
        else:
            self.ome_metadata = None
        self.pages = get_tiff_pages(tiff, only_tiled=True)
        for index, page in enumerate(self.pages):
            mag = self.get_mag(page)
            if mag != 0 and self.main_page < 0:
                self.main_page = index
            self.sizes.append((page.imagewidth, page.imagelength))
        if target_mag is not None:
            for page in self.pages:
                source_mag = self.get_mag(page)
                if source_mag == 0:
                    source_mag = self.calc_mag(page)
                self.source_mags.append(source_mag)
        source_mag, self.best_page = get_best_mag(self.source_mags, target_mag)
        self.best_factor = source_mag / target_mag
        if self.best_page < 0:
            self.best_page = 0
            self.best_factor = 1
            if target_mag is not None:
                raise ValueError(f'Error: No suitable magnification available ({self.source_mags})')
        self.fh = tiff.filehandle

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

    def get_thumbnail(self, target_size, precise=False):
        size, index = get_best_size(self.sizes, target_size)
        scale = np.divide(target_size, self.sizes[index])
        image = self.pages[index].asarray()
        if np.round(scale, 3)[0] == 1 and np.round(scale, 3)[1] == 1:
            return image
        elif precise:
            return precise_resize(image, scale)
        else:
            thumb = Image.fromarray(image)
            thumb.thumbnail(target_size, Image.ANTIALIAS)
            return np.asarray(thumb)

    def asarray(self, x0, y0, x1, y1):
        # ensure fixed patch size
        w0 = x1 - x0
        h0 = y1 - y0
        if self.best_factor != 1:
            ox0, oy0 = int(round(x0 * self.best_factor)), int(round(y0 * self.best_factor))
            ox1, oy1 = int(round(x1 * self.best_factor)), int(round(y1 * self.best_factor))
        else:
            ox0, oy0, ox1, oy1 = x0, y0, x1, y1
        image0 = self.asarray_level(self.best_page, ox0, oy0, ox1, oy1)
        if self.best_factor != 1:
            w = int(round(image0.shape[1] / self.best_factor))
            h = int(round(image0.shape[0] / self.best_factor))
            pil_image = Image.fromarray(image0).resize((w, h))
            image = np.array(pil_image)
        else:
            image = image0
        w = image.shape[1]
        h = image.shape[0]
        if (h, w) != (h0, w0):
            image = np.pad(image, ((0, h0 - h), (0, w0 - w), (0, 0)), 'edge')
        return image

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
        def process_decoded(decoded, out=out):
            segment, indices, shape = decoded
            index = indices[-2] // shape[-2]
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
            return func(decoded)

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
        yield from self.executor.map(decode, segments, timeout=1)

    def get_mag(self, page):
        mag = 0
        if page.is_ome and self.ome_metadata is not None:
            mag = find_ome_magnification(self.ome_metadata)
        if mag == 0:
            try:
                tags = {item.split('=')[0].strip(): item.split('=')[1].strip() for item in page.description.split('|')}
                if 'AppMag' in tags:
                    mag = float(tags['AppMag'])
            except:
                pass
        return mag

    def calc_mag(self, page):
        main_page = self.pages[self.main_page]
        mag0 = self.get_mag(main_page)
        size = (page.imagewidth, page.imagelength)
        main_size = (main_page.imagewidth, main_page.imagelength)
        mag = round_significants(np.mean(np.divide(size, main_size)) * mag0, 3)
        return mag

    def get_max_mag(self):
        return np.max(self.source_mags)
