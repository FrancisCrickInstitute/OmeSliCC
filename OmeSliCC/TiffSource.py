# https://pypi.org/project/tifffile/


from concurrent.futures import ThreadPoolExecutor
import dask.array as da
from enum import Enum
import numpy as np
import os
from tifffile import TiffFile, TiffPage, PHOTOMETRIC

from OmeSliCC import XmlDict
from OmeSliCC.OmeSource import OmeSource
from OmeSliCC.image_util import *
from OmeSliCC.util import *


class TiffSource(OmeSource):
    """Tiff-compatible image source"""

    filename: str
    """original filename"""
    compressed: bool
    """if image data is loaded compressed"""
    decompressed: bool
    """if image data is loaded decompressed"""
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
                 source_info_required: bool = False,
                 executor: ThreadPoolExecutor = None):

        super().__init__()
        self.compressed = False
        self.decompressed = False
        self.executor = executor
        self.data = bytes()
        self.arrays = []
        photometric = None
        nchannels = 1

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
        self.tags = tags_to_dict(self.first_page.tags)
        if 'FEI_TITAN' in self.tags:
            metadata = tifffile.xml2dict(self.tags.pop('FEI_TITAN'))
            if 'FeiImage' in metadata:
                metadata = metadata['FeiImage']
            self.metadata.update(metadata)

        if tiff.series:
            series0 = tiff.series[0]
            self.dimension_order = series0.axes
            photometric = series0.keyframe.photometric
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
                photometric = page.photometric
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
                images = self.metadata.get('Image', {})
                image = images
                if isinstance(images, list):
                    for image in images:
                        name = image.get('Name', '')
                        if name in filename:
                            break

                pixels = image.get('Pixels', {})
                depth = int(pixels.get('SizeZ', depth))
                nchannels = int(pixels.get('SizeC', nchannels))
                nt = int(pixels.get('SizeT', nt))
            self.sizes.append((width, height))
            self.sizes_xyzct.append((width, height, depth, nchannels, nt))
            self.pixel_types.append(page.dtype)
            self.pixel_nbits.append(bitspersample)

        self.fh = tiff.filehandle
        self.dimension_order = self.dimension_order.lower().replace('s', 'c')

        self.is_rgb = (photometric in (PHOTOMETRIC.RGB, PHOTOMETRIC.PALETTE) and nchannels in (3, 4))

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

    def _find_metadata(self):
        pixel_size = []
        page = self.first_page
        # from OME metadata
        if page.is_ome:
            self._get_ome_metadate()
            return

        # from imageJ metadata
        pixel_size_z = None
        pixel_size_unit = self.metadata.get('unit', '').encode().decode('unicode_escape')
        if pixel_size_unit == 'micron':
            pixel_size_unit = self.default_physical_unit
        for scale in self.metadata.get('scales', '').split(','):
            scale = scale.strip()
            if scale != '':
                pixel_size.append((float(scale), pixel_size_unit))
        if len(pixel_size) == 0 and self.metadata is not None and 'spacing' in self.metadata:
            pixel_size_z = (self.metadata['spacing'], pixel_size_unit)
        # from description
        if len(pixel_size) < 2 and 'pixelWidth' in self.metadata:
            pixel_info = self.metadata['pixelWidth']
            pixel_size.append((pixel_info['value'], pixel_info['unit']))
            pixel_info = self.metadata['pixelHeight']
            pixel_size.append((pixel_info['value'], pixel_info['unit']))
        if len(pixel_size) < 2 and 'MPP' in self.metadata:
            pixel_size.append((self.metadata['MPP'], self.default_physical_unit))
            pixel_size.append((self.metadata['MPP'], self.default_physical_unit))
        # from page TAGS
        if len(pixel_size) < 2:
            if pixel_size_unit == '':
                pixel_size_unit = self.tags.get('ResolutionUnit', '')
                if isinstance(pixel_size_unit, Enum):
                    pixel_size_unit = pixel_size_unit.name
                pixel_size_unit = pixel_size_unit.lower()
                if pixel_size_unit == 'none':
                    pixel_size_unit = ''
            res0 = convert_rational_value(self.tags.get('XResolution'))
            if res0 is not None and res0 != 0:
                pixel_size.append((1 / res0, pixel_size_unit))
            res0 = convert_rational_value(self.tags.get('YResolution'))
            if res0 is not None and res0 != 0:
                pixel_size.append((1 / res0, pixel_size_unit))

        position = []
        xpos = convert_rational_value(self.tags.get('XPosition'))
        ypos = convert_rational_value(self.tags.get('YPosition'))
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
        self.position = position

    def get_source_dask(self):
        return self._load_as_dask()

    def _load_as_dask(self):
        if len(self.arrays) == 0:
            for level in range(len(self.sizes)):
                data = da.from_zarr(self.tiff.aszarr(level=level))
                if data.chunksize == data.shape:
                    data = data.rechunk()
                self.arrays.append(data)
        return self.arrays

    def _load_as_zarr(self):
        if len(self.arrays) == 0:
            import zarr
            store = self.tiff.aszarr(multiscales=True)
            group = zarr.group(store=store)
            self.arrays = [arr for _, arr in group.arrays()]  # read-only zarr arrays
        return self.arrays

    def load(self, decompress: bool = False):
        if decompress:
            self.decompress()
            self.decompressed = True
        else:
            self.fh.seek(0)
            self.data = self.fh.read()
            self.compressed = True

    def unload(self):
        del self.data
        self.clear_arrays()
        self.compressed = False
        self.decompressed = False

    def decompress(self):
        self.clear_arrays()
        for page in self.pages:
            if isinstance(page, list):
                array = []
                for page1 in page:
                    data = page1.asarray()
                    if len(page) > 1:
                        array.append(data)
                    else:
                        array = data
                array = np.asarray(array)
            else:
                array = page.asarray()
            self.arrays.append(array)

    def clear_arrays(self):
        for array in self.arrays:
            del array
        self.arrays = []

    def _asarray_level(self, level: int, **slicing) -> np.ndarray:
        if self.compressed and not self.decompressed:
            out = self._decompress(level, **slicing)
        else:
            self._load_as_dask()
            redim = redimension_data(self.arrays[level], self.dimension_order, self.get_dimension_order())
            slices = get_numpy_slicing(self.get_dimension_order(), **slicing)
            out = redim[slices]
        return out

    def _decompress(self, level: int, **slicing) -> np.ndarray:
        # based on tiffile asarray

        if self.executor is None:
            max_workers = (os.cpu_count() or 1) + 4
            self.executor = ThreadPoolExecutor(max_workers)

        x0, x1 = slicing.get('x0', 0), slicing.get('x1', -1)
        y0, y1 = slicing.get('y0', 0), slicing.get('y1', -1)
        c, t, z = slicing.get('c'), slicing.get('t'), slicing.get('z')
        if x1 < 0 or y1 < 0:
            x1, y1 = self.sizes[level]

        dw = x1 - x0
        dh = y1 - y0
        xyzct = list(self.sizes_xyzct[level]).copy()
        nz = xyzct[2]
        nc = xyzct[3]

        pages = self.pages[level]
        if nz == self.npages and z is not None:
            pages = [pages[z]]
        elif t is not None:
            pages = [pages[t]]
        page = pages[0] if isinstance(pages, list) else pages
        tile_height, tile_width = page.chunks[:2]
        tile_y0, tile_x0 = y0 // tile_height, x0 // tile_width
        tile_y1, tile_x1 = np.ceil([y1 / tile_height, x1 / tile_width]).astype(int)
        w = (tile_x1 - tile_x0) * tile_width
        h = (tile_y1 - tile_y0) * tile_height
        niter_channels = nc if page.dims[0] == 'sample' else 1
        tile_per_line = int(np.ceil(page.imagewidth / tile_width))
        tile_per_channel = tile_per_line * int(np.ceil(page.imagelength / tile_height))
        xyzct[0] = w
        xyzct[1] = h

        # Match internal Tiff page dimensions [separate sample, depth, length, width, contig sample]
        n = self.npages
        d = self.depth
        s = nc
        if self.npages == s > 1:
            # in case channels represented as pages
            s = 1
        shape = n, d, h, w, s
        out = np.zeros(shape, page.dtype)

        dataoffsets = []
        databytecounts = []
        tile_locations = []
        for pagei, page in enumerate(pages):
            for channeli in range(niter_channels):
                for y in range(tile_y0, tile_y1):
                    for x in range(tile_x0, tile_x1):
                        index = channeli * tile_per_channel + y * tile_per_line + x
                        if index < len(page.databytecounts):
                            offset = page.dataoffsets[index]
                            count = page.databytecounts[index]
                            if count > 0:
                                dataoffsets.append(offset)
                                databytecounts.append(count)
                                target_y = (y - tile_y0) * tile_height
                                target_x = (x - tile_x0) * tile_width
                                tile_locations.append((pagei, 0, target_y, target_x, channeli))

            self._decode(page, dataoffsets, databytecounts, tile_locations, out)

        target_y0 = y0 - tile_y0 * tile_height
        target_x0 = x0 - tile_x0 * tile_width
        image = out[:, :, target_y0: target_y0 + dh, target_x0: target_x0 + dw, :]
        # 'ndyxs' -> 'tzyxc'
        if image.shape[0] == nc > 1:
            image = np.swapaxes(image, 0, -1)
        elif image.shape[0] == nz > 1:
            image = np.swapaxes(image, 0, 1)
        # 'tzyxc' -> 'tczyx'
        image = np.moveaxis(image, -1, 1)
        return image

    def _decode(self, page: TiffPage, dataoffsets: list, databytecounts: list, tile_locations: list, out: np.ndarray):
        def process_decoded(decoded, index, out=out):
            segment, indices, shape = decoded
            s = tile_locations[index]
            e = np.array(s) + ([1] + list(shape))
            # Note: numpy is not thread-safe
            out[s[0]: e[0],
                s[1]: e[1],
                s[2]: e[2],
                s[3]: e[3],
                s[4]: e[4]] = segment

        for _ in self._segments(
                process_function=process_decoded,
                page=page,
                dataoffsets=dataoffsets,
                databytecounts=databytecounts
        ):
            pass

    def _segments(self, process_function: callable, page: TiffPage, dataoffsets: list, databytecounts: list) -> tuple:
        # based on tiffile segments
        def decode(args, page=page, process_function=process_function):
            decoded = page.decode(*args, jpegtables=page.jpegtables)
            return process_function(decoded, args[1])

        tile_segments = []
        for index in range(len(dataoffsets)):
            offset = dataoffsets[index]
            bytecount = databytecounts[index]
            if self.compressed:
                segment = self.data[offset:offset + bytecount]
            else:
                fh = page.parent.filehandle
                fh.seek(offset)
                segment = fh.read(bytecount)
            tile_segment = (segment, index)
            tile_segments.append(tile_segment)
            # yield decode(tile_segment)
        yield from self.executor.map(decode, tile_segments, timeout=10)
