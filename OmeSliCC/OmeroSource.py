import io
import numpy as np
import PIL.Image
import omero.gateway

from OmeSliCC import Omero
from OmeSliCC.OmeSource import OmeSource
from OmeSliCC.color_conversion import *
from OmeSliCC.image_util import redimension_data
from OmeSliCC.omero_metadata import create_ome_metadata_from_omero
from OmeSliCC.util import *


class OmeroSource(OmeSource):
    """Omero image source"""

    omero: Omero
    """Omero instance"""
    image_id: int
    """Omero image id"""
    image_object: omero.gateway.ImageWrapper
    """Omero image object"""
    pixels_store: omero.gateway.ProxyObjectWrapper
    """Raw pixels store object"""
    pixels_store_pyramid_order: list
    """Raw pixels store pyramid sizes order (pixel store level order not guaranteed) """

    def __init__(self,
                 omero: Omero,
                 image_id: int,
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False):

        super().__init__()
        self.omero = omero
        self.image_id = image_id
        image_object = self.omero.get_image_object(image_id)
        self.image_object = image_object

        zsize = get_default(image_object.getSizeZ(), 1)
        nchannels = np.sum([channel.getLogicalChannel().getSamplesPerPixel() for channel in image_object.getChannels()])
        pixel_type = np.dtype(image_object.getPixelsType())

        self.pixels_store = self.omero.create_pixels_store(image_object)
        for resolution in self.pixels_store.getResolutionDescriptions():
            self.sizes.append((resolution.sizeX, resolution.sizeY))
            self.sizes_xyzct.append((resolution.sizeX, resolution.sizeY, zsize, nchannels, 1))
            self.pixel_types.append(pixel_type)
            self.pixel_nbits.append(pixel_type.itemsize * 8)

        if not self.sizes:
            xsize, ysize = image_object.getSizeX(), image_object.getSizeY()
            self.sizes.append((xsize, ysize))
            self.sizes_xyzct.append((xsize, ysize, zsize, nchannels, 1))
            self.pixel_types.append(pixel_type)
            self.pixel_nbits.append(pixel_type.itemsize * 8)

        # Omero API issue: pixel store level order not guaranteed
        default_level = self.pixels_store.getResolutionLevel()
        nlevels = self.pixels_store.getResolutionLevels()
        if default_level != 0:
            # reverse order
            self.pixels_store_pyramid_order = list(reversed(range(nlevels)))
        else:
            # default order
            self.pixels_store_pyramid_order = list(range(nlevels))

        self.is_rgb = nchannels in (3, 4)

        self._init_metadata(image_object.getName(),
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

        # currently only support/output yxc
        self.dimension_order = 'yxc'

    def _find_metadata(self):
        image_object = self.image_object
        self.source_pixel_size = [(get_default(image_object.getPixelSizeX(), 0), self.default_physical_unit),
                                  (get_default(image_object.getPixelSizeY(), 0), self.default_physical_unit),
                                  (get_default(image_object.getPixelSizeZ(), 0), self.default_physical_unit)]
        self.channels = []
        for channeli, channel0 in enumerate(image_object.getChannels()):
            channel = {'label': get_default(channel0.getName(), str(channeli)),
                       'color': int_to_rgba(channel0.getColor().getInt())}
            self.channels.append(channel)

    def create_xml_metadata(self, output_filename: str, combine_rgb: bool = True, pyramid_sizes_add: list = None) -> str:
        return create_ome_metadata_from_omero(self, self.image_object, output_filename, combine_rgb=combine_rgb,
                                              pyramid_sizes_add=pyramid_sizes_add)

    def get_thumbnail(self, target_size: tuple, precise: bool = False) -> np.ndarray:
        image_bytes = self.image_object.getThumbnail(target_size)
        image_stream = io.BytesIO(image_bytes)
        image = np.array(PIL.Image.open(image_stream))
        return image

    def _asarray_level(self, level: int, **slicing) -> np.ndarray:
        x0, x1 = slicing.get('x0', 0), slicing.get('x1', -1)
        y0, y1 = slicing.get('y0', 0), slicing.get('y1', -1)
        c, t, z = slicing.get('c'), slicing.get('t'), slicing.get('z')
        if x1 < 0 or y1 < 0:
            x1, y1 = self.sizes[level]
        if t is None:
            t = 0
        if z is None:
            z = 0

        w, h = x1 - x0, y1 - y0
        if c is not None:
            channels = [c]
        else:
            channels = range(self.get_nchannels())
        shape = h, w, len(channels)
        image = np.zeros(shape, dtype=self.pixel_types[level])
        pixels_store = self.pixels_store
        pixels_store_level = self.pixels_store_pyramid_order[level]
        if pixels_store.getResolutionLevel() != pixels_store_level:
            pixels_store.setResolutionLevel(pixels_store_level)
        for c in channels:
            tile0 = pixels_store.getTile(z, c, t, x0, y0, w, h)
            tile = np.frombuffer(tile0, dtype=image.dtype).reshape(h, w)
            image[..., c] = tile

        out = redimension_data(image, self.dimension_order, self.get_dimension_order())
        return out

    def close(self):
        self.pixels_store.close()
