import io
import numpy as np
import PIL.Image
import omero.gateway

from OmeSliCC import Omero
from OmeSliCC.OmeSource import OmeSource
from OmeSliCC.color_conversion import *
from OmeSliCC.omero_metadata import create_ome_metadata_from_omero
from OmeSliCC.XmlDict import XmlDict
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
        image_object = self.omero.get_image_object(image_id)
        self.image_object = image_object

        zsize = get_default(image_object.getPixelSizeZ(), 1)
        nchannels = np.sum([channel.getLogicalChannel().getSamplesPerPixel() for channel in image_object.getChannels()])
        pixel_type = np.dtype(image_object.getPixelsType())
        # currently only support/output yxc - allow default value
        #self.dimension_order = image_object.getPrimaryPixels().getDimensionOrder().getValue().lower()
        self.pixels_store = self.omero.create_pixels_store(self.image_object)
        for resolution in self.pixels_store.getResolutionDescriptions():
            self.sizes.append((resolution.sizeX, resolution.sizeY))
            self.sizes_xyzct.append((resolution.sizeX, resolution.sizeY, zsize, nchannels, 1))
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

        self._init_metadata(str(image_id),
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

    def _find_metadata(self):
        #metadata = get_omero_metadata_dict(self.image_object)
        image_object = self.image_object
        self.source_pixel_size = [(get_default(image_object.getPixelSizeX(), 0), 'µm'),
                                  (get_default(image_object.getPixelSizeY(), 0), 'µm'),
                                  (get_default(image_object.getPixelSizeZ(), 0), 'µm')]
        self.source_mag = image_object.getObjectiveSettings().getObjective().getNominalMagnification()
        self.channels = []
        for channel in image_object.getChannels():
            channel = XmlDict({'@Name': channel.getName(), '@Color': int_to_rgba(channel.getColor().getInt()),
                               '@SamplesPerPixel': channel.getLogicalChannel().getSamplesPerPixel()})
            self.channels.append(channel)

    def create_xml_metadata(self, output_filename: str, combine_rgb: bool = True, pyramid_sizes_add: list = None) -> str:
        return create_ome_metadata_from_omero(self, self.image_object, output_filename, combine_rgb=combine_rgb,
                                              pyramid_sizes_add=pyramid_sizes_add)

    def get_thumbnail(self, target_size: tuple, precise: bool = False) -> np.ndarray:
        image_bytes = self.image_object.getThumbnail(target_size)
        image_stream = io.BytesIO(image_bytes)
        image = np.array(PIL.Image.open(image_stream))
        return image

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        if x1 < 0 or y1 < 0:
            x1, y1 = self.sizes[level]
        pixels_store = self.pixels_store
        w, h = x1 - x0, y1 - y0
        nchannels = self.sizes_xyzct[level][3]
        image = np.zeros((h, w, nchannels), dtype=self.pixel_types[level])
        pixels_store.setResolutionLevel(self.pixels_store_pyramid_order[level])
        for c in range(nchannels):
            tile0 = pixels_store.getTile(0, c, 0, x0, y0, w, h)
            tile = np.frombuffer(tile0, dtype=image.dtype).reshape(h, w)
            image[..., c] = tile
        if nchannels == 1:
            return image[..., 0]
        else:
            return image

    def close(self):
        self.pixels_store.close()
