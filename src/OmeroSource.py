import numpy as np
import omero.gateway

from src import Omero
from src.OmeSource import OmeSource
from src.util import get_default


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

    def __init__(self, omero: Omero, image_id: int, source_mag: float = None, target_mag: float = None, source_mag_required: bool = False):
        super().__init__()
        self.omero = omero
        self.image_id = image_id
        image_object = self.omero.get_image_object(image_id)
        self.image_object = image_object
        self.target_mag = target_mag

        zsize = get_default(image_object.getPixelSizeZ(), 1)
        nchannels = np.sum([channel.getLogicalChannel().getSamplesPerPixel() for channel in image_object.getChannels()])
        pixel_type = np.dtype(image_object.getPixelsType())
        self.pixels_store = self.omero.create_pixels_store(self.image_object)
        for resolution in self.pixels_store.getResolutionDescriptions():
            self.sizes.append((resolution.sizeX, resolution.sizeY))
            self.sizes_xyzct.append((resolution.sizeX, resolution.sizeY, zsize, nchannels, 1))
            self.pixel_types.append(pixel_type)
            self.pixel_nbits.append(pixel_type.itemsize * 8)

        self._init_metadata(str(image_id), source_mag=source_mag, source_mag_required=source_mag_required)
        self.pixels_store.setResolutionLevel(self.best_level)

    def _find_metadata(self):
        # TODO: use objective settings to get matching mag instead
        image_object = self.image_object
        self.pixel_size = [(get_default(image_object.getPixelSizeX(), 0), 'µm'),
                           (get_default(image_object.getPixelSizeY(), 0), 'µm'),
                           (get_default(image_object.getPixelSizeZ(), 0), 'µm')]
        for channel in image_object.getChannels():
            channell = channel.getLogicalChannel()
            self.channel_info.append((channel.getName(), channell.getSamplesPerPixel()))
        self.mag0 = image_object.getInstrument().getObjectives()[0].getNominalMagnification()

    def close(self):
        self.pixels_store.close()

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        pixels_store = self.pixels_store
        pixels_store.setResolutionLevel(level)
        tile_size = pixels_store.getTileSize()
        w, h = x1 - x0, y1 - y0
        nchannels = self.sizes_xyzct[level][4]
        image = np.zeros((h, w, nchannels), dtype=self.pixel_types[level])
        for c in range(nchannels):
            tile0 = pixels_store.getTile(0, c, 0, x0, y0, w, h)
            tile = np.frombuffer(tile0, dtype=image.dtype)
            tile.resize(h, w)
            image[..., c] = tile
        if nchannels == 1:
            return image[..., 0]
        else:
            return image

    def _asarray_level0(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        pixels_store = self.pixels_store
        pixels_store.setResolutionLevel(level)
        width, height, _, nchannels, _ = self.sizes_xyzct[level]
        if x1 > width:
            x1 = width - x0
        if y1 > height:
            y1 = height - x0
        tile_size = pixels_store.getTileSize()
        tile_width, tile_height = tile_size
        tile_x0, tile_y0 = x0 // tile_width, y0 // tile_height
        tile_x1, tile_y1 = np.ceil([x1 / tile_width, y1 / tile_height]).astype(int)
        w = (tile_x1 - tile_x0) * tile_width
        h = (tile_y1 - tile_y0) * tile_height
        out = np.zeros((h, w, nchannels), dtype=self.pixel_types[level])

        tile_list = []
        tile_locations = []
        # TODO: test performance swapping order of channel (first or last)
        for c in range(nchannels):
            for y in range(tile_y0, tile_y1):
                for x in range(tile_x0, tile_x1):
                    tx, ty = x * tile_width, y * tile_height
                    tw, th = tile_size
                    if tx + tw > width:
                        tw = width - tx
                    if ty + th > height:
                        th = height - ty
                    tile_list.append((0, c, 0, tx, ty, tw, th))
                    target_x = (x - tile_x0) * tile_width
                    target_y = (y - tile_y0) * tile_height
                    tile_locations.append((target_x, target_y, c))

        for tile, (x, y, c) in zip(self.image_object.getTiles(tile_list)):
            th, tw = tile.shape
            # tile = np.frombuffer(tile0, dtype=image.dtype)
            # image[..., c] = tile.resize(th, tw)
            out[y:y + th, x:x + tw, c] = tile

        target_y0 = y0 - tile_y0 * tile_height
        target_x0 = x0 - tile_x0 * tile_width
        image = out[target_y0: target_y0 + h, target_x0: target_x0 + w]
        return image
