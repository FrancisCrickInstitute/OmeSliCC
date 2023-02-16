import numpy as np
import omero.gateway

from src import Omero
from src.OmeSource import OmeSource


class OmeroSource(OmeSource):
    """Omero image source"""

    omero: Omero
    """Omero instance"""
    image_id: int
    """Omero image id"""
    image_object: omero.gateway.ImageWrapper
    """Omero image object"""

    def __init__(self, omero: Omero, image_id: int, source_mag: float = None, target_mag: float = None, source_mag_required: bool = False):
        super().__init__()
        self.omero = omero
        self.image_id = image_id
        self.image_object = self.omero.get_image_object(image_id)
        self.target_mag = target_mag

        self.pixels_store = self.omero.create_pixels_store(self.image_object)
        for resolution in self.pixels_store.getResolutionDescriptions():
            print(resolution)
        # TODO: init sizes based on resolution
        self.sizes
        self.sizes_xyzct
        self.pixel_types
        self.pixel_nbits

        self._init_metadata(str(image_id), source_mag=source_mag, source_mag_required=source_mag_required)
        self.pixels_store.setResolutionLevel(self.best_level)

    def _find_metadata(self):
        # TODO: use objective settings to get matching mag instead
        image_object = self.image_object
        self.pixel_size = [(image_object.getPixelSizeX(), 'µm'),
                           (image_object.getPixelSizeY(), 'µm'),
                           (image_object.getPixelSizeZ(), 'µm')]
        for channel in image_object.getChannels():
            channell = channel.getLogicalChannel()
            self.channel_info.append((channel.getName(), channell.get.getSamplesPerPixel()))
        self.mag0 = image_object.getInstrument().getObjectives()[0].getNominalMagnification()

    def close(self):
        self.pixels_store.close()

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        w, h = x1 - x0, y1 - y0
        nchannels = self.get_size_xyzct()[4]
        image = np.zeros((w, h, nchannels))
        for c in range(nchannels):
            tile0 = self.pixels_store.getTile(0, c, 0, x0, y0, w, h)
            tile = np.frombuffer(tile0, dtype=image.dtype)
            image[:, :, c] = tile.resize(h, w)
        return image
