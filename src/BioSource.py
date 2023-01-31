import numpy as np
import bioformats
import javabridge
import xmltodict
from bioformats.formatreader import ImageReader

from src.OmeSource import OmeSource


class BioSource(OmeSource):
    """bioformats compatible image source"""

    filename: str
    """original filename"""
    indexes: list
    """list of relevant series indexes"""

    def __init__(self, filename: str, target_mag: float = None, source_mag_required: bool = False):
        super().__init__()
        self.filename = filename
        self.target_mag = target_mag

        open_javabridge()

        xml_metadata = bioformats.get_omexml_metadata(filename)
        self.bio_ome_metadata = bioformats.OMEXML(xml_metadata)
        self.metadata = xmltodict.parse(xml_metadata)
        if 'OME' in self.metadata:
            self.metadata = self.metadata['OME']
        self.reader = ImageReader(filename)

        self.indexes = []
        for i in range(self.bio_ome_metadata.get_image_count()):
            pmetadata = self.bio_ome_metadata.image(i).Pixels
            if pmetadata.PhysicalSizeX is not None:
                dtype = np.dtype(pmetadata.PixelType)
                self.indexes.append(i)
                self.sizes.append((pmetadata.SizeX, pmetadata.SizeY))
                self.sizes_xyzct.append((pmetadata.SizeX, pmetadata.SizeY, pmetadata.SizeZ, pmetadata.SizeC, pmetadata.SizeT))
                self.pixel_types.append(dtype)
                self.pixel_nbits.append(dtype.itemsize * 8)
        self._init_metadata(filename, source_mag_required=source_mag_required)

    def _find_metadata(self):
        self._get_ome_metadate()

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        if x1 < 0 or y1 < 0:
            x1, y1 = self.sizes[level]
        xywh = (x0, y0, x1 - x0, y1 - y0)
        image = self.reader.read(series=self.indexes[level], XYWH=xywh, rescale=False)      # don't 'rescale' to 0-1!
        return image

    def close(self):
        self.reader.close()


javabridge_open = False


def open_javabridge():
    global javabridge_open

    javabridge.start_vm(class_path=bioformats.JARS)

    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", 'WARN',
                                           "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

    javabridge_open = True


def close_javabridge():
    javabridge.kill_vm()
