import numpy as np
import bioformats
import javabridge
from bioformats.formatreader import ImageReader
import tifffile
from ome_types import OME

from src.OmeSource import OmeSource
from src.util import get_default


class BioSource(OmeSource):
    def __init__(self, filename, target_mag=None, source_mag_required=False):
        super().__init__()
        self.filename = filename
        self.target_mag = target_mag
        self.indexes = []

        open_javabridge()

        xml_metadata = bioformats.get_omexml_metadata(filename)
        self.bio_ome_metadata = bioformats.OMEXML(xml_metadata)
        self.ome_metadata = OME.from_xml(xml_metadata)
        self.metadata = tifffile.xml2dict(xml_metadata)
        if 'OME' in self.metadata:
            self.metadata = self.metadata['OME']
        self.reader = ImageReader(filename)

        for i in range(self.bio_ome_metadata.get_image_count()):
            pmetadata = self.bio_ome_metadata.image(i).Pixels
            if pmetadata.PhysicalSizeX is not None:
                dtype = np.dtype(pmetadata.PixelType)
                self.indexes.append(i)
                self.sizes.append((pmetadata.SizeX, pmetadata.SizeY))
                self.sizes_xyzct.append((pmetadata.SizeX, pmetadata.SizeY, pmetadata.SizeZ, pmetadata.SizeC, pmetadata.SizeT))
                self.pixel_types.append(dtype)
                self.pixel_nbits.append(dtype.itemsize * 8)
        self.init_metadata(filename, source_mag_required=source_mag_required)

    def find_metadata(self):
        pixel_info = self.bio_ome_metadata.image().Pixels
        xyzct = self.sizes_xyzct[0]
        pixel_size = [(get_default(pixel_info.get_PhysicalSizeX(), 0) / xyzct[0], get_default(pixel_info.get_PhysicalSizeXUnit(), '')),
                      (get_default(pixel_info.get_PhysicalSizeY(), 0) / xyzct[1], get_default(pixel_info.get_PhysicalSizeYUnit(), '')),
                      (get_default(pixel_info.get_PhysicalSizeZ(), 0) / xyzct[2], get_default(pixel_info.get_PhysicalSizeZUnit(), ''))]
        mag = int(float(self.bio_ome_metadata.instrument().Objective.get_NominalMagnification()))
        channel_info = []
        for c in range(pixel_info.get_channel_count()):
            channel = pixel_info.Channel(c)
            channel_info.append((channel.get_Name(), channel.get_SamplesPerPixel()))
        self.pixel_size = pixel_size
        self.channel_info = channel_info
        self.mag0 = mag

    def asarray_level(self, level, x0, y0, x1, y1):
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
