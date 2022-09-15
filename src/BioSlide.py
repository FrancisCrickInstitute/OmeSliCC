import numpy as np
import bioformats
import javabridge
from bioformats.formatreader import ImageReader

from src.OmeSlide import OmeSlide


class BioSlide(OmeSlide):
    def __init__(self, filename, target_mag=None):
        self.filename = filename
        self.target_mag = target_mag
        self.indexes = []
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.pixel_nbytes = []

        open_javabridge()

        self.ome_metadata = bioformats.OMEXML(bioformats.get_omexml_metadata(filename))
        self.reader = ImageReader(filename)

        for i in range(self.ome_metadata.get_image_count()):
            pmetadata = self.ome_metadata.image(i).Pixels
            if pmetadata.PhysicalSizeX is not None:
                dtype = np.dtype(pmetadata.PixelType)
                self.indexes.append(i)
                self.sizes.append((pmetadata.SizeX, pmetadata.SizeY))
                self.sizes_xyzct.append((pmetadata.SizeX, pmetadata.SizeY, pmetadata.SizeZ, pmetadata.SizeC, pmetadata.SizeT))
                self.pixel_types.append(dtype)
                self.pixel_nbytes.append(dtype.itemsize)
        self.mag0 = int(float(self.ome_metadata.instrument().Objective.get_NominalMagnification()))
        self.init_mags(filename)

    def get_metadata(self):
        return self.ome_metadata

    def get_xml_metadata(self, output_filename):
        return self.get_metadata().to_xml()

    def asarray_level(self, level, x0, y0, x1, y1):
        xywh = (x0, y0, x1 - x0, y1 - y0)
        image = self.reader.read(series=self.indexes[level], XYWH=xywh)
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
