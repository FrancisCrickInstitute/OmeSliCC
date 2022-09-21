import numpy as np
import bioformats
import javabridge
from bioformats.formatreader import ImageReader

from src.OmeSource import OmeSource


class BioSource(OmeSource):
    def __init__(self, filename, target_mag=None, source_mag_required=False):
        self.filename = filename
        self.target_mag = target_mag
        self.indexes = []
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.pixel_nbits = []

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
                self.pixel_nbits.append(dtype.itemsize * 8)
        self.init_res_mag(filename, source_mag_required=source_mag_required)

    def find_metadata_res_mag(self):
        pixel_info = self.ome_metadata.image().Pixels
        pixel_size = [(pixel_info.PhysicalSizeX, pixel_info.PhysicalSizeXUnit),
                      (pixel_info.PhysicalSizeY, pixel_info.PhysicalSizeYUnit),
                      (pixel_info.PhysicalSizeZ, pixel_info.PhysicalSizeZUnit)]
        mag = int(float(self.ome_metadata.instrument().Objective.get_NominalMagnification()))
        return pixel_size, mag

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
