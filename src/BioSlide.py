import numpy as np
import random
import bioformats
import javabridge
from bioformats.formatreader import ImageReader
from javabridge import jdictionary_to_string_dictionary
from tqdm import tqdm

from src.OmeSlide import OmeSlide
from src.util import stringdict_to_dict
from src.image_util import show_image


class BioSlide(OmeSlide):
    def __init__(self, filename, target_mag=None):
        self.reader: ImageReader
        self.indexes = []
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_nbytes = []
        self.mags = []

        open_javabridge()

        self.filename = filename
        self.ome_metadata = bioformats.OMEXML(bioformats.get_omexml_metadata(self.filename))
        self.reader = ImageReader(filename)

        # get magnification
        mag0 = int(float(self.ome_metadata.instrument().Objective.get_NominalMagnification()))

        for i in range(self.ome_metadata.get_image_count()):
            pmetadata = self.ome_metadata.image(i).Pixels
            if i == 0:
                pmetadata0 = pmetadata
            if pmetadata.PhysicalSizeX is not None:
                self.indexes.append(i)
                self.sizes.append((pmetadata.SizeX, pmetadata.SizeY))
                self.sizes_xyzct.append((pmetadata.SizeX, pmetadata.SizeY, pmetadata.SizeZ, pmetadata.SizeC, pmetadata.SizeT))
                self.pixel_nbytes.append(np.dtype(pmetadata.PixelType).itemsize)
                if i == 0:
                    mag = mag0
                    self.channels = []
                else:
                    mag = mag0 * (np.mean([pmetadata.SizeX, pmetadata.SizeY]) / np.mean([pmetadata0.SizeX, pmetadata0.SizeY]))
                self.mags.append(mag)

    def get_metadata(self):
        return self.ome_metadata

    def get_xml_metadata(self, output_filename):
        return self.ome_metadata.to_xml()

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


if __name__ == '__main__':
    #filename = 'D:/Personal/Crick/oRAScle i2i pathology/Oympus/CG_20210609_U_PEA_111_M_L_4_01.vsi'
    #filename = 'D:/Personal/Crick/oRAScle i2i pathology/slides/uncompressed/DB20210802_____03.vsi'
    filename = 'D:/Personal/Crick/oRAScle i2i pathology/slides/multi-region/GK123 R1-6.czi'

    slide = BioSlide(filename)
    size = slide.sizes[-1]
    xywh = (size[0]/2, size[1]/2, 512, 512)

    for i in tqdm(range(1000)):
        x = random.randrange(size[0] - 512)
        y = random.randrange(size[1] - 512)
        image = slide.asarray_level(0, x, y, x + 512, y + 512)
        #show_image(image)
    slide.close()
