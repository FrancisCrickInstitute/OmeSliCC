import numpy as np
import random
import bioformats
import javabridge
from bioformats.formatreader import ImageReader
from javabridge import jdictionary_to_string_dictionary
from tqdm import tqdm

from src.OmeSlide import OmeSlide
from src.util import stringdict_to_dict
from src.image_util import pil_resize, show_image


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
        self.reader = ImageReader(filename)

        self.meta = self.get_metadata1()
        #self.meta = self.get_metadata2()

        # get magnification
        mag0 = int(float(self.meta.instrument().Objective.get_NominalMagnification()))

        for i in range(self.meta.get_image_count()):
            pmeta = self.meta.image(i).Pixels
            if i == 0:
                pmeta0 = pmeta
            if pmeta.PhysicalSizeX is not None:
                self.indexes.append(i)
                self.sizes.append((pmeta.SizeX, pmeta.SizeY))
                self.sizes_xyzct.append((pmeta.SizeX, pmeta.SizeY, pmeta.SizeZ, pmeta.SizeC, pmeta.SizeT))
                self.pixel_nbytes.append(np.dtype(pmeta.PixelType).itemsize)
                if i == 0:
                    mag = mag0
                    self.channels = []
                else:
                    mag = mag0 * (np.mean([pmeta.SizeX, pmeta.SizeY]) / np.mean([pmeta0.SizeX, pmeta0.SizeY]))
                self.mags.append(mag)

    def get_metadata1(self):
        xml = bioformats.get_omexml_metadata(self.filename)
        return bioformats.OMEXML(xml)

    def get_metadata2(self):
        format_reader = self.reader.rdr
        meta = stringdict_to_dict(jdictionary_to_string_dictionary(format_reader.getGlobalMetadata()))
        series_meta = []
        nseries = format_reader.getSeriesCount()
        temp_series = format_reader.getSeries()
        for i in range(nseries):
            format_reader.setSeries(i)
            series_data = stringdict_to_dict(jdictionary_to_string_dictionary(format_reader.getSeriesMetadata()))
            series_meta.append(series_data)
        format_reader.setSeries(temp_series)
        return meta, series_meta

    def get_size(self):
        # size at selected magnification
        return np.divide(self.sizes[self.best_page], self.best_factor).astype(int)

    def get_thumbnail(self, target_size):
        size0 = self.sizes[-1]
        image = self.asarray_level(-1, 0, 0, size0[0], size0[1])
        return pil_resize(image, target_size)

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
