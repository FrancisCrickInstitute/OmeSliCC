import numpy as np
import random
import javabridge
import bioformats
from bioformats.formatreader import ImageReader
from javabridge import jdictionary_to_string_dictionary
from PIL import Image
from tqdm import tqdm

from src.util import stringdict_to_dict
from src.image_util import show_image


javabridge_open = False


class BioSlide:
    def __init__(self, filename):
        global javabridge_open
        self.reader: ImageReader
        self.indexes = []
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.mags = []

        if not javabridge_open:
            javabridge.start_vm(class_path=bioformats.JARS)
            javabridge_open = True

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
                self.pixel_types.append(np.dtype(pmeta.PixelType))
                if i == 0:
                    mag = mag0
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

    def get_thumbnail(self, target_size):
        size0 = self.sizes[-1]
        image = self.reader.read(series=self.indexes[-1], XYWH=(0, 0, size0[0], size0[1]))
        thumb = Image.fromarray(image)
        thumb.thumbnail(target_size, Image.ANTIALIAS)
        return np.asarray(thumb)

    def read(self, args):
        image = self.reader.read(series=self.indexes[0], XYWH=args)
        return image

    def close(self):
        self.reader.close()


if __name__ == '__main__':
    #filename = 'D:/Personal/Crick/oRAScle i2i pathology/Oympus/CG_20210609_U_PEA_111_M_L_4_01.vsi'
    #filename = 'D:/Personal/Crick/oRAScle i2i pathology/slides/uncompressed/DB20210802_____03.vsi'
    filename = 'D:/Personal/Crick/oRAScle i2i pathology/slides/multi-region/GK123 R1-6.czi'

    javabridge.start_vm(class_path=bioformats.JARS)
    #bioformats.init_logger()

    slide = BioSlide(filename)
    size = slide.sizes[-1]
    xywh = (size[0]/2, size[1]/2, 512, 512)

    for i in tqdm(range(1000)):
        xywh = (random.randrange(size[0]-512), random.randrange(size[1]-512), 512, 512)
        image = slide.read(xywh)
        #show_image(image)
    slide.close()

    javabridge.kill_vm()
