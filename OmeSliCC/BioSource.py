import numpy as np
import bioformats
import javabridge
from bioformats.formatreader import ImageReader

from OmeSliCC import XmlDict
from OmeSliCC.OmeSource import OmeSource
from OmeSliCC.image_util import redimension_data


class BioSource(OmeSource):
    """bioformats compatible image source"""

    filename: str
    """original filename"""
    indexes: list
    """list of relevant series indexes"""

    def __init__(self,
                 filename: str,
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False):

        super().__init__()

        open_javabridge()

        xml_metadata = bioformats.get_omexml_metadata(filename)
        self.bio_ome_metadata = bioformats.OMEXML(xml_metadata)
        self.metadata = XmlDict.xml2dict(xml_metadata)
        if 'OME' in self.metadata:
            self.metadata = self.metadata['OME']
            self.has_ome_metadata = True
        self.reader = ImageReader(filename)

        #self.reader.rdr.getSeriesCount()
        # good images have StageLabel in metadata?
        self.indexes = []
        # TODO: use self.metadata instead of self.bio_ome_metadata
        for i in range(self.bio_ome_metadata.get_image_count()):
            pmetadata = self.bio_ome_metadata.image(i).Pixels
            if pmetadata.PhysicalSizeX is not None:
                dtype = np.dtype(pmetadata.PixelType)
                self.indexes.append(i)
                self.sizes.append((pmetadata.SizeX, pmetadata.SizeY))
                self.sizes_xyzct.append((pmetadata.SizeX, pmetadata.SizeY, pmetadata.SizeZ, pmetadata.SizeC, pmetadata.SizeT))
                self.pixel_types.append(dtype)
                self.pixel_nbits.append(dtype.itemsize * 8)

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

        self.is_rgb = self.get_nchannels() in (3, 4)

        self.dimension_order = 'yx'
        if self.get_nchannels() > 1:
            self.dimension_order += 'c'

    def _find_metadata(self):
        self._get_ome_metadata()

    def _asarray_level(self, level: int, **slicing) -> np.ndarray:
        x0, x1 = slicing.get('x0', 0), slicing.get('x1', -1)
        y0, y1 = slicing.get('y0', 0), slicing.get('y1', -1)
        c, t, z = slicing.get('c'), slicing.get('t'), slicing.get('z')
        if x1 < 0 or y1 < 0:
            x1, y1 = self.sizes[level]
        if t is None:
            t = 0
        if z is None:
            z = 0
        xywh = (x0, y0, x1 - x0, y1 - y0)
        # don't 'rescale' to 0-1!
        image = self.reader.read(series=self.indexes[level], XYWH=xywh, c=c, z=z, t=t, rescale=False)
        out = redimension_data(image, self.dimension_order, self.get_dimension_order())
        return out

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
