import numpy as np

from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.conversion import save_tiff
from OmeSliCC.image_util import tiff_info
from OmeSliCC.ome_metadata_from_input import create_ome_metadata
from OmeSliCC.util import map_dict, file_to_dict


def convert_metadata_custom(metadata0: dict)-> dict:

    mapping_emd = [
        ['Image.AcquisitionDate', 'Acquisition.AcquisitionStartDatetime.DateTime', None, 'isoformat()'],
        ['Instrument.LightSourceGroup.Laser.@Type', 'Acquisition.SourceType'],
        ['Instrument.Detector.@Type', 'DetectorMetadata.DetectorType'],
        ['Image.Pixels.@PhysicalSizeX', 'BinaryResult.PixelSize.height'],
        ['Image.Pixels.@PhysicalSizeY', 'BinaryResult.PixelSize.width'],
        ['Image.Pixels.@PhysicalSizeXUnit', 'BinaryResult.PixelUnitX'],
        ['Image.Pixels.@PhysicalSizeYUnit', 'BinaryResult.PixelUnitY'],
        ['Instrument.Detector.@Model', 'DetectorMetadata.DetectorName'],
        ['Instrument.Detector.@Gain', 'DetectorMetadata.Gain'],
        ['Instrument.Detector.@Offset', 'DetectorMetadata.Offset'],
        ['Instrument.Microscope.@Type', 'Instrument.InstrumentClass'],
        ['Instrument.Microscope.@SerialNumber', 'Instrument.InstrumentId'],
        ['Instrument.Microscope.@Model', 'Instrument.InstrumentModel'],
        ['Instrument.Microscope.@Manufacturer', 'Instrument.Manufacturer'],
        ['Image.Pixels.Plane.@ExposureTime', 'Scan.DwellTime'],
        ['Image.Pixels.Plane.@ExposureTimeUnit', 'Scan.DwellTimeUnit', 's'],
        ['Image.Pixels.Plane.@PositionX', 'Stage.Position.x'],
        ['Image.Pixels.Plane.@PositionY', 'Stage.Position.y'],
        ['Image.Pixels.Plane.@PositionZ', 'Stage.Position.z'],
        ['Image.Pixels.Plane.@PositionXUnit', 'Stage.PositionUnit', 'm'],
        ['Image.Pixels.Plane.@PositionYUnit', 'Stage.PositionUnit', 'm'],
        ['Image.Pixels.Plane.@PositionZUnit', 'Stage.PositionUnit', 'm'],
    ]

    metadata = map_dict(metadata0, mapping_emd)

    return metadata


def convert_image_custom(output_filename, original_metadata, data):
    metadata = convert_metadata_custom(original_metadata)
    xml_metadata = create_ome_metadata(metadata, data, data_dimension_order, output_filename)
    save_tiff(output_filename, data, xml_metadata=xml_metadata)


if __name__ == '__main__':
    output_filename = 'D:/slides/metadata_test.ome.tiff'

    data = np.zeros((16, 16, 1), dtype=np.uint8)
    data_dimension_order = 'yxc'

    original_metadata_path = 'D:/slides/EMD/original_metadata.json'
    original_metadata = file_to_dict(original_metadata_path)

    convert_image_custom(output_filename, original_metadata, data)

    print(tiff_info(output_filename))
    source = TiffSource(output_filename)
    print('pixel size:', source.get_pixel_size())
    print('pixel size [um]:', source.get_pixel_size_micrometer())
