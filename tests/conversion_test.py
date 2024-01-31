from tqdm import tqdm

from OmeSliCC.OmeSource import get_resolution_from_pixel_size
from OmeSliCC.OmeZarr import OmeZarr
from OmeSliCC.OmeZarrSource import OmeZarrSource
from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.conversion import save_tiff
from OmeSliCC.image_util import calc_pyramid


def convert_tiff_to_zarr(input, output):
    print('Converting tiff to zarr')
    source = TiffSource(input)
    #data = source.asarray()
    data = source.get_output_dask()    # significantly slower
    zarr = OmeZarr(output)
    zarr.write(data, source, tile_size=tile_size, npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample)


def convert_zarr_to_tiff(input, output):
    print('Converting zarr to tiff')
    source = OmeZarrSource(input)
    dimension_order = source.get_dimension_order()
    #data = source.asarray()
    data = source.get_output_dask()    # significantly slower
    pyramid_sizes_add = calc_pyramid(source.get_size_xyzct(), npyramid_add, pyramid_downsample)
    xml_metadata = source.create_xml_metadata(output, combine_rgb=True,
                                              pyramid_sizes_add=pyramid_sizes_add)
    resolution, resolution_unit = get_resolution_from_pixel_size(source.get_pixel_size())
    save_tiff(output, data, dimension_order=dimension_order, tile_size=tile_size, pyramid_sizes_add=pyramid_sizes_add,
              xml_metadata=xml_metadata, resolution=resolution, resolution_unit=resolution_unit,
              compression='LZW')


if __name__ == '__main__':
    path = 'D:/slides/EM04573_01small.ome.tif'
    path2 = 'D:/slides/test.ome.zarr'
    path3 = 'D:/slides/test.ome.tiff'

    tile_size = 256
    npyramid_add = 3
    pyramid_downsample = 4

    progress = tqdm(range(2))
    convert_tiff_to_zarr(path, path2)
    progress.update(0)
    convert_zarr_to_tiff(path2, path3)
    progress.update(1)

    print('Done')
