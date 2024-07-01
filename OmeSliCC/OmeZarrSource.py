import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from OmeSliCC.OmeSource import OmeSource
from OmeSliCC.color_conversion import hexrgb_to_rgba, int_to_rgba
from OmeSliCC.image_util import get_numpy_slicing, redimension_data


class OmeZarrSource(OmeSource):
    """Zarr-compatible image source"""

    filename: str
    """original filename / URL"""
    levels: list
    """list of all image arrays for different sizes"""

    def __init__(self, filename: str,
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False):

        super().__init__()

        self.levels = []
        nchannels = 1
        try:
            location = parse_url(filename)
            if location is None:
                raise FileNotFoundError(f'Error parsing ome-zarr file {filename}')
            reader = Reader(location)
            # nodes may include images, labels etc
            # first node will be the image pixel data
            image_node = list(reader())[0]

            self.metadata = image_node.metadata
            # channel metadata from ome-zarr-py limited; get from root_attrs manually
            self.root_metadata = reader.zarr.root_attrs

            axes = self.metadata.get('axes', [])
            self.dimension_order = ''.join([axis.get('name') for axis in axes])

            for data in image_node.data:
                #if isinstance(data, np.ndarray):
                #    data = da.from_array(data)
                self.levels.append(data)

                xyzct = [1, 1, 1, 1, 1]
                for i, n in enumerate(data.shape):
                    xyzct_index = 'xyzct'.index(self.dimension_order[i])
                    xyzct[xyzct_index] = n
                self.sizes_xyzct.append(xyzct)
                self.sizes.append((xyzct[0], xyzct[1]))
                self.pixel_types.append(data.dtype)
                self.pixel_nbits.append(data.dtype.itemsize * 8)
                nchannels = xyzct[3]
        except Exception as e:
            raise FileNotFoundError(f'Read error: {e}')

        self.is_rgb = nchannels in (3, 4)

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

    def _find_metadata(self):
        pixel_size = []
        channels = []
        metadata = self.metadata
        axes = self.dimension_order

        units = [axis.get('unit', '') for axis in metadata.get('axes', [])]

        scale1 = [0, 0, 0, 0, 0]
        # get pixelsize using largest/first scale
        transform = self.metadata.get('coordinateTransformations', [])
        if transform:
            transform = transform[0]
            for transform_element in transform:
                if 'scale' in transform_element:
                    scale1 = transform_element['scale']
            for axis in 'xyz':
                if axis in axes:
                    index = axes.index(axis)
                    pixel_size.append((scale1[index], units[index]))
                else:
                    pixel_size.append((1, ''))
        nchannels = self.sizes_xyzct[0][3]
        # look for channel metadata
        for data in self.root_metadata.values():
            if isinstance(data, dict) and 'channels' in data:
                channels = data['channels'].copy()
                for channel in channels:
                    color = channel.pop('color', '')
                    if color != '':
                        if isinstance(color, int):
                            color = int_to_rgba(color)
                        else:
                            color = hexrgb_to_rgba(color)
                        channel['color'] = color
        if len(channels) == 0:
            if self.is_rgb:
                channels = [{'label': ''}]
            else:
                channels = [{'label': ''}] * nchannels
        self.source_pixel_size = pixel_size
        self.channels = channels
        self.source_mag = 0

    def get_source_dask(self):
        return [redimension_data(level, self.dimension_order, self.get_dimension_order())
                for level in self.levels]

    def _asarray_level(self, level: int, **slicing) -> np.ndarray:
        redim = redimension_data(self.levels[level], self.dimension_order, self.get_dimension_order())
        slices = get_numpy_slicing(self.get_dimension_order(), **slicing)
        out = redim[slices]
        return out
