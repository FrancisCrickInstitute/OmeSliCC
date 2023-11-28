from OmeSliCC.color_conversion import *


def create_axes_metadata(dimension_order):
    axes = []
    for dimension in dimension_order:
        unit1 = None
        if dimension == 't':
            type1 = 'time'
            unit1 = 'millisecond'
        elif dimension == 'c':
            type1 = 'channel'
        else:
            type1 = 'space'
            unit1 = 'micrometer'
        axis = {'name': dimension, 'type': type1}
        if unit1 is not None and unit1 != '':
            axis['unit'] = unit1
        axes.append(axis)
    return axes


def create_transformation_metadata(dimension_order, pixel_size_um, scale):
    pixel_size_scale = []
    for dimension in dimension_order:
        if dimension == 'z':
            pixel_size_scale1 = pixel_size_um[2]
        elif dimension == 'y':
            pixel_size_scale1 = pixel_size_um[1] / scale
        elif dimension == 'x':
            pixel_size_scale1 = pixel_size_um[0] / scale
        else:
            pixel_size_scale1 = 1
        if pixel_size_scale1 == 0:
            pixel_size_scale1 = 1
        pixel_size_scale.append(pixel_size_scale1)
    return [{'scale': pixel_size_scale, 'type': 'scale'}]


def create_channel_metadata(source):
    channels = []
    for channeli, channel0 in enumerate(source.get_channels()):
        channel = {'label': channel0.get('Name', '')}
        color = channel0.get('Color')
        if color is None:
            color = 'FFFFFF'
        else:
            color = rgba_to_hexrgb(color)
        channel['color'] = color
        if 'window' not in channel:
            start, end, min, max = source.get_min_max(channeli)
            channel['window'] = {'start': start, 'end': end, 'min': min, 'max': max}
        channels.append(channel)

    metadata = {
        'version': '0.4',
        'channels': channels,
    }
    return metadata


def calc_shape_scale(shape0, dimension_order, scale):
    shape = []
    if scale == 1:
        return shape0
    for shape1, dimension in zip(shape0, dimension_order):
        if dimension in ['x', 'y']:
            shape1 = int(round(shape1 / scale))
        shape.append(shape1)
    return shape
