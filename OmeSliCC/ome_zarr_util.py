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


def create_transformation_metadata(dimension_order, pixel_size_um, scale, translation=[]):
    metadata = []
    pixel_size_scale = []
    translation_scale = []
    for dimension in dimension_order:
        if dimension == 'z' and len(pixel_size_um) > 2:
            pixel_size_scale1 = pixel_size_um[2]
        elif dimension == 'y' and len(pixel_size_um) > 1:
            pixel_size_scale1 = pixel_size_um[1] / scale
        elif dimension == 'x' and len(pixel_size_um) > 0:
            pixel_size_scale1 = pixel_size_um[0] / scale
        else:
            pixel_size_scale1 = 1
        if pixel_size_scale1 == 0:
            pixel_size_scale1 = 1
        pixel_size_scale.append(pixel_size_scale1)

        if dimension == 'z' and len(translation) > 2:
            translation1 = translation[2]
        elif dimension == 'y' and len(translation) > 1:
            translation1 = translation[1] / scale
        elif dimension == 'x' and len(translation) > 0:
            translation1 = translation[0] / scale
        else:
            translation1 = 0
        translation_scale.append(translation1)

    metadata.append({'type': 'scale', 'scale': pixel_size_scale})
    if not all(v == 0 for v in translation_scale):
        metadata.append({'type': 'translation', 'translation': translation_scale})
    return metadata


def create_channel_metadata(source):
    channels = source.get_channels()
    nchannels = source.get_nchannels()

    if len(channels) < nchannels == 3:
        labels = ['Red', 'Green', 'Blue']
        colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        channels = [{'label': label, 'color': color} for label, color in zip(labels, colors)]

    omezarr_channels = []
    for channeli, channel0 in enumerate(channels):
        channel = channel0.copy()
        if 'color' in channel:
            color = rgba_to_hexrgb(channel['color'])
        else:
            color = ''
        channel['color'] = color
        if 'window' not in channel:
            channel['window'] = source.get_channel_window(channeli)
        omezarr_channels.append(channel)

    metadata = {
        'version': '0.4',
        'channels': omezarr_channels,
    }
    return metadata


def calc_shape_scale(shape0, dimension_order, scale):
    shape = []
    if scale == 1:
        return shape0
    for shape1, dimension in zip(shape0, dimension_order):
        if dimension in ['x', 'y']:
            shape1 = int(round(shape1 * scale))
        shape.append(shape1)
    return shape
