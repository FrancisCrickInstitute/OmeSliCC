import cv2 as cv
import itertools
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances
from tifffile import tifffile
from tqdm import tqdm


matplotlib.rcParams['figure.dpi'] = 300


def convert_colors(colors, dtype=np.uint8):
    if np.dtype(dtype).kind != 'f':
        max_val = 2 ** (8 * np.dtype(dtype).itemsize) - 1
        colors = np.multiply(colors, max_val).astype(dtype).tolist()
    return colors


def vary_colors(color0, color_var, items):
    n = len(items)
    colors = color0 + (np.random.rand(n, 3).astype(np.float32) - 0.5) * color_var
    final_colors = np.clip(colors, 0, 1)
    return final_colors


def vary_colors1(color0, color_var, link_map):
    colors = [color0]
    for map0 in link_map[1:]:
        # find near colors, take mean, add variance
        colors1 = np.mean([colors[index] for index in map0], 0) + (np.random.rand(3) - 0.5) * color_var
        colors.append(colors1)
    final_colors = np.clip(colors, 0, 1)
    return final_colors


class ImageGenerator:
    def __init__(self, size, nscales=2, seed=None):
        if seed is not None:
            np.random.seed(seed)
        factor = 20
        range1 = np.arange(-0.5, 0.5, 1 / factor)
        range2 = [range1] * len(size)
        range3 = []
        for position in itertools.product(*range2):
            dist = np.linalg.norm(position)
            if dist < 0.48:
                range3.append(position)
        range3.sort(key=lambda pos: np.linalg.norm(pos))
        rads = [np.linalg.norm(r) for r in range3]

        # pre-calc & efficiently store whole centers1/colors1?
        dist_matrix = euclidean_distances(range3)
        link_matrix = (dist_matrix != 0) & (dist_matrix < 0.1)
        link_map = [(np.where(link_matrix[i] & (rads < rads[i] - 0.01))[0]) for i, r1 in enumerate(range3)]

        centers = [[np.divide(size, 2, dtype=np.float32)]]
        colors = [[np.array([0.5, 0.5, 0.5], dtype=np.float32)]]
        diameters = []
        color_var = 0.5
        diameter = size
        for _ in tqdm(range(nscales)):
            centers1 = []
            colors1 = []
            for center0, color0 in zip(centers[-1], colors[-1]):
                new_centers = center0 + np.multiply(range3, diameter)
                centers1.extend(new_centers)
                colors1.extend(vary_colors(color0, color_var, link_map))
            centers.append(centers1)
            colors.append(colors1)
            diameter = np.divide(diameter, factor)
            diameters.append(diameter)
            color_var *= 0.5

        # only use final centers
        self.centers = np.round(np.asarray(centers[-1])).astype(int)
        self.colors = convert_colors(colors[-1], dtype)
        self.diameter = diameters[-1]

    def get_tiles(self, tile_size, dtype=np.uint8):
        centers = self.centers
        colors = np.array(self.colors)
        diameter = self.diameter
        radius = np.ceil(diameter / 2).astype(int)

        ranges = np.ceil(np.divide(size, tile_size)).astype(int)
        # flip: cycle over indices in x, y, z order using range = [z, y, x]
        for indices in tqdm(list(np.ndindex(tuple(ranges)))):
            range0 = np.flip(indices) * tile_size
            range1 = np.min([range0 + tile_size, size], 0)
            shape = list(reversed(range1 - range0)) + [3]
            tile = np.zeros(shape, dtype=dtype)
            selected = np.all(centers >= range0, 1) & np.all(centers < range1, 1)

            centers1 = centers[selected] - range0
            colors1 = colors[selected]
            for center, color in zip(centers1, colors1):
                slice1 = tuple([slice(starts, ends) for starts, ends in np.transpose((center - radius, center + radius))])
                tile[slice1] = color
            # slices & tile in (z,),y,x,c
            yield tile


class SimpleImageGenerator:
    def __init__(self, size, tile_size, dtype=np.uint8, seed=None):
        self.size = size
        self.tile_size = tile_size
        self.dtype = dtype

        if seed is not None:
            np.random.seed(seed)

        self.color_value_table = [np.sin(np.divide(range(dim), dim, dtype=np.float32) * np.pi)
                                  for dim in np.flip(size)]

        # self.noise = np.random.random(size=shape) - 0.5     # uniform
        self.noise = np.random.normal(loc=0, scale=0.1, size=np.flip(tile_size))  # gaussian

    def calc_color(self, *args):
        channels = []
        channel = None
        for index, value in enumerate(reversed(args)):
            #channel = np.sin((value + self.range0[index]) / self.size[index] * np.pi)
            channel = self.color_value_table[index][value + self.range0[index]]
            channels.append(channel)
        while len(channels) < 3:
            channels.append(channel)
        return np.stack(channels, axis=-1)

    def get_tiles(self):
        if np.dtype(dtype).kind != 'f':
            max_val = 2 ** (8 * np.dtype(dtype).itemsize) - 1
        else:
            max_val = 1
        ranges = np.flip(np.ceil(np.divide(self.size, self.tile_size)).astype(int))
        # flip: cycle over indices in x, y, z order using range = [z, y, x]
        for indices in tqdm(list(np.ndindex(tuple(ranges)))):
            self.range0 = np.flip(indices) * tile_size
            self.range1 = np.min([self.range0 + self.tile_size, self.size], 0)
            shape = list(reversed(self.range1 - self.range0))

            tile = np.fromfunction(self.calc_color, shape, dtype=int)

            # apply noise to each channel separately
            for channeli in range(3):
                tile[..., channeli] = np.clip(tile[..., channeli] + self.noise, 0, 1)

            if np.dtype(dtype).kind != 'f':
                tile *= max_val

            # tile in (z,),y,x,c
            yield tile.astype(dtype)
            # TODO: wrap into dask array?


def save_tiff(filename, data, shape=None, dtype=None, tile_size=None, bigtiff=None, ome=False, compression=None):
    if bigtiff is None:
        if shape is not None and dtype is not None:
            datasize = np.prod(shape, dtype=np.uint64) * np.dtype(dtype).itemsize
        else:
            datasize = data.size * data.itemsize
        bigtiff = (datasize > 2 ** 32 - 2 ** 25 and not compression)
    tifffile.imwrite(filename, data, shape=shape, dtype=dtype, tile=tile_size, bigtiff=bigtiff, ome=ome,
                     compression=compression)


def render_image(data, shape, dtype):
    #for slice1, tile in tqdm(data):
    #    image[slice1] = tile

    image = np.zeros(shape, dtype=dtype)
    position = np.array([0] * len(shape))
    for tile in data:
        image[position: position + tile.shape] = tile
        position += tile.shape
        # TODO: increase/carry-over (x,) y, z
    image = image.reshape(shape)

    if len(shape) <= 3 and shape[-1] <= 4:
        show_image(image)
    else:
        i = shape[3] // 2 + 2
        show_image(image[:, :, i, :])


def show_image(image):
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    # (tile) size in x,y(,z)
    size = 2048, 2048, 2048
    tile_size = 2048, 2048, 1
    path = 'D:/slides/test.tiff'
    dtype = np.uint8
    #nscales = 2
    seed = 0

    shape = list(reversed(size)) + [3]
    tile_shape = list(reversed(tile_size[:2]))

    print('init')
    #image_generator = ImageGenerator(size, nscales, seed)
    image_generator = SimpleImageGenerator(size, tile_size, dtype, seed)
    print('init done')

    data = image_generator.get_tiles()
    save_tiff(path, data, shape, dtype, tile_size=tile_shape, ome=True)
    print('save done')

    data = image_generator.get_tiles()
    render_image(data, shape, dtype)
