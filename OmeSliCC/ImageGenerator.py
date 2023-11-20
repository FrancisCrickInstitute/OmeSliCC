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
        colors = (np.asarray(colors) * max_val).astype(dtype).tolist()
    return colors


def vary_colors0(color0, color_var, items):
    n = len(items)
    colors = color0 + (np.random.rand(n, 3) - 0.5) * color_var
    final_colors = np.clip(colors, 0, 1)
    return final_colors


def vary_colors(color0, color_var, link_map):
    colors = [color0]
    for map0 in link_map[1:]:
        # find near colors, take mean, add variance
        pool = [colors[index] for index in map0 if index < len(colors)]
        mean_color = np.mean(pool, 0)
        colors.append(mean_color + (np.random.rand(3) - 0.5) * color_var)
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

        centers = [[np.divide(size, 2)]]
        colors = [[(0.5, 0.5, 0.5)]]
        diameters = []
        color_var = 0.5
        diameter = size
        for _ in range(nscales):
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
        colors = self.colors
        diameter = self.diameter
        radius = np.ceil(diameter / 2).astype(int)
        #radius = max(radius)

        range1 = np.ceil(np.divide(size, tile_size)).astype(int)
        for indices in list(np.ndindex(tuple(range1))):
            range0 = np.array(indices) * tile_size
            range1 = np.min([range0 + tile_size, size], 0)
            slices = tuple(reversed([slice(range[0], range[1]) for range in np.transpose((range0, range1))]))
            shape = list(reversed(range1 - range0)) + [3]
            tile = np.zeros(shape, dtype=dtype)
            selected = np.all(centers >= range0, 1) & np.all(centers < range1, 1)

            #for i, selected1, in enumerate(selected):
            #    if selected1:
            #        center = centers[i] - range0
            #        line_type = cv.LINE_AA if radius > 1 else cv.LINE_8
            #        cv.circle(tile, center, radius, colors[i], -1, lineType=line_type)

            centers1 = centers[selected] - range0
            colors1 = np.array(colors)[selected]
            #tile[tuple(reversed(centers1.T))] = colors1
            for center, color in zip(centers1, colors1):
                slice1 = tuple([slice(starts, ends) for starts, ends in np.transpose((center - radius, center + radius))])
                tile[slice1] = color
            # slices & tile in (z,),y,x,c
            #yield slices, tile
            yield tile


def save_tiff(filename, data, shape=None, dtype=None, ome=False):
    tifffile.imwrite(filename, data, shape=shape, dtype=dtype, ome=ome)


def render_image(data, shape, dtype):
    #for slice1, tile in tqdm(data):
    #    image[slice1] = tile

    image = np.zeros(np.prod(shape, dtype=np.uint64), dtype=dtype)
    position = np.uint64(0)
    for tile in tqdm(data):
        tshape = np.prod(tile.shape, dtype=np.uint64)
        image[position: position + tshape] = tile.flatten()
        position += tshape
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
    size = 1024, 1024, 1024
    tile_size = size
    dtype = np.uint8
    nscales = 2
    seed = 0

    shape = list(reversed(size)) + [3]

    print('init')
    image_generator = ImageGenerator(size, nscales, seed)
    print('init done')

    data = image_generator.get_tiles(tile_size, dtype)
    save_tiff('test.tiff', data, shape, dtype, ome=True)
    print('save done')

    data = image_generator.get_tiles(tile_size, dtype)
    render_image(data, shape, dtype)
