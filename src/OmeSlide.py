import numpy as np
from PIL import Image

from src.image_util import precise_resize, resize


class OmeSlide:
    def asarray(self, x0=0, y0=0, x1=-1, y1=-1):
        # ensure fixed patch size
        if x1 < 0 or y1 < 0:
            x1, y1 = self.get_size()
        # ensure fixed patch size
        w0 = x1 - x0
        h0 = y1 - y0
        if self.mag_factor != 1:
            ox0, oy0 = int(round(x0 * self.mag_factor)), int(round(y0 * self.mag_factor))
            ox1, oy1 = int(round(x1 * self.mag_factor)), int(round(y1 * self.mag_factor))
        else:
            ox0, oy0, ox1, oy1 = x0, y0, x1, y1
        image0 = self.asarray_level(0, ox0, oy0, ox1, oy1)
        if self.mag_factor != 1:
            w = int(round(image0.shape[1] / self.mag_factor))
            h = int(round(image0.shape[0] / self.mag_factor))
            pil_image = Image.fromarray(image0).resize((w, h))
            image = np.array(pil_image)
        else:
            image = image0
        w = image.shape[1]
        h = image.shape[0]
        if (h, w) != (h0, w0):
            image = np.pad(image, ((0, h0 - h), (0, w0 - w), (0, 0)), 'edge')
        return image

    def get_size(self):
        # size at selected magnification
        return np.divide(self.sizes[self.best_page], self.best_factor).astype(int)

    def get_thumbnail(self, target_size, precise=False):
        size, index = get_best_size(self.sizes, target_size)
        scale = np.divide(target_size, self.sizes[index])
        image = self.asarray_level(index, 0, 0, size[0], size[1])
        if np.round(scale, 3)[0] == 1 and np.round(scale, 3)[1] == 1:
            return image
        elif precise:
            return precise_resize(image, scale)
        else:
            return resize(image, target_size)


def get_best_size(sizes, target_size):
    # find largest scale but smaller to 1
    best_index = -1
    best_scale = 0
    for index, size in enumerate(sizes):
        scale = np.mean(np.divide(target_size, size))
        if 1 >= scale > best_scale:
            best_index = index
            best_scale = scale
    return sizes[best_index], best_index
