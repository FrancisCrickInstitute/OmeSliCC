import numpy as np
import os

from OmeSliCC.TiffSource import TiffSource
from OmeSliCC.image_util import show_image, compare_image_dist_simple, image_resize, precise_resize


def precise_resize0(image: np.ndarray, scale: np.ndarray, use_max: bool = False) -> np.ndarray:
    h, w = np.ceil(image.shape[:2] * scale).astype(int)
    shape = list(image.shape).copy()
    shape[:2] = h, w
    new_image = np.zeros(shape, dtype=np.float32)
    step_size = 1 / scale
    for y in range(h):
        for x in range(w):
            y0, y1 = np.round([y * step_size[1], (y + 1) * step_size[1]]).astype(int)
            x0, x1 = np.round([x * step_size[0], (x + 1) * step_size[0]]).astype(int)
            image1 = image[y0:y1, x0:x1]
            if image1.size > 0:
                if use_max:
                    value = np.max(image1, axis=(0, 1))
                else:
                    value = np.mean(image1, axis=(0, 1))
                new_image[y, x] = value
    return new_image.astype(image.dtype)


def precise_downscale0(image: np.ndarray, patch_size: tuple) -> np.ndarray:
    scale = 1 / np.array(patch_size)
    image2 = precise_resize0(image, scale)
    return image2


def precise_downscale(image: np.ndarray, patch_size: tuple) -> np.ndarray:
    image2 = precise_resize(image, patch_size)
    return image2


if __name__ == '__main__':
    path = 'E:/Personal/Crick/slides/test_images/H&E K130_PR003.ome.tiff'
    patch_size = (256, 256)

    source = TiffSource(path)
    image = source.render(source.asarray(), source.get_dimension_order())
    image1 = precise_downscale0(image, patch_size)
    image2 = precise_downscale(image, patch_size)
    new_size = tuple(np.flip(image2.shape[:2]))
    image3 = image_resize(image, new_size)
    new_size = np.divide(source.get_size(), patch_size)
    image3b = image_resize(image, new_size)
    print(compare_image_dist_simple(image1[:-1, :-1], image2[:-1, :-1]))
    show_image(image1)
    show_image(image2)
    show_image(image3)
    show_image(image3b)
