"""
modified from https://github.com/dvlab-research/GridMask/blob/master/imagenet_grid/utils/grid.py
"""
import torch
import numpy as np
from PIL import Image


class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.shape[1]
        w = img.shape[2]

        hh = int(1.5 * h)

        self.d1 = 2
        self.d2 = min(h, w)
        d = np.random.randint(self.d1, self.d2)

        box_len = np.random.randint(1, d)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + box_len
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + box_len
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        mask = torch.tensor(mask.copy()).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        img = img * mask

        return img