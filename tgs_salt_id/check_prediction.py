import cv2
import utils
import numpy as np
from config import *
import os.path as osp


def main():
    lines = []
    with open('outputs/outputs.csv') as file:
        lines = file.readlines()

    for line in lines[1:]:
        file_name, coords = line.split(',')
        white_pixels = np.array(coords.strip().split(' '))

        mask = cv2.imread('outputs/masks/{}.png'.format(file_name), cv2.IMREAD_GRAYSCALE)
        c_mask = np.zeros(mask.shape, dtype=np.uint8).reshape(-1)
        if white_pixels.shape[0] != 1:
            white_pixels = white_pixels.astype(np.int)
            for i in range(0, len(white_pixels), 2):
                c_mask[white_pixels[i] - 1: white_pixels[i] - 1 + white_pixels[i + 1]] = 255

        c_mask = c_mask.reshape(mask.shape)
        utils.named_window(mask, 'Mask', (320, 320), False)
        utils.named_window(c_mask, 'c_mask', (320, 320), True)


if __name__ == '__main__':
    main()
