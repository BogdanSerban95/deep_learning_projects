import cv2
import numpy as np
import os.path as osp
from config import *
import utils

PREV_SIZE = (640, 640)


def do_main():
    files, paths = utils.get_files_from_dir(PTH_TRAIN_IMGS)
    num_files = len(files)
    perm = np.random.permutation(num_files)
    for i in perm:
        img = cv2.imread(paths[i])
        img = cv2.resize(img, PREV_SIZE, img, interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(osp.join(PTH_TRAIN_MASKS, files[i]))
        mask = cv2.resize(mask, PREV_SIZE, mask, interpolation=cv2.INTER_LINEAR)

        utils.named_window(img, 'Image', PREV_SIZE, False)
        utils.named_window(mask, 'Mask', PREV_SIZE)


if __name__ == '__main__':
    do_main()
