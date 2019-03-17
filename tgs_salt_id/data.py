import cv2
import utils
from config import *
import numpy as np
import pandas as pd
import os.path as osp
from utils import get_files_from_dir


def load_data(target_size=None):
    files, paths = get_files_from_dir(PTH_TRAIN_IMGS)
    print('Reading images...', end='\r')
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]
    masks = [cv2.imread(osp.join(PTH_TRAIN_MASKS, file), cv2.IMREAD_GRAYSCALE) for file in files]

    if target_size is not None:
        images = [
            cv2.resize(
                image,
                target_size
            ) for image in images]

        masks = [
            cv2.resize(
                mask,
                target_size
            ) for mask in masks]

    images = np.array(images, dtype=np.float32) / 255
    masks = np.array(masks, dtype=np.float32) / 255

    df_depths = pd.read_csv(osp.join(PTH_TRAIN_IMGS, '../depths.csv'))

    images = np.reshape(images, (-1, *target_size, 1))
    masks = np.reshape(masks, (-1, *target_size, 1))

    dict_depths = {}
    for idx, row in df_depths.iterrows():
        dict_depths[row['id']] = row['z']
    # dict_depths = {i[0]: i[1] for i in df_depths.iterrows()}
    depths = [dict_depths[file.split('.')[0]] for file in files]
    print('Done.')
    return images, masks, depths


def load_test_images(target_size):
    files, paths = get_files_from_dir(PTH_TEST_IMAGES)
    images = [
        cv2.resize(
            cv2.imread(path, cv2.IMREAD_GRAYSCALE),
            target_size
        ).astype(np.float32) / 255 for path in paths]
    images = np.reshape(images, (-1, *target_size, 1))

    return files, images


def do_main():
    images, masks, depths = load_data((128, 128))
    for img, msk, d in zip(images, masks, depths):
        print('Depth: {}'.format(d))
        utils.named_window(img, 'Image', (640, 640), False)
        utils.named_window(msk, 'Mask', (640, 640), True)


if __name__ == '__main__':
    do_main()
