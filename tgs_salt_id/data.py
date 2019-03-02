import cv2
import utils
from config import *
import numpy as np
import pandas as pd
import os.path as osp
from utils import get_files_from_dir


def load_data(target_size):
    files, paths = get_files_from_dir(PTH_TRAIN_IMGS)
    images = [
        cv2.resize(
            cv2.imread(path, cv2.IMREAD_GRAYSCALE),
            target_size
        ).astype(np.float32) / 255 for path in paths]
    masks = [
        cv2.resize(
            cv2.imread(osp.join(PTH_TRAIN_MASKS, file), cv2.IMREAD_GRAYSCALE),
            target_size
        ).astype(np.float32) / 255 for file in files]
    df_depths = pd.read_csv(osp.join(PTH_TRAIN_IMGS, '../depths.csv'))

    images = np.reshape(images, (-1, *target_size, 1))
    masks = np.reshape(masks, (-1, *target_size, 1))

    dict_depths = {}
    for idx, row in df_depths.iterrows():
        dict_depths[row['id']] = row['z']
    # dict_depths = {i[0]: i[1] for i in df_depths.iterrows()}
    depths = [dict_depths[file.split('.')[0]] for file in files]

    return images, masks, depths


if __name__ == '__main__':
    images, masks, depths = load_data((128, 128))
    for img, msk, d in zip(images, masks, depths):
        print('Depth: {}'.format(d))
        utils.named_window(img, 'Image', (640, 640), False)
        utils.named_window(msk, 'Mask', (640, 640), True)
