import cv2
import utils
from config import *
from os.path import join
from keras.models import load_model
import numpy as np
from model import get_model
from data import load_test_images
from progressbar import ProgressBar

MODEL_NAME = 'unet_32/model_tgs_salt.h5'
OUT_DIR = 'outputs_32'
OUT_FILE = 'outputs_32.csv'
ORIG_IMG_SIZE = 101

DO_OPEN = True
SAVE_MASKS =False


def get_rle_mask(mask):
    out = []
    non_zero = cv2.findNonZero(mask)
    if non_zero is None:
        return np.array([])
    non_zero = non_zero.reshape(-1, 2)
    non_zero = list(map(lambda x: x[1] * ORIG_IMG_SIZE + x[0] + 1, non_zero))
    if len(non_zero) == 0:
        return np.array([])
    start_px = non_zero[0]
    out.append(start_px)
    num_non_zero = len(non_zero)
    for i in range(1, num_non_zero):
        if non_zero[i] - non_zero[i - 1] != 1:
            length = non_zero[i - 1] - start_px + 1
            out.append(length)
            out.append(non_zero[i])
            start_px = non_zero[i]

    if len(out) & 2 != 1:
        out.append(non_zero[num_non_zero - 1] - start_px + 1)
    assert len(out) % 2 == 0
    return np.array(out)


def save_rle_masks():
    names, paths = utils.get_files_from_dir(PTH_MASKS)
    print('Loading masks...')
    masks = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]
    print('Done.')

    utils.ensure_dir(OUT_DIR)
    with open(join(OUT_DIR, OUT_FILE), 'w') as out_file:
        out_file.write('id,rle_mask\n')
        bar = ProgressBar(0, len(masks))
        for name, mask in bar(zip(names, masks)):
            rle_mask = get_rle_mask(mask)
            file_name = name.split('.')[0]
            rle = ' '.join(rle_mask.astype(str))
            out_file.write('{}, {}\n'.format(file_name, rle))


def save_masks():
    names, images = load_test_images((128, 128))
    print('Loading model...')
    model_name = join(PTH_CHECKPOINTS, MODEL_NAME)
    print(model_name)
    model = load_model(model_name)

    temp_model = get_model(128, 128, n_filters=32, dropout=0.0)
    temp_model.set_weights(model.get_weights())
    print('Done.')

    mask_preds = temp_model.predict(images[:], verbose=1, batch_size=64)
    mask_preds = [
        cv2.normalize(cv2.resize(mask, (ORIG_IMG_SIZE, ORIG_IMG_SIZE)),
                      None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1) for mask in mask_preds]
    if DO_OPEN:
        print('Performing morphological opening...', end='\r')
        op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_preds = [cv2.morphologyEx(pred, cv2.MORPH_OPEN, op) for pred in mask_preds]
        print('Done.')
    print('Saving masks...')
    out = join(OUT_DIR, PTH_MASKS)
    utils.ensure_dir(out)
    for name, mask in zip(names, mask_preds):
        cv2.imwrite(join(out, name), mask)


def do_main():
    if SAVE_MASKS:
        save_masks()
    else:
        save_rle_masks()
    print('Done.')


if __name__ == '__main__':
    do_main()
