import cv2
import utils
from config import *
from os.path import join
from keras.models import load_model
import numpy as np
from model import get_model
from data import load_test_images

MODEL_NAME = 'model_tgs_salt.h5'


def do_main():
    names, images = load_test_images((96, 96))
    model = load_model(join(PTH_CHECKPOINTS, MODEL_NAME))

    temp_model = get_model(96, 96)
    temp_model.set_weights(model.get_weights())

    mask_preds = temp_model.predict(images[:10], verbose=1)
    mask_preds = [cv2.resize(mask, (101, 101)) for mask in mask_preds]



if __name__ == '__main__':
    do_main()
