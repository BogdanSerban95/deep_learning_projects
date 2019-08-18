from keras.models import load_model
import numpy as np
import cv2
import rectangle_detection.utils as utils
from rectangle_detection.config import *
import os.path as osp
from rectangle_detection.coord_conv import CoordinateChannel2D

CKPT_PATH = osp.join(CHECKPOINTS_FLD, 'sq_reg_v1', 'model-240.h5')


def get_random_square(im_size, sq_size_range):
    img = np.zeros(im_size, dtype=np.float32)
    sq_width = np.random.randint(int(im_size[1] * sq_size_range[0]), int(im_size[1] * sq_size_range[1]))
    sq_height = np.random.randint(int(im_size[0] * sq_size_range[0]), int(im_size[0] * sq_size_range[1]))
    tl_x = np.random.randint(0, im_size[1] - sq_width)
    tl_y = np.random.randint(0, im_size[0] - sq_height)
    img[tl_y: tl_y + sq_height, tl_x: tl_x + sq_width] = np.ones((sq_height, sq_width), dtype=np.float32)

    return img


def eval_random_images():
    model = load_model(CKPT_PATH, custom_objects={'CoordinateChannel2D': CoordinateChannel2D})
    for i in range(10):
        test_img = get_random_square((IN_HEIGHT, IN_WIDTH), [0.1, 0.85])
        test_img = np.expand_dims(np.expand_dims(test_img, axis=-1), axis=0)
        preds = model.predict(test_img)[0]
        c_x = int(preds[0] * IN_WIDTH)
        c_y = int(preds[1] * IN_HEIGHT)
        r_w = int(preds[2] * IN_WIDTH)
        r_h = int(preds[3] * IN_HEIGHT)
        disp_img = test_img[0]
        cv2.circle(disp_img, (c_x, c_y), 2, (0.5, 0.5, 0.5), 5)
        cv2.circle(disp_img, (c_x - r_w // 2, c_y - r_h // 2), 2, (0.5, 0.5, 0.5), 5)
        cv2.circle(disp_img, (c_x + r_w // 2, c_y + r_h // 2), 2, (0.5, 0.5, 0.5), 5)
        utils.named_window(disp_img, 'Prev', (1024, 768))


def eval_real_image():
    model = load_model(CKPT_PATH, custom_objects={'CoordinateChannel2D': CoordinateChannel2D})
    for i in range(7):
        test_img = cv2.imread('./test_images/test_{}.jpg'.format(i + 1), cv2.IMREAD_GRAYSCALE)
        test_img = cv2.resize(test_img, (IN_WIDTH, IN_HEIGHT))
        test_img = test_img.astype(np.float32) / 255
        test_img = np.expand_dims(test_img, axis=0)
        test_img = np.expand_dims(test_img, axis=-1) if len(test_img.shape) == 3 else test_img
        preds = model.predict(test_img)[0]
        print(preds)
        c_x = int(preds[0] * IN_WIDTH)
        c_y = int(preds[1] * IN_HEIGHT)
        r_w = int(preds[2] * IN_WIDTH)
        r_h = int(preds[3] * IN_HEIGHT)
        disp_img = test_img[0]
        cv2.circle(disp_img, (c_x, c_y), 2, (0.5, 0.5, 0.5), 5)
        cv2.circle(disp_img, (c_x - r_w // 2, c_y - r_h // 2), 2, (0.5, 0.5, 0.5), 5)
        cv2.circle(disp_img, (c_x + r_w // 2, c_y + r_h // 2), 2, (0.5, 0.5, 0.5), 5)
        utils.named_window(disp_img, 'Prev', (1024, 768))


def do_main():
    eval_real_image()
    # eval_random_images()


if __name__ == '__main__':
    do_main()
