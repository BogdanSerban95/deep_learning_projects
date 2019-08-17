import rectangle_detection.utils as utils
from rectangle_detection.config import *
from rectangle_detection.config import *
from rectangle_detection.data_loaders import BaseDataLoader

import cv2
import numpy as np


def rand_square_im_worker_fun(queue, im_size, sq_size_range):
    while True:
        batch_x = []
        batch_y = []
        for _ in range(BATCH_SIZE):
            img = np.zeros(im_size, dtype=np.float32)
            sq_width = np.random.randint(int(im_size[1] * sq_size_range[0]), int(im_size[1] * sq_size_range[1]))
            sq_height = np.random.randint(int(im_size[0] * sq_size_range[0]), int(im_size[0] * sq_size_range[1]))
            tl_x = np.random.randint(0, im_size[1] - sq_width)
            tl_y = np.random.randint(0, im_size[0] - sq_height)
            img[tl_y: tl_y + sq_height, tl_x: tl_x + sq_width] = np.ones((sq_height, sq_width), dtype=np.float32)
            batch_x.append(np.expand_dims(img, axis=-1))
            batch_y.append(np.array(
                [(tl_x + sq_width / 2) / im_size[1], (tl_y + sq_height / 2) / im_size[0], sq_width / im_size[1],
                 sq_height / im_size[0]]))
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)
        queue.put((batch_x, batch_y))


def do_main():
    worker_fn_args = {'im_size': (IN_HEIGHT, IN_WIDTH), 'sq_size_range': SQ_SIZE_RANGE}
    data_loader = BaseDataLoader(num_workers=3, worker_func=rand_square_im_worker_fun, worker_func_args=worker_fn_args)
    for i in range(2):
        for x, y in data_loader.get_next_batch(BATCH_SIZE):
            for im, lbl in zip(x, y):
                print(lbl)
                c_x = int(lbl[0] * IN_WIDTH)
                c_y = int(lbl[1] * IN_HEIGHT)
                r_w = int(lbl[2] * IN_WIDTH)
                r_h = int(lbl[3] * IN_HEIGHT)
                cv2.circle(im, (c_x, c_y), 2, (0.5, 0.5, 0.5), 5)
                cv2.circle(im, (c_x - r_w // 2, c_y - r_h // 2), 2, (0.5, 0.5, 0.5), 5)
                cv2.circle(im, (c_x + r_w // 2, c_y + r_h // 2), 2, (0.5, 0.5, 0.5), 5)
                utils.named_window(im, 'Prev', (1024, 768))

    data_loader.stop_all()


if __name__ == '__main__':
    do_main()
