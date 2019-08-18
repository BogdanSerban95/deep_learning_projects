import rectangle_detection.data_loaders as loaders
import rectangle_detection.worker_functions as worker_functions
from rectangle_detection.config import *
import rectangle_detection.models as models
import os.path as osp
import rectangle_detection.utils as utils

MODEL_NAME = 'sq_reg_v1'
CKPT_FLD = osp.join(CHECKPOINTS_FLD, MODEL_NAME)


def do_main():
    utils.ensure_dir(CKPT_FLD)
    model = models.get_model(IN_WIDTH, IN_HEIGHT)
    worker_fn_args = {'im_size': (IN_HEIGHT, IN_WIDTH), 'sq_size_range': SQ_SIZE_RANGE}
    train_loader = loaders.BaseDataLoader(num_workers=2, worker_func=worker_functions.rand_square_im_worker_fun,
                                          worker_func_args=worker_fn_args, limit=TRAIN_STEPS_PER_EPOCH)

    test_loader = loaders.BaseDataLoader(num_workers=2, worker_func=worker_functions.rand_square_im_worker_fun,
                                         worker_func_args=worker_fn_args, limit=TEST_STEPS_PER_EPOCH)

    for epoch in range(EPOCHS):
        print('*' * 10 + 'EPOCH {}'.format(epoch + 1), '*' * 10)

        total_loss = 0.0

        for b_x, b_y in train_loader.get_next_batch():
            res = model.fit(b_x, b_y, batch_size=BATCH_SIZE, epochs=1, verbose=0)
            loss = res.history['loss'][0]
            total_loss += loss
            # print('Loss: {}'.format(loss))

        print("TrainLoss: {}".format(total_loss / TRAIN_STEPS_PER_EPOCH))

        total_loss = 0.0
        for b_x, b_y in test_loader.get_next_batch():
            res = model.evaluate(b_x, b_y, batch_size=BATCH_SIZE, verbose=0)
            loss = res[0]
            total_loss += loss
            # print('Loss: {}'.format(loss))

        print("TestLoss: {}".format(total_loss / TEST_STEPS_PER_EPOCH))
        model.save(osp.join(CKPT_FLD, 'model-{}.h5'.format(epoch + 1)))

    train_loader.stop_all()
    test_loader.stop_all()


if __name__ == '__main__':
    do_main()
