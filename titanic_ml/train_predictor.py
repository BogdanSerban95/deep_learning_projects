from model import get_model
from data import get_train_data
from config import *
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from utils import ensure_dir

MDL_NAME = 'pred_v2'
ckpt_fld = osp.join(CKPT_FLD, MDL_NAME)
log_fld = osp.join(LOGS_FLD, MDL_NAME)


def get_callbacks():
    return [
        EarlyStopping(patience=20, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint('{}/model.h5'.format(ckpt_fld), verbose=1, save_best_only=True)
    ]


def do_plot(results):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
             label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.show()


def do_train():
    ensure_dir(ckpt_fld)
    X_train, X_test, y_train, y_test = get_train_data(test_split=0.33)
    model = get_model(X_train.shape[1])

    results = model.fit(
        X_train,
        y_train,
        BATCH_SIZE,
        EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=get_callbacks()
    )
    do_plot(results)


if __name__ == '__main__':
    do_train()
