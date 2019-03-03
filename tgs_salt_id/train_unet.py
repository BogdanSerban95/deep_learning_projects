from config import *
import data
from model import get_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from utils import ensure_dir
from keras.preprocessing.image import ImageDataGenerator

CKPT_DIR = './checkpoints'
LOG_DIR = './logs'


def get_callbacks():
    return [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint('checkpoints/model_tgs_salt.h5', verbose=1, save_best_only=True),
        TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=0,
            write_images=True,
            update_freq='epoch'
        )
    ]


def get_generators(X_train, y_train):
    augmentations = dict(horizontal_flip=True,
                         vertical_flip=True,
              		 rotation_range=15,
                         fill_mode='constant',
                         cval=0,
                         zoom_range=0.05)
    images_data_generator = ImageDataGenerator(**augmentations)
    masks_data_generator = ImageDataGenerator(**augmentations)

    random_state = 101
    return zip(images_data_generator.flow(X_train, seed=random_state, batch_size=BATCH_SIZE,
                                          shuffle=True), masks_data_generator.flow(y_train, seed=random_state,
                                                                                   batch_size=BATCH_SIZE, shuffle=True))


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


def do_main():
    ensure_dir(CKPT_DIR)
    ensure_dir(LOG_DIR)

    X, y, d = data.load_data((IMG_SIZE, IMG_SIZE))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model = get_model(IMG_SIZE, IMG_SIZE, n_filters=16, dropout=0.0, batch_norm=True)

    train_gen = get_generators(X_train, y_train)

    results = model.fit_generator(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        callbacks=get_callbacks(),
        validation_data=(X_test, y_test)
    )
    # do_plot(results)


if __name__ == '__main__':
    do_main()
