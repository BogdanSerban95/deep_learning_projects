from keras.models import load_model
from model import get_model
import utils
from config import *
import data
from os.path import join

MODEL_NAME = 'model_tgs_salt.h5'


def main():
    X, y, d = data.load_data((IMG_SIZE, IMG_SIZE))
    model_name = join(PTH_CHECKPOINTS, MODEL_NAME)
    model = load_model(model_name)
    model.summary()
    temp_model = get_model(128, 128, n_filters=16, dropout=0.0)
    temp_model.set_weights(model.get_weights())
    print('Done.')

    # mask_preds = temp_model.predict(X[:], verbose=1, batch_size=32)
    res = temp_model.evaluate(X[:], y[:])
    print(res)
    pass


if __name__ == '__main__':
    main()
