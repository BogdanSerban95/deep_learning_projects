from keras.models import load_model
from model import get_model
import utils
from config import *
import data
from os.path import join
from sklearn.model_selection import train_test_split
import cv2
from custom_metric import comp_metric

MODEL_NAME = 'unet_16_v2/model.h5'


def main():
    X, y, d = data.load_data((IMG_SIZE, IMG_SIZE))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

    model_name = join(PTH_CHECKPOINTS, MODEL_NAME)
    model = load_model(model_name, custom_objects={'comp_metric': comp_metric})
    model.summary()
    temp_model = get_model(IMG_SIZE, IMG_SIZE, n_filters=16, dropout=0.0)
    temp_model.set_weights(model.get_weights())
    print('Done.')

    mask_preds = temp_model.predict(X_test[:], verbose=1, batch_size=32)
    # res = temp_model.evaluate(X_test[:], y_test[:])
    # print(res)
    for t_mask, p_mask in zip(y_test, mask_preds):
        p_mask = cv2.normalize(p_mask, p_mask, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        _, th_p_mask = cv2.threshold(p_mask, 128, 255, cv2.THRESH_BINARY)
        op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        op_pred = cv2.morphologyEx(th_p_mask, cv2.MORPH_OPEN, op)
        utils.named_window(p_mask, 'Predicted mask', (320, 320), False)
        utils.named_window(th_p_mask, 'Predicted mask th', (320, 320), False)
        utils.named_window(op_pred, 'Predicted mask th open', (320, 320), False)

        utils.named_window(t_mask, 'True mask', (320, 320), True)

    pass


if __name__ == '__main__':
    main()
