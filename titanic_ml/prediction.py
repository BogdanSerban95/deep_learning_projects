from keras.models import load_model
from data import get_test_data
import pandas as pd
from config import *
import os.path as osp
import numpy as np
from utils import ensure_dir

MDL_PTH = osp.join(CKPT_FLD, 'pred_v2/model.h5')
OUTPUTS_FILE = osp.join(OUTPUTS_FLD, 'outputs_v1.csv')


def predict_labels():
    ensure_dir(OUTPUTS_FLD)
    X_test, ids = get_test_data()
    model = load_model(MDL_PTH)
    preds = np.round(model.predict(X_test)).reshape(-1)
    print(preds)
    df = pd.DataFrame(np.array([ids.astype(np.int), preds.astype(np.int)]).transpose(),
                      columns=['PassengerId', 'Survived'])
    df.to_csv(OUTPUTS_FILE, index=False)
    print('Done.')


if __name__ == '__main__':
    predict_labels()
