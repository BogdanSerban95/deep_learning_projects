import pandas as pd
import numpy as np
from config import *
import os.path as osp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

TRAIN_FILE = osp.join(DATA_FLD, 'train.csv')
TEST_FILE = osp.join(DATA_FLD, 'test.csv')
TGT_COL = 'Survived'
CAT_COLS = ['Sex', 'Cabin', 'Embarked', 'Pclass', 'Age']
RM_COLS = ['Name', 'PassengerId', 'Ticket']
STD_COLS = ['Fare', 'SibSp', 'Parch'] + CAT_COLS

age_intervals = [0, 18, 30, 50, 75, 100]
age_labels = [1, 2, 3, 4, 5]


def convert_to_cat(df, col_name, num_vals=None):
    if num_vals is None:
        num_vals = np.unique(df[col_name]).shape[0]
    new_cols = np.reshape([to_categorical(df[col_name], num_vals)], (-1, num_vals))
    headers = ['{}_{}'.format(col_name, i) for i in range(num_vals)]
    n_df = pd.DataFrame(new_cols, columns=headers)
    df = df.join(n_df)
    return df.drop(col_name, axis=1)


def preprocess_dataframe(df, df_eval, test=False):
    drop_cols = RM_COLS
    # if test:
    id_col = df_eval['PassengerId']
    # else:
    y = df[TGT_COL]

    X = df.drop([TGT_COL], axis=1)
    num_train = X.shape[0]
    X = X.append(df_eval)
    X = X.drop(drop_cols, axis=1)

    X['Fare'] = X['Fare'].fillna(0.)
    X['Cabin'] = X['Cabin'].fillna('None')
    X['Cabin'] = X['Cabin'].apply(lambda x: x[0])

    X['Age'] = pd.cut(X['Age'], bins=age_intervals, labels=age_labels).cat.codes
    X['Age'] = X['Age'].replace(-1, 6)

    X['Embarked'] = X['Embarked'].fillna('Z')

    # if test:
    #     categories = np.load('data/cats.npy').item()
    #     for cat_col in CAT_COLS:
    #         if cat_col == 'Cabin':
    #             X[cat_col] = X[cat_col].apply(lambda x: categories[cat_col][x[0]])
    #         else:
    #             X[cat_col] = X[cat_col].apply(lambda x: categories[cat_col][x])
    #         X = convert_to_cat(X, cat_col, len(list(categories[cat_col].keys())))
    # else:
    categories = {}
    for cat_col in CAT_COLS:
        X[cat_col] = X[cat_col].astype('category')
        categories[cat_col] = {cat: i for i, cat in enumerate(X[cat_col].cat.categories)}
        X[cat_col] = X[cat_col].cat.codes
        # X = convert_to_cat(X, cat_col)

        np.save('data/cats.npy', categories)
    # if test:
    #     return X, id_col
    # else:
    return X[:num_train], y, id_col, X[num_train:]


def get_data(test_split=0.33):
    df = pd.read_csv(TRAIN_FILE)
    df_eval = pd.read_csv(TEST_FILE)
    X, y, id_cols, X_eval = preprocess_dataframe(df, df_eval)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=101)

    for std_col in STD_COLS:
        preprocessor = MinMaxScaler()
        preprocessor.fit(np.reshape(X_train[std_col].tolist(), (-1, 1)))
        X_train.loc[:, std_col] = preprocessor.transform(np.reshape(X_train[std_col].tolist(), (-1, 1))).reshape(-1)
        X_test.loc[:, std_col] = preprocessor.transform(np.reshape(X_test[std_col].tolist(), (-1, 1))).reshape(-1)
        X_eval.loc[:, std_col] = preprocessor.transform(np.reshape(X_eval[std_col].tolist(), (-1, 1))).reshape(-1)

    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), X_eval.to_numpy(), id_cols.to_numpy()


def get_eval_data():
    return get_data()[4:]


def get_train_data(test_split=0.33):
    return get_data(test_split=0.33)[:4]


# def get_test_data():
#     df = pd.read_csv(TEST_FILE)
#     X, id = preprocess_dataframe(df, test=True)
#     return X.to_numpy(), id.to_numpy()


def test_data():
    X_train, X_test, y_train, y_test = get_train_data()
    X_eval, id_cols = get_eval_data()
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(X_eval.shape)
    print(id_cols.shape)
    pass


if __name__ == '__main__':
    test_data()
