import utils
import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def print_stats(y, show=True):
    num_whites = np.array([
        np.sum(mask == 1) for mask in y
    ])
    print(num_whites.mean())
    print(num_whites.min())
    print(num_whites.max())
    print(num_whites.std())
    zero_count = np.sum(num_whites == 0)
    n_zero_count = np.sum(num_whites != 0)
    print(zero_count, n_zero_count)

    plt.hist(num_whites, 50)
    if show:
        plt.show()


def main():
    X, y, d = data.load_data((101, 101))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)
    print('Train')
    print_stats(y_train, False)
    print('Test')
    print_stats(y_test)
    # plt.hist(d, 50)
    # plt.show()
    # corr = np.corrcoef(num_whites, d)
    # print(corr)
    # num_whites = num_whites / num_whites.max()
    # d = d / np.max(d)
    # plt.scatter(num_whites, d)
    # plt.show()


if __name__ == '__main__':
    main()
