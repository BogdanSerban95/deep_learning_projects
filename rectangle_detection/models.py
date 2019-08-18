from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from rectangle_detection.coord_conv import CoordinateChannel2D


def get_model(in_w, in_h):
    xi = Input(shape=(in_h, in_w, 1))
    x = CoordinateChannel2D()(xi)
    x = Conv2D(filters=8, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=4, activation=None)(x)

    xo = x
    model = Model(inputs=xi, outputs=xo)
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model


def do_main():
    model = get_model(128, 128)
    model.summary()


if __name__ == '__main__':
    do_main()
