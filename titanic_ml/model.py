from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.regularizers import l2

L2_NORM = 1e-6


def get_model(in_size):
    input = Input((in_size,), name='input')
    x = Dense(16, activation='relu', kernel_regularizer=l2(L2_NORM))(input)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(L2_NORM))(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[x])

    # opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3, clipvalue=0.01)
    model.compile(optimizer=opt, loss=binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


def test_model():
    get_model(20)


if __name__ == '__main__':
    test_model()
