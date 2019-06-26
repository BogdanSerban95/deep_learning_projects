from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.regularizers import l2, l1
from custom_losses import jaccard_distance_loss, focal_loss

L2_NORM = 1e-3


def get_model(in_size):
    input = Input((in_size,), name='input')
    x = Dense(512, activation='sigmoid', kernel_regularizer=l1(L2_NORM))(input)
    # x = Dropout(0.25)(x)
    # x = Dense(8, activation='sigmoid', kernel_regularizer=l2(L2_NORM))(x)
    # x = Dropout(0.25)(x)
    # x = Dense(8, activation='sigmoid', kernel_regularizer=l2(L2_NORM))(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    #
    # x = Dense(128, activation='relu', kernel_regularizer=l2(L2_NORM))(x)
    # x = Dropout(0.25)(x)
    # x = Dense(128, activation='relu', kernel_regularizer=l2(L2_NORM))(x)
    # x = Dropout(0.25)(x)
    # x = Dense(128, activation='relu', kernel_regularizer=l2(L2_NORM))(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[x])

    # opt = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    model.compile(optimizer=opt, loss=binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


def test_model():
    get_model(20)


if __name__ == '__main__':
    test_model()
