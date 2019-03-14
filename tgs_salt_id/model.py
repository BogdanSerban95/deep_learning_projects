from keras.layers import *
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.models import Model
from custom_metric import comp_metric
from custom_losses import jaccard_distance_loss


def conv2d_block(input_tensor, n_filters, k_size=3, batch_norm=True):
    x = Conv2D(
        filters=n_filters,
        kernel_size=(k_size, k_size),
        kernel_initializer='he_normal',
        padding='same'
    )(input_tensor)
    if batch_norm:
        x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(
        filters=n_filters,
        kernel_size=(k_size, k_size),
        kernel_initializer='he_uniform',
        padding='same'
    )(x)
    if batch_norm:
        x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)

    return x


def get_model(in_w, in_h, n_filters=16, dropout=0.5, batch_norm=True):
    input = Input((in_h, in_w, 1), name="input_layer")
    c1 = conv2d_block(input, n_filters=n_filters, k_size=3, batch_norm=batch_norm)
    p1 = MaxPooling2D((2, 2))(c1)
    # p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, k_size=3, batch_norm=batch_norm)
    p2 = MaxPooling2D((2, 2))(c2)
    # p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, k_size=3, batch_norm=batch_norm)
    p3 = MaxPooling2D((2, 2))(c3)
    # p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, k_size=3, batch_norm=batch_norm)
    p4 = MaxPooling2D((2, 2))(c4)
    # p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, k_size=3, batch_norm=batch_norm)

    u6 = Conv2DTranspose(n_filters * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    # u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, k_size=3, batch_norm=batch_norm)

    u7 = Conv2DTranspose(n_filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    # u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, k_size=3, batch_norm=batch_norm)

    u8 = Conv2DTranspose(n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    # u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, k_size=3, batch_norm=batch_norm)

    u9 = Conv2DTranspose(n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    # u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters, k_size=3, batch_norm=batch_norm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input], outputs=[outputs])
    model.compile(optimizer=Adam(lr=1e-3, clipvalue=0.01), loss=jaccard_distance_loss, metrics=['acc', comp_metric])
    model.summary()
    return model


if __name__ == '__main__':
    model = get_model(96, 96, batch_norm=False)
