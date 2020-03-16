import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, AveragePooling2D, BatchNormalization, LeakyReLU


# three convolutional layers, with the last layer performing stride-2 down-sampling
def normal_block(x, output_filters, alpha=0.3, dropout_rate=0):
    x = Conv2D(output_filters, (3, 3), padding='same')(x)     # Xavier initialization
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)       # x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(output_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)       # x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(output_filters * 2, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)       # x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    return x

# receptive field: ~380     consider using ~256x256 inputs?
def normal_block_classifier(image_shape, levels, alpha=0.3, init_channels=32, dropout_rate=0):
    inputs = keras.Input(shape=image_shape)
    x = Conv2D(init_channels, (3, 3), padding='same')(inputs)  # Xavier initialization
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)       # x = Activation('relu')(x)
    x = Conv2D(init_channels, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)       # x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    for i in range(levels-1):
        x = normal_block(x, 2**(i+6), alpha=alpha, dropout_rate=dropout_rate)

    x = AveragePooling2D(pool_size=(2, 2), padding='valid')(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(4, activation='softmax')(x)

    return keras.Model(inputs=inputs, outputs=x)



# This model has receptive field of only about 150
def fake_alexnet_model(image_shape):
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=image_shape, kernel_size=(11, 11),
                     strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(4))
    model.add(Activation('softmax'))

    return model



