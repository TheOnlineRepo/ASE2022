from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal


def lr_schedule(epoch):
    """学习率调度

    学习率将在 80, 120, 160, 180 轮后依次下降。
    他作为训练期间回调的一部分，在每个时期自动调用。

    # 参数
        epoch (int): 轮次

    # 返回
        lr (float32): 学习率
    """
    lr = 1e-3
    if epoch > 45:
        lr *= 0.5e-3
    elif epoch > 30:
        lr *= 1e-2
    elif epoch > 15:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def conv_layer(inputs, filters):
    x = inputs
    x = Conv2D(filters=filters, kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def baseline_model(layer):
    init = RandomNormal(mean=0.0, stddev=0.1, seed=None)
    input_size = [32, 32, 16, 8]
    model = Sequential()
    input_shape = (input_size[layer], input_size[layer], 16)
    inputs = Input(shape=input_shape)
    x = conv_layer(inputs, 96)
    if layer < 3:
        x = AveragePooling2D(pool_size=2)(x)
    x = conv_layer(x, 32)
    if layer < 2:
        x = AveragePooling2D(pool_size=2)(x)
    x = conv_layer(x, 32)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', kernel_initializer=init)(x)
    model = Model(inputs=inputs, outputs=x)
    return model

def small_model(i):

    input_size = [32, 32, 16, 8]
    input_shape = (32, 32, 16)
    inputs = Input(shape=input_shape)
    x = conv_layer(inputs, 32)
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    return model
