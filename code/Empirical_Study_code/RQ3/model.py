import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet import MobileNet

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D 卷积批量标准化 - 激活栈构建器

    # 参数
        inputs (tensor): 从输入图像或前一层来的输入张量
        num_filters (int): Conv2D 过滤器数量
        kernel_size (int): Conv2D 方形核维度
        strides (int): Conv2D 方形步幅维度
        activation (string): 激活函数名
        batch_normalization (bool): 是否包含批标准化
        conv_first (bool): conv-bn-activation (True) 或
            bn-activation-conv (False)

    # 返回
        x (tensor): 作为下一层输入的张量
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet 版本 1 模型构建器 [a]

    2 x (3 x 3) Conv2D-BN-ReLU 的堆栈
    最后一个 ReLU 在快捷连接之后。
    在每个阶段的开始，特征图大小由具有 strides=2 的卷积层减半（下采样），
    而滤波器的数量加倍。在每个阶段中，这些层具有相同数量的过滤器和相同的特征图尺寸。
    特征图尺寸:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    参数数量与 [a] 中表 6 接近:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # 参数
        input_shape (tensor): 输入图像张量的尺寸
        depth (int): 核心卷积层的数量
        num_classes (int): 类别数 (CIFAR10 为 10)

    # 返回
        model (Model): Keras 模型实例
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # 开始模型定义
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # 实例化残差单元的堆栈
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # 第一层但不是第一个栈
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # 线性投影残差快捷键连接，以匹配更改的 dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)  # 24
        num_filters *= 2

    # 在顶层加分类器。
    # v1 不在最后一个快捷连接 ReLU 后使用 BN
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 实例化模型。
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet 版本 2 模型构建器 [b]

    (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D 的堆栈，也被称为瓶颈层。
    每一层的第一个快捷连接是一个 1 x 1 Conv2D。
    第二个及以后的快捷连接是 identity。
    在每个阶段的开始，特征图大小由具有 strides=2 的卷积层减半（下采样），
    而滤波器的数量加倍。在每个阶段中，这些层具有相同数量的过滤器和相同的特征图尺寸。
    特征图尺寸:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # 参数
        input_shape (tensor): 输入图像张量的尺寸
        depth (int): 核心卷积层的数量
        num_classes (int): 类别数 (CIFAR10 为 10)

    # 返回
        model (Model): Keras 模型实例
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # 开始模型定义。
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 在将输入分离为两个路径前执行带 BN-ReLU 的 Conv2D 操作。
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # 实例化残差单元的栈
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # 瓶颈残差单元
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # 线性投影残差快捷键连接，以匹配更改的 dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # 在顶层添加分类器
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 实例化模型。
    model = Model(inputs=inputs, outputs=outputs)
    return model


def leNet_models(input_shape, modelname):
    model = Sequential()

    kernel_size = (5, 5)
    if modelname == 'leNet_1':
        model.add(Conv2D(4, kernel_size, activation='relu', padding='same', input_shape=input_shape, name='activation1'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

        model.add(Conv2D(12, kernel_size, activation='relu', padding='same', name='activation2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(10, name='dense'))
        model.add(Activation('softmax', name='activation3'))

    elif modelname == 'leNet_4':
        model.add(Conv2D(6, kernel_size, activation='relu', padding='same', input_shape=input_shape, name='activation1'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

        model.add(Conv2D(16, kernel_size, activation='relu', padding='same', name='activation2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

        model.add(Flatten(name='flatten'))

        model.add(Dense(84, activation='relu', name='fc'))
        model.add(Dense(10, name='dense'))
        model.add(Activation('softmax', name='predictions'))

    elif modelname == 'leNet_5':
        model.add(Conv2D(6, kernel_size, activation='relu', padding='same', input_shape=input_shape, name='activation1'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

        model.add(Conv2D(16, kernel_size, activation='relu', padding='same', name='activation2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(120, activation='relu', name='fc1'))
        model.add(Dense(84, activation='relu', name='fc2'))
        model.add(Dense(10, name='dense'))
        model.add(Activation('softmax', name='prediction'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def get_model(dataset, model_name):
    if dataset == 'mnist':
        input_shape = (28, 28, 1)
        model = leNet_models(input_shape, model_name)
    elif model_name == 'resnet20_cifar10':
        n = 3
        depth = n * 6 + 2
        input_shape = (32, 32, 3)
        model =  resnet_v1(input_shape, depth)
    elif model_name == 'resnet50_cifar10':
        n = 9
        depth = n * 6 + 2
        input_shape = (32, 32, 3)
        model =  resnet_v1(input_shape, depth)
    elif model_name == 'MobileNet':
        base_model = MobileNet( weights=None, include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
    model.summary()
    return model