import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import sys
sys.path.append('/home/zhiyu/DeepSuite/adversarial')
from adv_function import *
import argparse
from empirical_study.RQ3.model import *
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator


def load_adv_samples(path, dataset, model, fitness):
    save_dir = os.path.join(path, 'PGD_DeepXplore_2layer', model, fitness)
    train_x = np.load(os.path.join(save_dir, 'train_x.npy'))
    train_y = np.load(os.path.join(save_dir, 'train_y.npy'))
    test_x = np.load(os.path.join(save_dir, 'test_x.npy'))
    test_y = np.load(os.path.join(save_dir, 'test_y.npy'))
    train_y = np.eye(10)[train_y]
    test_y = np.eye(10)[test_y]
    return train_x, train_y, test_x, test_y


def main():
    epochs = 50
    batch_size = 64
    fitness_list = ['kmnc', 'nbc', 'nc', 'snac', 'tknc', 'None']
    # fitness_list = ['None']
    model = get_model(args.dataset, args.model)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [lr_reducer, lr_scheduler]

    _, train_x_ori, train_y_ori, test_x_ori, test_y_ori = load_model_and_testcase(args.path1, args.model, args.dataset)
    train_y_ori = np.eye(10)[train_y_ori]
    test_y_ori = np.eye(10)[test_y_ori]
    for fitness in fitness_list:
        try:
            train_x_adv, train_y_adv, test_x_adv, test_y_adv = load_adv_samples(args.path1, args.dataset, args.model, fitness)
        except:
            continue
        train_x = np.concatenate((train_x_ori, train_x_adv))
        train_y = np.concatenate((train_y_ori, train_y_adv))
        test_x = np.concatenate((test_x_ori, test_x_adv))
        test_y = np.concatenate((test_y_ori, test_y_adv))
        datagen = ImageDataGenerator(
            # 在整个数据集上将输入均值置为 0
            featurewise_center=False,
            # 将每个样本均值置为 0
            samplewise_center=False,
            # 将输入除以整个数据集的 std
            featurewise_std_normalization=False,
            # 将每个输入除以其自身 std
            samplewise_std_normalization=False,
            # 应用 ZCA 白化
            zca_whitening=False,
            # ZCA 白化的 epsilon 值
            zca_epsilon=1e-06,
            # 随机图像旋转角度范围 (deg 0 to 180)
            rotation_range=0,
            # 随机水平平移图像
            width_shift_range=0.1,
            # 随机垂直平移图像
            height_shift_range=0.1,
            # 设置随机裁剪范围
            shear_range=0.,
            # 设置随机缩放范围
            zoom_range=0.,
            # 设置随机通道切换范围
            channel_shift_range=0.,
            # 设置输入边界之外的点的数据填充模式
            fill_mode='nearest',
            # 在 fill_mode = "constant" 时使用的值
            cval=0.,
            # 随机翻转图像
            horizontal_flip=True,
            # 随机翻转图像
            vertical_flip=False,
            # 设置重缩放因子 (应用在其他任何变换之前)
            rescale=None,
            # 设置应用在每一个输入的预处理函数
            preprocessing_function=None,
            # 图像数据格式 "channels_first" 或 "channels_last" 之一
            data_format=None,
            # 保留用于验证的图像的比例 (严格控制在 0 和 1 之间)
            validation_split=0.0)
        datagen.fit(train_x)
        model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
                            validation_data=(test_x, test_y),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)
        scores = model.evaluate(test_x, test_y, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        save_dir = os.path.join(args.path0, 'adv_retrain_models', args.model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, fitness+'.h5')
        model.save(save_path)


if __name__ == '__main__':
    start_time = time.asctime(time.localtime(time.time()))
    print("start time :", start_time)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='cifar10')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='resnet20_cifar10')
    parser.add_argument('-path0', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-path1', type=str, default='/media/data1/DeepSuite')
    args = parser.parse_args()
    print(args)
    main()
