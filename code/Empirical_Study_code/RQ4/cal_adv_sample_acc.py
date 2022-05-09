# encoding=utf-8
import tensorflow as tf
import sys
sys.path.append('/home/zhiyu/DeepSuite/adversarial')
from adv_function import *
import time
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()


from PIL import Image
import PIL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    for dataset in ['cifar10', 'mnist']:
        print("##########", dataset, "##########")
        if dataset == 'mnist':
            model_list = ['leNet_1', 'leNet_4', 'leNet_5']
        else:
            model_list = ['resnet20_cifar10', 'resnet50_cifar10', 'MobileNet']
        for model_name in model_list:
            print("model name:", model_name)
            model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path1, model_name, dataset)

            adv_method_list = ['FGSM', 'DF', 'CW_Linf']
            for adv_method in adv_method_list:
                if args.dataset == 'cifar10':
                    if adv_method == 'FGSM':
                        adv_eps = 0.01
                    elif adv_method == 'DF':
                        adv_eps = 0.1
                    else:
                        adv_eps = 0.3
                else:
                    if adv_method in ['PGD', 'FGSM', 'BIM', 'CW_Linf']:
                        adv_eps = 0.3
                    else:
                        adv_eps = 0.1

                adv_train_x, adv_train_y, adv_valid_x, adv_valid_y, adv_test_x, adv_test_y = load_adv_sample(args.path0, dataset, model_name, adv_method, adv_eps)
                adv_train_y = np.concatenate((adv_train_y, adv_valid_y))
                adv_train_x = np.concatenate((adv_train_x, adv_valid_x))

                # try_sample = x_test[0]*255
                # adv_try_sample = adv_test_x[0]*255
                # # try_sample = x_test[0, :, :, 0]*255
                # # adv_try_sample = adv_test_x[0, :, :, 0]*255
                # test_image = Image.fromarray(try_sample.astype('uint8'))
                # plt.imshow(test_image)
                # plt.savefig('test.png')
                # adv_image = Image.fromarray(adv_try_sample.astype('uint8'))
                # plt.imshow(adv_image)
                # plt.savefig('adv.png')

                adv_train_pred = np.argmax(model.predict(adv_train_x), axis=1)
                adv_test_pred = np.argmax(model.predict(adv_test_x), axis=1)
                adv_train_acc = np.sum(adv_train_pred == adv_train_y) / adv_train_y.shape[0]
                adv_test_acc = np.sum(adv_test_pred == adv_test_y) / adv_test_y.shape[0]
                print("train acc:", adv_train_acc)
                print("test acc:", adv_test_acc)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='cifar10')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='resnet20_cifar10')
    parser.add_argument('-path1', help='directory where models and datasets are stored', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-path0', help='directory where models and datasets are stored', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-save_path', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-test_size', type=float, default=0.1)
    # parser.add_argument('-adv_method', type=str,
    #                     choices=['PGD', 'FGSM', 'BIM', 'DF', 'CW', 'JSMA'], default='FGSM')
    parser.add_argument('-attack_eps_id', type=int, default=0)

    args = parser.parse_args()
    print(args)
    main()