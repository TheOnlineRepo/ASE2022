# encoding=utf-8
import os
import csv
import sys
sys.path.append('/home/zhiyu/DeepSuite/adversarial')
from adv_function import *
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True


def main():
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path1, args.model, args.dataset, args.cal_acc)

    adv_method_list = ['PGD', 'FGSM', 'BIM', 'DF', 'CW_Linf']
    # adv_method_list = ['DF']
    for adv_method in adv_method_list:
        print("————————————", adv_method, "————————————")
        if args.dataset == 'cifar10':
            if adv_method in ['PGD', 'FGSM', 'BIM']:
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
        adv_train, _, adv_valid, _, adv_test, _ = load_adv_sample(args.path0, args.dataset, args.model, adv_method, adv_eps)

        save_dir = create_adv_record_dir(args.path0, args.dataset, args.model, adv_method, str(adv_eps))
        train_traces = cal_samples_trace(model, adv_train, args.fitness)
        valid_traces = cal_samples_trace(model, adv_valid, args.fitness)
        test_traces = cal_samples_trace(model, adv_test, args.fitness)
        # if args.attack_eps == 0.3:
        #     suf = '.npy'
        # else:
        #     suf = f'_{args.attack_eps}.npy'
        suf = '.npy'
        if args.fitness == 'nc':
            train_path = os.path.join(save_dir, 'train_nc' + suf)
            valid_path = os.path.join(save_dir, 'valid_nc' + suf)
            test_path = os.path.join(save_dir, 'test_nc' + suf)
        else:
            train_path = os.path.join(save_dir, 'train' + suf)
            valid_path = os.path.join(save_dir, 'valid' + suf)
            test_path = os.path.join(save_dir, 'test' + suf)
        np.save(train_path, train_traces)
        np.save(valid_path, valid_traces)
        np.save(test_path, test_traces)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='mnist')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='leNet_1')
    parser.add_argument('-path1', help='directory where models and datasets are stored', type=str,default='/media/data1/DeepSuite')
    parser.add_argument('-path0', help='directory where models and datasets are stored', type=str,default='/media/data0/DeepSuite')
    parser.add_argument('-save_path', type=str, default='/media/data1/DeepSuite')
    parser.add_argument('-pool_type', help='pool using training or testing data', type=str, default='test')
    parser.add_argument('-fitness', help='the way of calculating fitness', type=str, default='nc')
    parser.add_argument('-cal_acc', type=bool, default=False)

    args = parser.parse_args()
    print(args)
    main()
