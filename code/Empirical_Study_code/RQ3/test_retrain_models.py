import os
import numpy as np
import pandas as pd
from adv_function import *
import time
import argparse
from keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

dataset_list = ['cifar10', 'mnist']

adv_method = 'FGSM'
adv_eps_list = [0.01, 0.3]
fitness_list = ['nc', 'kmnc', 'tknc', 'snac', 'nbc', 'None']
retrain_model_path = '/media/data0/DeepSuite/adv_retrain_models'


def main():
    for i, dataset in enumerate(dataset_list):
        # print("##########", dataset, "##########")
        if dataset == 'mnist':
            model_list = ['leNet_1', 'leNet_4', 'leNet_5']
        else:
            model_list = ['resnet20_cifar10', 'resnet50_cifar10', 'MobileNet']
        adv_eps = adv_eps_list[i]
        for model_name in model_list:
            # print("model name:", model_name)
            model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, model_name, dataset, verbose=False)
            adv_train_x, adv_train_y, adv_valid_x, adv_valid_y, adv_test_x, adv_test_y = load_adv_sample(args.path, dataset, model_name, adv_method, adv_eps)

            for fitness in fitness_list:
                fitness_model_path = os.path.join(retrain_model_path, model_name, fitness+'.h5')
                if not os.path.exists(fitness_model_path):
                    continue
                fitness_model = load_model(fitness_model_path)

                pred_test = np.argmax(fitness_model.predict(x_test), axis=1)
                ori_test_acc = np.sum(pred_test == y_test) / y_test.shape[0]

                adv_test_pred = np.argmax(fitness_model.predict(adv_test_x), axis=1)
                adv_test_acc = np.sum(adv_test_pred == adv_test_y) / adv_test_y.shape[0]
                print(model_name, fitness, ori_test_acc, adv_test_acc)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', help='directory where models and datasets are stored', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-test_size', type=float, default=0.1)
    parser.add_argument('-attack_eps_id', type=int, default=0)

    args = parser.parse_args()
    print(args)
    main()