import os
import numpy as np
import pandas as pd


def cal_sa(path):
    sa = np.load(path)
    sa[np.where(sa == np.inf)] = 0
    sa[np.where(sa > 1e10)] = 0
    sa[np.where(sa < -1e10)] = 0
    sa[np.where(sa == -np.inf)] = 0
    return np.nanmean(sa)


base_dir = '/media/data0/DeepSuite/RQ5/RQ5_2'
# dataset = 'SVHN'
dataset = 'cifar100'
# ood_dataset = 'cifar10'
ood_dataset = 'SVHN'
# model_list = ['vgg19', 'WRN', 'resnet34']
model_list = ['vgg13', 'vgg16', 'vgg19']
# model_list = ['leNet_1', 'leNet_4', 'leNet_5']
# model_list = ['resnet20_cifar10', 'resnet50_cifar10', 'MobileNet']
fitness_list = ['lsa', 'dsa']
result = []
for j, model in enumerate(model_list):
    for i, fitness in enumerate(fitness_list):
        result_dir = os.path.join(base_dir, dataset, model, ood_dataset)
        ood_path = os.path.join(result_dir, fitness + '.ood.npy')
        test_path = os.path.join(result_dir, fitness + '.test.npy')
        try:
            ood_sa = cal_sa(ood_path)
            test_sa = cal_sa(test_path)
            print(model, fitness, ood_sa, test_sa)
        except:
            a = 0
