import os
import numpy as np
import pandas as pd

base_dir = '/media/data0/DeepSuite/RQ5/RQ5_2'
dataset = 'SVHN'
ood_dataset = 'cifar10'
model_list = ['vgg19', 'WRN', 'resnet34']
model_neu_num = [12644, 15226, 1866]
# model_list = ['leNet_1', 'leNet_4', 'leNet_5']
# model_list = ['resnet20_cifar10', 'resnet50_cifar10', 'MobileNet']
fitness_list = ['nc', 'kmnc', 'tknc', 'snac', 'nbc', 'idc']
neu_cot = [1, 10, 1, 1, 2, 10]
result = []
for j, model in enumerate(model_list):
    for i, fitness in enumerate(fitness_list):
        if fitness == 'idc':
            neu_all = 10
        else:
            neu_all = neu_cot[i] * model_neu_num[j]
        result_dir = os.path.join(base_dir, dataset, model, ood_dataset)
        fitness_path = os.path.join(result_dir, fitness+'.result.csv')
        try:
            data = pd.read_csv(fitness_path, index_col=0, header=None)
            ood_cov = int(data.loc['ood coverage']) / neu_all * 100
            test_cov = int(data.loc['test coverage']) / neu_all * 100
            ood_act = int(data.loc['ood activable samples']) / 100
            test_act = int(data.loc['test activable samples']) / 100
            print(model, fitness, '%.2f' % ood_cov, '%.2f' % ood_act, '%.2f' % test_cov, '%.2f' % test_act)
        except:
            a = 0
