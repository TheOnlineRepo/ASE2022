import os
import numpy as np
import pandas as pd
ood_dataset = 'FashionMNIST'
base_dir = '/media/data0/DeepSuite/RQ5/RQ5_2'
model_list = ['leNet_1', 'leNet_4', 'leNet_5']
# model_list = ['resnet20_cifar10', 'resnet50_cifar10', 'MobileNet']
fitness_list = ['nc', 'kmnc', 'tknc', 'snac', 'nbc']
result = []
for fitness in fitness_list:
    fitness_result = []
    for model in model_list:
        model_dir = os.path.join(base_dir, model)
        fitness_path = os.path.join(model_dir, ood_dataset, fitness+'.result.csv')
        data = pd.read_csv(fitness_path, index_col=0, header=None)
        fitness_result.append(data.values[:, 0])
    fitness_result = np.sum(fitness_result, axis=0)
    result.append(fitness_result)
result = np.vstack(result)
result[:, 5] = result[:, 5] / 30000
result[:, 6] = result[:, 6] / 30000

result = pd.DataFrame(data=result)
result.to_csv('./result.csv')
# a = 0
# print(result)
