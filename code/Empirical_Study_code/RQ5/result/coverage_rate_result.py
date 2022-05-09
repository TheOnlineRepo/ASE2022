import os
import numpy as np
import pandas as pd
base_dir = '/media/data0/DeepSuite/RQ5/RQ5_2'


def get_fitness_neuron(fitness, dataset_id, model_id):
    neuron_num_list = [[52, 148, 268], [2570, 7274, 34893]]
    neu_num = neuron_num_list[dataset_id][model_id]
    if fitness == 'kmnc':
        neu_num *= 10
    elif fitness == 'nbc':
        neu_num *= 2
    elif fitness == 'idc':
        neu_num = 12
    return neu_num


dataset_list = [['FashionMNIST', 'SVHN', 'test'], ['SVHN', 'SUN', 'test']]
model_list = [['leNet_1', 'leNet_4', 'leNet_5'], ['resnet20_cifar10', 'resnet50_cifar10', 'MobileNet']]
# fitness_list = ['nc', 'kmnc', 'tknc', 'snac', 'nbc']
fitness_list = ['idc']
result = []
for dataset in range(2):
    for i, model in enumerate(model_list[dataset]):
        for ood_dataset in dataset_list[dataset]:
            for fitness in fitness_list:
                neu_num = get_fitness_neuron(fitness, dataset, i)
                model_dir = os.path.join(base_dir, model)
                if ood_dataset == 'test':
                    fitness_path = os.path.join(model_dir, 'SVHN', fitness+'.result.csv')
                    data = pd.read_csv(fitness_path, index_col=0, header=None)
                    coverage_rate = data.loc['test coverage'].values[0] / neu_num
                    # coverage_rate = data.loc['test activable samples'].values[0]
                else:
                    fitness_path = os.path.join(model_dir, ood_dataset, fitness+'.result.csv')
                    data = pd.read_csv(fitness_path, index_col=0, header=None)
                    coverage_rate = data.loc['ood coverage'].values[0] / neu_num
                    # coverage_rate = data.loc['ood activable samples'].values[0]
                print(ood_dataset, fitness, model, '%.2f' % (coverage_rate*100))
                # print(ood_dataset, fitness, model, '%.2f' % (coverage_rate / 100))
                # print(ood_dataset, fitness, model, '%.2f' % (coverage_rate * 100))
#             fitness_result = np.sum(fitness_result, axis=0)
#             result.append(fitness_result)
# result = np.vstack(result)
# result[:, 5] = result[:, 5] / 30000
# result[:, 6] = result[:, 6] / 30000
#
# result = pd.DataFrame(data=result)
# result.to_csv('./result.csv')
# a = 0
# print(result)
