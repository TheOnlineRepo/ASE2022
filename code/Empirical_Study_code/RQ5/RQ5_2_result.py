import numpy as np
import os
import pandas as pd
base_dir = '/media/data1/DeepSuite/RQ5/RQ5_2'
model_list = ['MobileNet']
ood_list = ['SVHN', 'SUN']
# model_list = ['leNet_1', 'leNet_4', 'leNet_5']
# ood_list = ['FashionMNIST', 'SVHN']
fitness_list = ['nc', 'kmnc', 'nbc', 'snac', 'tknc', 'idc']
for ood_dataset in ood_list:
    print("---------------", ood_dataset, "--------------")
    for fitness in fitness_list:
        ood_fitness_coverage = 0
        test_fitness_coverage = 0
        for model in model_list:
            fitness_path = os.path.join(base_dir, model, ood_dataset, fitness+'.result.csv')
            fitness_data = pd.read_csv(fitness_path, sep=',', index_col=0, header=None)
            ood_fitness_coverage += int(fitness_data.loc['ood coverage'])
            test_fitness_coverage += int(fitness_data.loc['test coverage'])
        print(fitness, ood_fitness_coverage, test_fitness_coverage)
