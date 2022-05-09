# encoding=utf-8
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data_dir = '/media/data0/DeepSuite/jaccard'
    # model_list = ['resnet50_cifar10', 'MobileNet', 'resnet20_cifar10']
    model_list = ['vgg19_SVHN_backup_lly']
    for model_name in model_list:
        if model_name == 'log':
            continue
        model_dir = os.path.join(data_dir, model_name)

        fitness_path = os.path.join(model_dir, 'lsa')
        for file_name in os.listdir(fitness_path):
            file_path = os.path.join(fitness_path, file_name)
            data = pd.read_csv(file_path)
            jaccard = data['Jaccard'].values
            if jaccard.shape[0] == 0:
                print(file_path, 'mem err')
            else:
                print(file_path, np.nanmean(jaccard))

