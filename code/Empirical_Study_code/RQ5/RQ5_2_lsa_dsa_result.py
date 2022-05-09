import os
import numpy as np
import pandas as pd
dir = '/media/data1/DeepSuite/RQ5/RQ5_2/'
model_list = ['MobileNet', 'resnet50_cifar10', 'resnet20_cifar10']
fitness_list = ['lsa', 'dsa']
result = []
for fitness in fitness_list:
    ood_sum = 0
    test_sum = 0
    for model in model_list:
        model_dir = os.path.join(dir, model, 'SVHN')
        ood_file_path = os.path.join(model_dir, fitness+'.ood.npy')
        test_file_path = os.path.join(model_dir, fitness+'.test.npy')
        ood_sa = np.load(ood_file_path)
        test_sa = np.load(test_file_path)

        ood_sa[np.where(ood_sa == np.inf)] = 0
        test_sa[np.where(test_sa == np.inf)] = 0
        ood_sa[np.where(ood_sa == -np.inf)] = 0
        test_sa[np.where(test_sa == -np.inf)] = 0

        ood_sa = np.nan_to_num(ood_sa)
        test_sa = np.nan_to_num(test_sa)

        ood_mean = np.mean(ood_sa)
        ood_std = np.std(ood_sa)
        test_mean = np.mean(test_sa)
        test_std = np.std(test_sa)
        print(model, fitness, "ood | mean:", str(ood_mean), "| std:", str(ood_std))
        print(model, fitness, "test | mean:", str(test_mean), "| std:", str(test_std))
        result.append([ood_mean, test_mean, ood_std, test_std])
        ood_sum += ood_mean
        test_sum += test_mean
    print(fitness, ood_sum, test_sum)
result = np.array(result)
result = pd.DataFrame(data=result)
result.to_csv('./result.csv')