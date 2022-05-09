import os
import numpy as np
import pandas as pd
import scipy.stats as stats

def get_fitness_para(fitness):
    if fitness == 'nc':
        para = ['0.75']
    elif fitness == 'kmnc':
        para = ['10']
    elif fitness == 'snac' or fitness == 'nbc':
        para = ['NONE']
    elif fitness == 'tknc':
        para = ['1']
    elif fitness == 'lsa' or fitness == 'dsa':
        para = ['10', '1000']
    elif fitness == 'idc':
        para = ['6', '8']
    return para


# choosed_column = 'jaccard'
# choosed_column = 'wrong_case_num'
choosed_column = 'wrong_class_num'

base_dir = '/media/data0/DeepSuite/correlation'
model_list = ['leNet_1', 'leNet_4', 'leNet_5', 'resnet20_cifar10', 'resnet50_cifar10', 'MobileNet']
fitness_list = ['nc', 'kmnc', 'snac', 'nbc', 'tknc', 'dsa', 'lsa', 'idc']
wrong_rate = '0.1'
columns = ['model', 'choosed num']
for fitness in fitness_list:
    para_list = get_fitness_para(fitness)
    for para in para_list:
        columns.append(fitness+' '+para+'_corr')
        columns.append(fitness+' '+para+'_p')

result_array = []
for model in model_list:
    for sample_num in range(50, 1000, 50):
        sample_num_result = [model, sample_num]
        for fitness in fitness_list:
            fitness_dir = os.path.join(base_dir, model, fitness)
            para_list = get_fitness_para(fitness)
            for para in para_list:
                file_path = os.path.join(fitness_dir, 'parodam_' + para + '_' + wrong_rate + '_' + str(sample_num) + '.csv')
                para_file = pd.read_csv(file_path)
                fitness = para_file['Fitness'].values
                if choosed_column == 'jaccard':
                    choosed_values = para_file['Jaccard'].values
                elif choosed_column == 'wrong_case_num':
                    choosed_values = para_file['No. Wrong Cases'].values
                elif choosed_column == 'wrong_class_num':
                    choosed_values = para_file['No. Wrong Classes'].values
                choosed_values = np.nan_to_num(choosed_values)
                r, p = stats.pearsonr(fitness, choosed_values)
                sample_num_result.append(r)
                sample_num_result.append(p)
        sample_num_result = np.array(sample_num_result)
        result_array.append(sample_num_result)
result_array = np.vstack(result_array)
result = pd.DataFrame(data=result_array, columns=columns)
result.to_csv('RQ2_' + choosed_column + '.csv')

