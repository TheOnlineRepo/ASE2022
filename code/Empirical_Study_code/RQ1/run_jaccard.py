import os


def get_param(fitness):
    if fitness == 'nc':
        param_list = ['0.1', '0.25', '0.5', '0.75', '0.9']
        para_name = ' -k_nc '
    elif fitness == 'kmnc':
        param_list = ['10', '50', '100', '200', '500', '1000', '10000']
        para_name = ' -k_kmnc '
    elif fitness == 'idc_all_layer':
        param_list = ['4', '6', '8', '10', '12']
        para_name = ' -idc_n '
    elif fitness == 'tknc':
        param_list = ['1', '2', '3', '4', '5']
        para_name = ' -k_tknc '
    else:
        param_list = ['10', '50', '100', '200', '500', '1000']
        para_name = ' -sa_n '
    return param_list, para_name


python = '/home/zhiyu/anaconda3/envs/GPU/bin/python3.6'
code = '/home/zhiyu/DeepSuite/adversarial/empirical_study/RQ1/jaccard.py'
base_dir = '/media/data0/DeepSuite/jaccard'
dataset_list = ['mnist', 'cifar10']
model_list = [['leNet_1', 'leNet_4', 'leNet_5'], ['resnet20_cifar10', 'resnet50_cifar10', 'MobileNet']]
fitness_list = ['nc', 'kmnc', 'idc_all_layer', 'tknc', 'lsa', 'dsa']

for t, fitness in enumerate(fitness_list):
    fitness_param_list, param_name = get_param(fitness)
    for i, dataset in enumerate(dataset_list):
        dataset_model_list = model_list[i]
        for j, model in enumerate(dataset_model_list):
            for param in fitness_param_list:
                param_str = ' -dataset ' + dataset + ' -model ' + model + ' -fitness ' + fitness + param_name + param
                run = python + ' ' + code + param_str
                print(run)
                os.system(run)
