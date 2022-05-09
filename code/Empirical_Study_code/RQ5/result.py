import numpy as np
import pandas as pd
import os
file_dir = '/media/data0/DeepSuite/RQ5/'
model_list = ['leNet_1', 'leNet_4' ,'leNet_5']

for model in model_list:
    model_dir = os.path.join(file_dir, model, 'fashion')
    save_path = os.path.join('./'+model+'.csv')
    fitness_result_list = []
    for fitness in os.listdir(model_dir):
        if fitness == 'idc_all_layer':
            continue
        print(fitness)
        fitness_dir = os.path.join(model_dir, fitness)
        parm_result_list = []
        for parm in os.listdir(fitness_dir):
            para_path = os.path.join(fitness_dir, parm)
            data = pd.read_csv(para_path, sep=',').values.T
            Ori_fitness = np.mean(data[0])
            Ori_Ori_fitness = np.mean(data[3])
            Ori_Ood_fitness = np.mean(data[4])
            if Ori_fitness != 0:
                result = (Ori_Ood_fitness - Ori_Ori_fitness) / Ori_fitness
            else:
                result = 0
            parm_result_list.append(result)
        fitness_result_list.append(parm_result_list)
    fitness_result_list = np.array(fitness_result_list)
    model_data = pd.DataFrame(data=fitness_result_list)
    model_data.to_csv(save_path)
    a = 0