import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import sys
sys.path.append('/home/zhiyu/DeepSuite/adversarial')
from adv_function import *
import argparse
from lightgbm import LGBMClassifier
from keras.utils import to_categorical


def trace2fitness(fitness_type, model, trace, traces_low, traces_high, x_train, y_train, x_test, y_test):
    fitness = None
    if fitness_type == 'None':
        fitness = trace
    elif fitness_type == 'nc':
        fitness = nc(trace, args.k_nc)
    elif fitness_type == 'kmnc':
        fitness = kmnc(trace, args.k_kmnc, traces_low, traces_high)
    elif fitness_type == 'nbc':
        fitness = nbc(trace, traces_low, traces_high)
    elif fitness_type == 'snac':
        fitness = snac(trace, traces_high)
    elif fitness_type == 'tknc':
        fitness = tknc(trace, args.k_tknc, model)
    elif fitness_type == 'tknp':
        fitness = tknc(trace, args.k_tknc, model)
        type_list = np.zeros(fitness.shape[0]).astype(int)
        index = 0
        for i in range(fitness.shape[0]):
            type_list[i] = index
            for j in range(i):
                if (fitness[i] == fitness[j]).all():
                    type_list[i] = type_list[j]
                    break
            if type_list[i] == index:
                index += 1
        fitness = to_categorical(type_list)
    elif fitness_type == 'idc':
        if args.dataset == 'cifar10':
            only_last_layer = True
        else:
            only_last_layer = False
        fitness = idc(model, args.model, x_train, y_train, x_test, y_test, args.idc_n, only_last_layer=only_last_layer)
    else:
        train_trace = np.load(os.path.join(args.path1, 'trace', 'train_trace', args.model + '.npy'))
        trainable_layers = get_trainable_layers(model)
        fitness = []
        layer_names = get_layers(model)
        for layer_num in range(len(layer_names)):
            if layer_num not in trainable_layers:
                continue
            start, end = get_layer_bound(model, args.sa_layer, layer_num)
            if fitness_type == 'lsa':
                sa = lsa(trace, y_test, train_trace, y_train, args.sa_n, 1, start, end)

            elif fitness_type == 'dsa':
                sa = dsa(trace, y_test, train_trace, y_train, args.sa_n, start, end)
            fitness.append(sa)
        m_fitness = np.stack(fitness)

        param = 10
        fitness_set = []
        for i in range(m_fitness.shape[0]):
            cov = get_sc(
                np.amin(m_fitness[i]), m_fitness[i][np.argsort(-m_fitness[i])[5]], int(param), m_fitness[i]
            )
            cov[np.where(cov > int(param) - 1)] = int(param) - 1
            layer_fitness = to_categorical(cov)
            fitness_set.append(layer_fitness)
        fitness = np.concatenate(fitness_set, axis=1)
    return fitness


def load_traces(path0, dataset, model_name, adv_method, fitness):
    save_dir = os.path.join(path0, 'adv_trace', dataset, model_name, adv_method)
    if fitness == 'nc':
        suf = '_nc.npy'
    else:
        suf = '.npy'
    train_path = os.path.join(save_dir, 'train' + suf)
    valid_path = os.path.join(save_dir, 'valid' + suf)
    test_path = os.path.join(save_dir, 'test' + suf)
    train_traces = np.load(train_path)
    valid_traces = np.load(valid_path)
    test_traces = np.load(test_path)
    return train_traces, valid_traces, test_traces


def load_ori_traces(path0, path1, dataset, model_name, fitness):
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(path1, model_name, dataset)
    if fitness == 'nc':
        suf = '_nc.npy'
    else:
        suf = '.npy'
    save_dir = os.path.join(path1, 'trace')

    train_traces_path = os.path.join(save_dir, 'train_trace', model_name + suf)
    test_traces_path = os.path.join(save_dir, 'test_trace', model_name + suf)
    train_traces = np.load(train_traces_path)
    test_traces = np.load(test_traces_path)
    return train_traces, test_traces, y_train, y_test


def main():
    fitness_list = ['idc', 'lsa', 'dsa', 'None', 'nc', 'kmnc', 'tknc', 'snac', 'nbc']
    # fitness_list = ['None', 'nc',  'dsa',  'lsa', 'tknc']
    # adv_method_list = ['CW_Linf', 'PGD', 'FGSM', 'BIM', 'DF']
    adv_method_list = ['FGSM', 'DF', 'CW_Linf']
    # adv_method_list = ['DF']
    result_list = []
    save_dir = os.path.join(args.path0, 'RQ4', args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, args.model + '.csv')
    already_done = args.done
    for adv_method in adv_method_list:
        if args.dataset == 'cifar10':
            if adv_method == 'FGSM':
                adv_eps = 0.01
            elif adv_method == 'DF':
                adv_eps = 0.1
            else:
                adv_eps = 0.3
        else:
            if adv_method in ['PGD', 'FGSM', 'BIM', 'CW_Linf']:
                adv_eps = 0.3
            else:
                adv_eps = 0.1
        for fitness in fitness_list:
            if already_done > 0:
                already_done -=1
                continue
            print("adv method:", adv_method)
            print("fitness type:", fitness)
            adv_train_x, adv_train_y, adv_valid_x, adv_valid_y, adv_test_x, adv_test_y = load_adv_sample(args.path0, args.dataset, args.model, adv_method,adv_eps)
            adv_train_traces, adv_valid_traces, adv_test_traces = load_traces(args.path0, args.dataset, args.model, adv_method, fitness)
            adv_train_traces = np.concatenate((adv_train_traces, adv_valid_traces))
            adv_train_y = np.concatenate((adv_train_y, adv_valid_y))
            adv_train_x = np.concatenate((adv_train_x, adv_valid_x))

            ori_train_traces, ori_test_traces, ori_train_y, ori_test_y = load_ori_traces(args.path0, args.path1, args.dataset, args.model, fitness)
            model, ori_train_x, _, ori_test_x, _ = load_model_and_testcase(args.path1, args.model, args.dataset)

            clf_train_x = np.concatenate((adv_train_x, ori_train_x))
            clf_test_x = np.concatenate((adv_test_x, ori_test_x))
            clf_train_traces = np.concatenate((adv_train_traces, ori_train_traces))
            clf_test_traces = np.concatenate((adv_test_traces, ori_test_traces))
            clf_train_y = np.concatenate((adv_train_y, ori_train_y))
            clf_test_y = np.concatenate((adv_test_y, ori_test_y))
            clf_train_label = np.concatenate((np.ones(adv_train_y.shape[0]), np.zeros(ori_train_y.shape[0])))
            clf_test_label = np.concatenate((np.ones(adv_test_y.shape[0]), np.zeros(ori_test_y.shape[0])))
            traces_low, traces_high = get_trace_boundary(args.path1, args.model)

            clf_traces = np.concatenate((clf_train_traces, clf_test_traces))
            clf_y = np.concatenate((clf_train_y, clf_test_y))
            clf_x = np.concatenate((clf_train_x, clf_test_x))
            clf_fitness = trace2fitness(fitness, model, clf_traces, traces_low,traces_high, ori_train_x, ori_train_y, clf_x, clf_y)
            clf_train_fitness = clf_fitness[:clf_train_y.shape[0]]
            clf_test_fitness = clf_fitness[clf_train_y.shape[0]:]
            del adv_train_traces, adv_train_x, adv_train_y, ori_train_x, ori_train_traces, ori_train_y, clf_train_x, clf_train_traces, clf_test_x, clf_test_traces
            gc.collect()
            lgb = LGBMClassifier
            opti_lgb_param, lgb_acc, opti_model = param_adj(lgb, lgb_adj_list, lgb_params, clf_train_fitness, clf_train_label,clf_test_fitness, clf_test_label, None)
            test_pred_y = opti_model.predict(clf_test_fitness)
            test_acc = np.sum(test_pred_y == clf_test_label) / len(clf_test_label)
            print("test_acc:", test_acc)
            with open(save_path, 'a+') as f:
                f.write(adv_method+","+fitness+","+str(test_acc)+"\n")
    #         result.append(test_acc)
    #     result_list.append(result)
    # result_list = np.array(result_list)
    # np.save(save_path, result_list)


if __name__ == '__main__':
    start_time = time.asctime(time.localtime(time.time()))
    print("start time :", start_time)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='mnist')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='leNet_1')
    parser.add_argument('-path0', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-path1', type=str, default='/media/data1/DeepSuite')
    parser.add_argument('-k_kmnc', help='the parameter k for kmnc', type=int, default=10)
    parser.add_argument('-k_tknc', help='the parameter k for tknc', type=int, default=3)
    parser.add_argument('-k_nc', help='the threshold for nc', type=float, default=0.75)
    parser.add_argument('-sa_layer', help='layer_num for sa, 0:1st layer; 1:inner layer ; 2:last layer; -1:all layer', type=int, default=-1)
    parser.add_argument('-sa_n', type=int, default=1000)
    parser.add_argument('-idc_n', type=int, default=6)
    parser.add_argument('-done', type=int, default=0)
    args = parser.parse_args()
    print(args)
    main()
