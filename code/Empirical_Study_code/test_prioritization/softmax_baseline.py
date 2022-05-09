import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
sys.path.append('/home/zhiyu/DeepSuite/adversarial')
import pandas as pd
import numpy as np
from adv_function import load_model_and_testcase, get_trace_boundary, get_layers, get_layer_bound
from adv_function import nc, kmnc, tknc, snac, nbc, lsa, dsa, get_sc, idc
import argparse
import time


def load_trace(path, dataset, model_name, fitness):
    if fitness == 'nc':
        trace_path = os.path.join(path, 'trace', 'test_trace', dataset, model_name + '_nc.npy')
    else:
        trace_path = os.path.join(path, 'trace', 'test_trace', dataset, model_name + '.npy')
    trace = np.load(trace_path)
    return trace


def trace2fitness(fitness_type, model, trace, traces_low, traces_high, x_train, y_train, x_test, y_test):
    if args.dataset == 'cifar100':
        num_class = 100
    else:
        num_class = 10
    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        skip = True
    else:
        skip = False
    if fitness_type == 'nc':
        fitness = nc(trace, args.k_nc)
    elif fitness_type == 'kmnc':
        fitness = kmnc(trace, args.k_kmnc, traces_low, traces_high)
    elif fitness_type == 'nbc':
        fitness = nbc(trace, traces_low, traces_high)
    elif fitness_type == 'snac':
        fitness = snac(trace, traces_high)
    elif fitness_type == 'tknc':
        fitness = tknc(trace, args.k_tknc, model, skip)
    elif fitness_type == 'idc':
        if args.dataset != 'mnist':
            only_last_layer = True
        else:
            only_last_layer = False
        fitness = idc(model, args.dataset, args.model, x_train, y_train, x_test, y_test, args.idc_n, only_last_layer=only_last_layer)
    else:
        test_sa_dir = os.path.join(args.path, 'trace', 'test_sa', fitness_type, args.dataset)
        test_sa_path = os.path.join(test_sa_dir, args.model + '.npy')
        if os.path.exists(test_sa_path):
            fitness = np.load(test_sa_path)
        else:
            train_trace = np.load(os.path.join(args.path, 'trace', 'train_trace', args.dataset, args.model + '.npy'))
            fitness = []
            layer_names = get_layers(model, skip=True)
            for layer_num in range(len(layer_names)):
                start, end = get_layer_bound(model, -1, layer_num)
                if fitness_type == 'lsa':
                    sa = lsa(trace, y_test, train_trace, y_train, args.sa_n, True, start, end, num_classes=num_class)
                elif fitness_type == 'dsa':
                    sa = dsa(trace, y_test, train_trace, y_train, args.sa_n, start, end)
                if sa is not None:
                    fitness.append(sa)
                    print("SA layer", layer_names[layer_num])
                else:
                    print("None layer", layer_names[layer_num])
            fitness = np.stack(fitness)
            if not os.path.exists(test_sa_dir):
                os.makedirs(test_sa_dir)
            np.save(test_sa_path, fitness)
    return fitness


def greedy_rank(fitness):
    subset = []
    lst = list(range(fitness.shape[0]))
    CHOICE = np.random.choice(range(fitness.shape[0]))
    initial = CHOICE
    lst.remove(initial)
    subset.append(initial)
    covered_neuron, max_covered_neuron = fitness[initial], fitness[initial]
    max_coverage = np.sum(np.max(fitness, axis=0))
    while np.sum(covered_neuron) < max_coverage:
        index_covered_neuron = np.sum(fitness[lst] + covered_neuron, axis=1)
        max_covered_index = lst[np.argmax(index_covered_neuron)]
        covered_neuron = covered_neuron + fitness[max_covered_index]
        lst.remove(max_covered_index)
        subset.append(max_covered_index)
    lst_rank = np.argsort(-np.sum(fitness[lst], axis=1))
    test_rank = np.hstack((subset, np.array(lst)[lst_rank]))
    return test_rank


def apfd(right, sort):
    sum_all = np.sum(sort[right != 1])
    n = len(sort)
    m = np.sum(right == 0)
    return 1 - float(sum_all) / (n * m) + 1. / (2 * n)


def main():
    # function_list = ['idc', 'lsa', 'dsa', 'nc', 'kmnc', 'snac', 'nbc', 'tknc', 'softmax', 'deepgini']
    # function_list = ['nc', 'kmnc', 'snac', 'nbc', 'tknc', 'softmax', 'deepgini', 'dsa', 'lsa']
    function_list = ['idc', 'lsa', 'dsa', 'softmax', 'deepgini']
    # function_list = ['idc']
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, args.model, args.dataset)
    save_path = os.path.join(args.path, 'test_prioritization', args.model+'.csv')
    result = []

    traces_low, traces_high = get_trace_boundary(args.path, args.dataset, args.model)
    right = np.argmax(model.predict(x_test), axis=1) == y_test
    for function in function_list:
        trace = load_trace(args.path, args.dataset, args.model, function)
        if function == 'softmax':
            y_pred = model.predict(x_test)
            # higher is uncertain
            softmax_score = -1 * np.sum((np.log(y_pred) * y_pred), axis=1)
            test_rank = np.argsort(softmax_score)
        elif function == 'deepgini':
            y_pred = model.predict(x_test)
            # higher is uncertain
            deepgini_score = 1 - np.sum(y_pred ** 2, axis=1)
            test_rank = np.argsort(deepgini_score)
        elif function in ['dsa', 'lsa']:
            sa = trace2fitness(function, model, trace, traces_low, traces_high, x_train, y_train, x_test, y_test)
            sa = np.nan_to_num(sa)
            test_rank = np.argsort(sa)[-1]
        else:
            fitness = trace2fitness(function, model, trace, traces_low, traces_high, x_train, y_train, x_test, y_test)
            test_rank = greedy_rank(fitness)
        test_rank = test_rank[::-1]
        sort_list = np.zeros(x_test.shape[0], dtype='int')
        sort_list[test_rank] = [i + 1 for i in range(x_test.shape[0])]
        apfd_score = apfd(right, sort_list)
        print(function, apfd_score)
        result.append([function, apfd_score])
    result = pd.DataFrame(data=result, columns=['fitness', 'apfd'])
    result.to_csv(save_path, index=False)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='SVHN'
                                            '', choices=['mnist', 'cifar10', 'cifar100', 'SVHN'], type=str)
    parser.add_argument('-model', type=str, default='resnet34', choices=['leNet_1', 'leNet_4', 'leNet_5', 'resnet20_cifar10', 'resnet50_cifar10', 'MobileNet', 'vgg13', 'vgg16', 'vgg19', 'DenseNet121', 'WRN', 'GoogleNet'])
    parser.add_argument('-path', type=str, default='/media/data0/DeepSuite/')
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-pool_type', help='pool using training or testing data', type=str, default='test')
    parser.add_argument('-k_kmnc', help='the parameter k for kmnc', type=int, default=10)
    parser.add_argument('-k_tknc', help='the parameter k for tknc', type=int, default=1)
    parser.add_argument('-k_nc', help='the threshold for nc', type=float, default=0.75)
    parser.add_argument('-sa_n', type=int, default=10)
    parser.add_argument('-idc_n', type=int, default=6)
    parser.add_argument('-idc_layer', type=str, default='all')

    args = parser.parse_args()
    print(args)
    main()
