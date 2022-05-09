# encoding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
sys.path.append('/home/zhiyu/DeepSuite/adversarial')
from adv_function import *
import csv
import random
import argparse
import time
from keras.applications.vgg16 import decode_predictions
from keras.utils import to_categorical

def init():
    num_error = int(args.pool_num * args.wrong_rate)
    num_correct = args.pool_num - num_error

    if args.fitness == 'nc':
        trace_path = os.path.join(args.path, 'trace', 'test_trace', args.dataset, args.model + '_nc.npy')
    else:
        trace_path = os.path.join(args.path, 'trace', 'test_trace', args.dataset, args.model + '.npy')
    trace = np.load(trace_path)

    if args.fitness == 'nc':
        param = [args.k_nc]
    elif args.fitness == 'tknc' or args.fitness == 'tknp':
        param = [args.k_tknc]
    elif args.fitness == 'kmnc':
        param = [args.k_kmnc]
    elif args.fitness == 'lsa' or args.fitness == 'dsa':
        param = ['10', '1000']
    elif args.fitness == 'idc':
        param = [args.idc_n]
    else:
        param = ['NONE']

    return num_error, num_correct, trace, param


def filter_test_case(x_test, y_test, model):
    global predictions
    if args.dataset == 'mnist':
        predictions = model.predict_classes(x_test, verbose=1)
    elif args.dataset == 'cifar10':
        y_pred = model.predict(x_test, verbose=1)
        predictions = np.argmax(y_pred, axis=1)
    elif args.dataset == 'ImageNet':
        y_pred = model.predict(x_test)
        y_pred = decode_predictions(y_pred, top=1)
        predictions = [y_pred[i][0][0] for i in range(len(y_pred))]
    elif args.dataset == 'cifar100' or args.dataset == 'SVHN':
        y_pred = model.predict(x_test, verbose=1)
        # print(y_pred)
        predictions = np.argmax(y_pred, axis=1)
        # print(predictions)
    filter_check = np.array([predictions[i] == y_test[i] for i in range(len(y_test))])
    return filter_check


def create_record_file(param, num_samples):
    record_model_dir = os.path.join(args.save_path, 'correlation', args.model)
    if not os.path.exists(record_model_dir):
        os.mkdir(record_model_dir)
    record_dir = os.path.join(record_model_dir, args.fitness)
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    record_file = 'parodam_' + str(param) + "_" + str(args.wrong_rate) + "_" + str(num_samples) + ".csv"

    record_path = os.path.join(record_dir, record_file)
    if os.path.exists(record_path):
        os.remove(record_path)
    content = ['No.', 'Fitness', 'No. Wrong Cases', 'No. Wrong Classes', 'Jaccard', 'Runtime']
    with open(record_path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(content)
    return record_path


def trace2fitness(model, trace, traces_low, traces_high, x_train, y_train, x_test, y_test):
    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        skip = True
    else:
        skip = False
    if args.dataset == 'cifar100':
        num_class = 100
    else:
        num_class = 10
    fitness = None
    if args.fitness == 'nc':
        fitness = nc(trace, args.k_nc)
    elif args.fitness == 'kmnc':
        fitness = kmnc(trace, args.k_kmnc, traces_low, traces_high)
    elif args.fitness == 'nbc':
        fitness = nbc(trace, traces_low, traces_high)
    elif args.fitness == 'snac':
        fitness = snac(trace, traces_high)
    elif args.fitness == 'tknc':
        fitness = tknc(trace, args.k_tknc, model, skip)
    elif args.fitness == 'tknp':
        fitness = tknc(trace, args.k_tknc, model, skip)
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
    elif args.fitness == 'idc':
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            only_last_layer = True
        else:
            only_last_layer = False
        fitness = idc(model, args.dataset, args.model, x_train, y_train, x_test, y_test, args.idc_n, only_last_layer=only_last_layer)
    else:
        test_sa_dir = os.path.join(args.path, 'trace', 'test_sa', args.fitness, args.dataset)
        test_sa_path = os.path.join(test_sa_dir, args.model + '.npy')
        if os.path.exists(test_sa_path):
            fitness = np.load(test_sa_path)
        else:
            train_trace = np.load(os.path.join(args.path, 'trace', 'train_trace', args.dataset, args.model + '.npy'))
            fitness = []
            layer_names = get_layers(model, skip=True)
            for layer_num in range(len(layer_names)):
                start, end = get_layer_bound(model, -1, layer_num)
                if args.fitness == 'lsa':
                    sa = lsa(trace, y_test, train_trace, y_train, args.sa_n, True, start, end, num_classes=num_class)
                elif args.fitness == 'dsa':
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


def cal_jaccard_score(fitness):
    fitness = fitness.astype(np.bool)
    set_num = fitness.shape[0]
    jaccard_list = []
    for i in range(set_num):
        for j in range(i + 1, set_num):
            jaccard = 1 - float(np.sum(fitness[i] & fitness[j])) / np.sum(fitness[i] | fitness[j])
            if not np.isnan(jaccard):
                jaccard_list.append(jaccard)
    return np.mean(jaccard_list)


def main():
    # print(tf.test.is_gpu_available())
    num_error, num_correct, trace, params = init()
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, args.model, args.dataset)
    filter_check = filter_test_case(x_test, y_test, model)

    # layer_names = get_layers(model)
    traces_low, traces_high = get_trace_boundary(args.path, args.dataset, args.model)

    fitness = trace2fitness(model, trace, traces_low, traces_high, x_train, y_train, x_test, y_test)
    # m_fitness = fitness
    m_fitness = np.delete(fitness, np.isnan(np.sum(fitness, axis=1)), axis=0)
    for param in params:
        if args.sa_layer == -1 and (args.fitness == 'lsa' or args.fitness == 'dsa'):
            fitness_set = []
            for i in range(m_fitness.shape[0]):
                cov = get_sc(
                    np.amin(m_fitness[i]), m_fitness[i][np.argsort(-m_fitness[i])[5]], int(param), m_fitness[i]
                )
                cov[np.where(cov > int(param) - 1)] = int(param) - 1
                layer_fitness = to_categorical(cov)
                fitness_set.append(layer_fitness)
            fitness = np.concatenate(fitness_set, axis=1)
        for num_samples in range(50, 1050, 50):
            record_path = create_record_file(param, num_samples)

            for m_iter in range(args.pool_iter):
                print("-----pool iter:", m_iter, "-----")
                pool = []
                sub_idx = random.sample(list(np.where(filter_check == 0)[0]), num_error)
                pool.extend(sub_idx)
                sub_idx = random.sample(list(np.where(filter_check == 1)[0]), num_correct)
                pool.extend(sub_idx)
                print('wrong samples: ', num_error)
                print('correct samples: ', num_correct)

                pool = np.array(pool)
                print('pool.shape: ', len(pool))

                for j in range(args.iterations):
                    print("check iter:", j)
                    # create a subset of pool
                    sub_idx = np.array(random.sample(list(pool), num_samples))

                    check_set_fitness = fitness[sub_idx]

                    check_set_fitness = np.sum(np.max(check_set_fitness, axis=0))
                    wrong_cases_id = np.where(filter_check[sub_idx] == 0)[0]
                    wrong_cases_id = sub_idx[wrong_cases_id]
                    check_set_wrong_case_num = len(wrong_cases_id)
                    check_set_wrong_class_num = len(set(y_test[wrong_cases_id]))

                    wrong_case_jaccard_score = cal_jaccard_score(fitness[wrong_cases_id])

                    print('fit: ', check_set_fitness)
                    print('No. wrong: ', check_set_wrong_case_num)
                    print('No. wrong classes: ', check_set_wrong_class_num)
                    print('Jaccard score: ', wrong_case_jaccard_score)
                    with open(record_path, 'a+') as f:
                        runtime = time.clock() - start_time
                        writer = csv.writer(f)
                        content = [str(m_iter) + "_" + str(j), check_set_fitness, check_set_wrong_case_num,
                                   check_set_wrong_class_num, wrong_case_jaccard_score, runtime]
                        writer.writerow(content)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='mnist')
    parser.add_argument('-model', type=str, default='leNet_1')
    parser.add_argument('-fitness', help='the way of calculating fitness', type=str, default='nc')
    parser.add_argument('-path', help='directory where models and datasets are stored', type=str,
                        default='/media/data0/DeepSuite/')
    parser.add_argument('-save_path', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-pool_type', help='pool using training or testing data', type=str, default='test')
    parser.add_argument('-k_kmnc', help='the parameter k for kmnc', type=int, default=10)
    parser.add_argument('-k_tknc', help='the parameter k for tknc', type=int, default=1)
    parser.add_argument('-k_nc', help='the threshold for nc', type=float, default=0.75)
    parser.add_argument('-sa_layer', help='layer_num for sa, 0:1st layer; 1:inner layer ; 2:last layer; -1:all layer', type=int, default=-1)
    parser.add_argument('-sa_n', type=int, default=1000)
    parser.add_argument('-idc_n', type=int, default=6)
    parser.add_argument('-idc_layer', type=str, default='all')
    parser.add_argument('-pool_iter', type=int, default=10)
    parser.add_argument('-pool_num', help='the number of samples', type=int, default=1000)
    parser.add_argument('-wrong_rate', type=float, default=0.1)
    parser.add_argument('-num_samples', help='the number of samples', type=int, default=100)
    parser.add_argument('-iterations', help='the number of interations', type=int, default=30)

    args = parser.parse_args()
    print(args)
    main()
