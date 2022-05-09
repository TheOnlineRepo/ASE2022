# encoding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from functions import *
import csv
import random
import argparse
import time
from keras.applications.vgg16 import decode_predictions
import sys

def init():

    if args.fitness == 'nc':
        trace_path = os.path.join(args.path, 'trace', args.model + '_nc.npy')
    else:
        trace_path = os.path.join(args.path, 'trace', args.model + '.npy')
    trace = np.load(trace_path)

    fitness_cot = 1
    if args.fitness == 'nc':
        param = args.k_nc
    elif args.fitness == 'tknc':
        param = args.k_tknc
        fitness_cot = args.k_tknc
    elif args.fitness == 'kmnc':
        param = args.k_kmnc
        fitness_cot = args.k_kmnc
    else:
        if args.fitness == 'nbc':
            fitness_cot = 2
        param = 'NONE'

    num_samples = args.num_samples * fitness_cot
    return num_samples, trace, param


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
    filter_check = np.array([predictions[i] == y_test[i] for i in range(len(y_test))])
    return filter_check


def create_record_file(param):
    record_model_dir = os.path.join(args.save_path, 'no_pool_correlation', args.model)
    if not os.path.exists(record_model_dir):
        os.mkdir(record_model_dir)
    record_dir = os.path.join(record_model_dir, args.fitness)
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    record_file = 'parodam_' + str(param) + ".csv"
    record_path = os.path.join(record_dir, record_file)
    if os.path.exists(record_path):
        os.remove(record_path)
    content = ['No.', 'Fitness', 'No. Wrong Cases', 'No. Wrong Classes', 'Jaccard', 'Runtime']
    with open(record_path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(content)
    return record_path


def trace2fitness(model, trace, traces_low, traces_high):
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
        fitness = tknc(trace, args.k_tknc, model)
    return fitness


def cal_jaccard_score(fitness):
    fitness = fitness.astype(np.bool)
    set_num = fitness.shape[0]
    jaccard_list = []
    for i in range(set_num):
        for j in range(i + 1, set_num):
            jaccard = float(np.sum(fitness[i] & fitness[j])) / np.sum(fitness[i] | fitness[j])
            jaccard_list.append(jaccard)
    return np.mean(jaccard_list)


def main():
    # print(tf.test.is_gpu_available())
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, args.model, args.dataset)
    num_samples, trace, param = init()
    filter_check = filter_test_case(x_test, y_test, model)
    record_path = create_record_file(num_samples)

    # layer_names = get_layers(model)
    traces_low, traces_high = get_trace_boundary(args.path, args.model)
    # sys.exit()
    fitness = trace2fitness(model, trace, traces_low, traces_high)
    pool = np.array([i for i in range(x_test.shape[0])])
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
            content = [str(j), check_set_fitness, check_set_wrong_case_num,
                       check_set_wrong_class_num, wrong_case_jaccard_score, runtime]
            writer.writerow(content)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='cifar10')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='MobileNet')
    parser.add_argument('-path', help='directory where models and datasets are stored', type=str,
                        default='/media/data1/DeepSuite')
    parser.add_argument('-save_path', type=str, default='/media/data1/DeepSuite')
    parser.add_argument('-pool_type', help='pool using training or testing data', type=str, default='test')
    parser.add_argument('-fitness', help='the way of calculating fitness', type=str, default='kmnc')
    parser.add_argument('-k_kmnc', help='the parameter k for kmnc', type=int, default=10)
    parser.add_argument('-k_tknc', help='the parameter k for tknc', type=int, default=3)
    parser.add_argument('-k_nc', help='the threshold for nc', type=float, default=0.75)
    parser.add_argument('-num_samples', help='the number of samples', type=int, default=500)
    parser.add_argument('-iterations', help='the number of interations', type=int, default=300)

    args = parser.parse_args()
    print(args)
    main()
