# encoding=utf-8
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys

sys.path.append('/home/zhiyu/DeepSuite/adversarial')
from adv_function import *
import csv
import argparse
import time
from keras.applications.vgg16 import decode_predictions
from keras.utils import to_categorical


def init():
    if args.fitness == 'nc':
        trace_path = os.path.join(args.path, 'trace', 'test_trace', args.dataset, args.model + '_nc.npy')
    else:
        trace_path = os.path.join(args.path, 'trace', 'test_trace', args.dataset, args.model + '.npy')
    trace = np.load(trace_path)

    fitness_cot = 1
    if args.fitness == 'nc':
        param = [args.k_nc]
    elif args.fitness == 'tknc' or args.fitness == 'tknp':
        param = [args.k_tknc]
        fitness_cot = args.k_tknc
    elif args.fitness == 'kmnc':
        param = [args.k_kmnc]
        fitness_cot = args.k_kmnc
    elif args.fitness == 'idc':
        param = [args.idc_n]
    elif args.fitness == 'lsa' or args.fitness == 'dsa':
        # param = ['10', '50', '100', '200', '500', '1000']
        param = [args.sa_n]
    else:
        if args.fitness == 'nbc':
            fitness_cot = 2
        param = ['NONE']
    num_samples = args.num_samples * fitness_cot
    return num_samples, trace, param


def filter_test_case(x_test, y_test, model):
    global predictions
    if args.dataset == 'mnist':
        predictions = model.predict_classes(x_test, verbose=1)
    elif args.dataset == 'cifar10' or args.dataset == 'SVHN':
        y_pred = model.predict(x_test, verbose=1)
        predictions = np.argmax(y_pred, axis=1)
    elif args.dataset == 'ImageNet':
        y_pred = model.predict(x_test)
        y_pred = decode_predictions(y_pred, top=1)
        predictions = [y_pred[i][0][0] for i in range(len(y_pred))]
    elif args.dataset == 'cifar100':
        y_pred = model.predict(x_test, verbose=1)
        # print(y_pred)
        predictions = np.argmax(y_pred, axis=1)
        # print(predictions)
    filter_check = np.array([predictions[i] == y_test[i] for i in range(len(y_test))])
    return filter_check


def create_record_file(param):
    record_model_dir = os.path.join(args.save_path, 'jaccard', args.model)
    if not os.path.exists(record_model_dir):
        os.mkdir(record_model_dir)
    if args.fitness == 'idc' and args.idc_layer == 'all':
        record_dir = os.path.join(record_model_dir, 'idc_all_layer')
    else:
        record_dir = os.path.join(record_model_dir, args.fitness)
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    record_file = 'same_parodam_' + str(param) + ".csv"
    record_path_same = os.path.join(record_dir, record_file)
    if os.path.exists(record_path_same):
        os.remove(record_path_same)
    content = ['No.', 'id1', 'id2', 'Jaccard']
    with open(record_path_same, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(content)
    record_file = 'diff_parodam_' + str(param) + ".csv"
    record_path_diff = os.path.join(record_dir, record_file)
    if os.path.exists(record_path_diff):
        os.remove(record_path_diff)
    content = ['No.', 'id1', 'id2', 'Jaccard']
    with open(record_path_diff, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(content)

    return record_path_same, record_path_diff


def trace2fitness(model, trace, traces_low, traces_high, x_train, y_train, x_test, y_test):
    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        skip = True
    else:
        skip = False
    if args.dataset == 'cifar100':
        num_class = 100
    else:
        num_class = 10
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
    elif args.fitness == 'idc':
        if args.dataset != 'mnist':
            only_last_layer = True
        else:
            only_last_layer = False
        fitness = idc(model, args.dataset, args.model, x_train, y_train, x_test, y_test, args.idc_n, args.idc_layer,
                      only_last_layer=only_last_layer)
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


def cal_jaccard_score(fitness1, fitness2):
    fitness1 = fitness1.astype('int')
    fitness2 = fitness2.astype('int')
    jaccard = 1 - float(np.sum(fitness1 & fitness2)) / np.sum(fitness1 | fitness2)
    return jaccard


def choose_tuple(label, class_num=10, choose_num=10000):
    class_idx = []
    same_tuple = []
    for i in range(class_num):
        class_idx.append(np.where(label == i)[0])
        for j in range(int(choose_num / class_num)):
            same_tuple.append(np.random.choice(class_idx[i], 2, replace=False))
    same_tuple = np.array(same_tuple)
    idx = np.array([i for i in range(label.shape[0])])
    temp_diff_tuple = [np.random.choice(idx, choose_num * 2) for i in range(2)]
    diff_tuple = []
    for i in range(choose_num * 2):
        if label[temp_diff_tuple[0][i]] != label[temp_diff_tuple[1][i]]:
            diff_tuple.append([temp_diff_tuple[0][i], temp_diff_tuple[1][i]])
        if len(diff_tuple) == choose_num:
            break
    diff_tuple = np.array(diff_tuple)
    return same_tuple, diff_tuple


def angle_based_tuple(label, choose_num=10000):
    idx = np.array([i for i in range(label.shape[0])])
    rank = np.argsort(label)
    diff_tuple, same_tuple = [], []
    for i in range(choose_num):
        id1 = np.random.choice(idx[:-1], 1)[0]
        same_tuple.append((rank[id1], rank[id1 + 1]))
    while (True):
        id1, id2 = np.random.choice(idx, 2, replace=False)
        if abs(label[id1] - label[id2]) > 0.01:
            diff_tuple.append((id1, id2))
            if len(diff_tuple) == choose_num:
                break
    same_tuple = np.array(same_tuple)
    diff_tuple = np.array(diff_tuple)
    return same_tuple, diff_tuple


def load_train_traces():
    file_path = os.path.join(args.path, 'trace_boundary', args.model + '_train_trace.npy')
    train_trace = np.load(file_path)
    return train_trace


def train_tuple(label, choose_num=10000):
    left_index = np.array(['left' in label[i] for i in range(label.shape[0])])
    center_index = np.array(['center' in label[i] for i in range(label.shape[0])])
    right_index = np.array(['right' in label[i] for i in range(label.shape[0])])
    label_index = np.vstack([left_index, right_index, center_index])
    same_tuple, diff_tuple = [], []
    for i in range(3):
        m_label = np.where(label_index[i] == 1)[0]
        o_label = np.where(label_index[i] == 0)[0]
        m, o = np.random.choice(m_label, int(choose_num / 3)), np.random.choice(o_label, int(choose_num / 3))
        for j in range((int(choose_num / 3))):
            same_tuple.append(np.random.choice(m_label, 2, replace=False))
            diff_tuple.append((m[j], o[j]))
    same_tuple = np.array(same_tuple)
    diff_tuple = np.array(diff_tuple)
    return same_tuple, diff_tuple


def main():
    # print(tf.test.is_gpu_available())
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, args.model, args.dataset)
    num_samples, trace, params = init()
    # trace = load_train_traces()
    if args.dataset == 'Udacity':
        '''
        plan A: classify by left and right
        '''
        # class_num = 2
        # unit = (np.max(y_test) - np.min(y_test)) / class_num + 10e-5
        # class_label = ((y_test - np.min(y_test)) / unit).astype(int)
        # # choosed = np.where((y_test>0.1) | (y_test < -0.1))
        # # x_test, y_test, trace = x_test[choosed], y_test[choosed], trace[choosed]
        # # y_test = y_test > 0
        # same_tuple, diff_tuple = choose_tuple(class_label, class_num)
        '''
        plan B: classify by difference of angle
        '''
        same_tuple, diff_tuple = angle_based_tuple(y_test)
        '''
        plan C: classify by difference of angle
        '''
        # same_tuple, diff_tuple = train_tuple(x_train)
    else:
        if args.dataset != 'cifar100':
            class_num = 10
        else:
            class_num = 100
        same_tuple, diff_tuple = choose_tuple(y_test, class_num)

    traces_low, traces_high = get_trace_boundary(args.path, args.dataset, args.model)
    fitness = trace2fitness(model, trace, traces_low, traces_high, x_train, y_train, x_test, y_test)
    # m_fitness = fitness
    m_fitness = np.delete(fitness, np.isnan(np.sum(fitness, axis=1)), axis=0)
    for param in params:
        record_path_same, record_path_diff = create_record_file(param)
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
        jaccard_score_list = []
        for i in range(same_tuple.shape[0]):
            jaccard_score = cal_jaccard_score(fitness[same_tuple[i][0]], fitness[same_tuple[i][1]])
            jaccard_score_list.append(jaccard_score)


        print("aaaaaaaa=", record_path_same)
        print("aaaaaaaa=", record_path_diff)


        with open(record_path_same, 'a+') as f:
            for i in range(same_tuple.shape[0]):
                writer = csv.writer(f)
                content = [str(i), same_tuple[i][0], same_tuple[i][1], jaccard_score_list[i]]
                writer.writerow(content)

        jaccard_score_list = []
        for i in range(diff_tuple.shape[0]):
            jaccard_score = cal_jaccard_score(fitness[diff_tuple[i][0]], fitness[diff_tuple[i][1]])
            jaccard_score_list.append(jaccard_score)
        with open(record_path_diff, 'a+') as f:
            for i in range(diff_tuple.shape[0]):
                writer = csv.writer(f)
                content = [str(i), diff_tuple[i][0], diff_tuple[i][1], jaccard_score_list[i]]
                writer.writerow(content)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', choices=['mnist', 'cifar10', 'cifar100', 'SVHN'], type=str, default='SVHN')
    parser.add_argument('-model',
                        choices=['leNet_1', 'leNet_4', 'leNet_5', 'resnet20_cifar10', 'resnet50_cifar10', 'MobileNet',
                                 'vgg13', 'vgg16', 'vgg19', 'DenseNet121', 'WRN', 'GoogleNet', 'resnet34'], type=str,
                        default='vgg19')
    parser.add_argument('-path', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-save_path', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-pool_type', help='pool using training or testing data', type=str, default='test')
    parser.add_argument('-fitness', help='the way of calculating fitness', type=str, default='kmnc')
    parser.add_argument('-k_kmnc', help='the parameter k for kmnc', type=int, default=10)
    parser.add_argument('-k_tknc', help='the parameter k for tknc', type=int, default=3)
    parser.add_argument('-k_nc', help='the threshold for nc', type=float, default=0.25)
    parser.add_argument('-sa_layer', help='layer_num for sa, 0:1st layer; 1:inner layer ; 2:last layer; -1:all layer',
                        type=int, default=-1)
    parser.add_argument('-sa_n', type=int, default=10)
    parser.add_argument('-idc_n', type=int, default=6)
    parser.add_argument('-idc_layer', type=str, default='all')
    parser.add_argument('-num_samples', help='the number of samples', type=int, default=500)
    parser.add_argument('-iterations', help='the number of interations', type=int, default=300)

    args = parser.parse_args()
    print(args)
    main()
