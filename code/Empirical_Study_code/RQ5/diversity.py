# encoding=utf-8
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.compat.v1.disable_eager_execution()
import sys

sys.path.append('/home/zhiyu/DeepSuite/adversarial/')
from adv_function import *
import csv
import argparse
from keras.utils import to_categorical


def get_traces(model, data, fitness, category):
    if fitness == 'nc':
        suffix = '_nc.npy'
    else:
        suffix = '.npy'
    if category == 'ood':
        save_dir = os.path.join(args.path, 'ood_trace', args.ood_dataset + "_" + args.dataset)
    elif category == 'train':
        save_dir = os.path.join(args.path, 'trace', 'train_trace', args.dataset)
    elif category == 'test':
        save_dir = os.path.join(args.path, 'trace', 'test_trace', args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, args.model + suffix)
    if os.path.exists(save_path):
        traces = np.load(save_path)
    else:
        layer_names = get_layers(model)
        traces = None
        print("Calculating traces ...")
        for layer_name in tqdm(layer_names):
            layer = model.get_layer(layer_name)
            temp_model = Model(inputs=model.input, outputs=layer.output)
            layer_output = temp_model.predict(data, batch_size=args.batch_size, verbose=0)
            if fitness == 'nc':
                layer_output = scale_output(layer_output)
            if layer_output[0].ndim == 3:
                # for convulutional layer where layer_output[0].ndim == 1
                # layer_vector = cov2vec(layer_output)
                layer_vector = list(map(cov2vec, layer_output))
            else:
                # layer_output[0].ndim == 1
                layer_vector = layer_output
            layer_vector = np.array(layer_vector)
            if traces is None:
                traces = layer_vector
            else:
                traces = np.append(traces, layer_vector, axis=1)
        np.save(save_path, traces)
    return traces


def init():
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, args.model, args.dataset)
    # ood_data_x, ood_data_y = load_ood_data(args.path, args.dataset, args.ood_dataset)
    # ood_data_x, ood_data_y = ood_data_x[:y_test.shape[0]], ood_data_y[:y_test.shape[0]]
    return model, x_train, y_train, x_test, y_test


def trace2fitness(model, trace, traces_low, traces_high, fitness_type, x_train=None, y_train=None, x_test=None, y_test=None, param=None, train_trace=None):
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
            layer_names = get_layers(model, skip=skip)
            for layer_num in range(len(layer_names)):
                start, end = get_layer_bound(model, -1, layer_num)
                if fitness_type == 'lsa':
                    sa = lsa(trace, y_test, train_trace, y_train, args.sa_n, True, start, end, num_classes=num_class)
                elif fitness_type == 'dsa':
                    sa = dsa(trace, y_test, train_trace, y_train, args.sa_n, start, end)
                if sa is not None:
                    fitness.append(sa)
            fitness = np.stack(fitness)
            if not os.path.exists(test_sa_dir):
                os.makedirs(test_sa_dir)
            np.save(test_sa_path, fitness)
    return fitness


def main():
    fitness_list = ['nc', 'kmnc', 'nbc', 'snac', 'tknc','lsa',  'dsa', 'idc']
    # fitness_list = ['idc', 'dsa', 'lsa', 'nc', 'nbc', 'kmnc', 'tknc', 'snac']
    # fitness_list = ['idc']
    # fitness_list = ['nbc', 'nc', 'kmnc', 'tknc', 'snac']

    model, x_train, y_train, x_test, y_test = init()
    traces_low, traces_high = get_trace_boundary(args.path, args.dataset, args.model)
    save_dir = os.path.join(args.path, 'RQ1', 'diversity', args.dataset, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for fitness_type in fitness_list:
        save_path = os.path.join(save_dir, fitness_type + '.csv')
        if not os.path.exists(save_path):
            with open(save_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['fitness', 'num_div', 'each_div', 'coverage'])

        test_trace = get_traces(model, x_test, fitness_type, category='test')
        train_trace = get_traces(model, x_train, fitness_type, category='train')
        print(type(test_trace))
        print(len(test_trace))

        D = {}
        for i in range(0, len(y_test)):
            if y_test[i] not in D.keys():
                D[y_test[i]] = [i]
            else:
                D[y_test[i]].append(i)
        # print('div: ', D.keys())

        for d in D.keys():
            print('key: ', d, ' D[d]: ', len(D[d]))
        test_fitness = trace2fitness(model, test_trace, traces_low, traces_high, fitness_type, x_train, y_train, x_test, y_test, train_trace=train_trace)
        if fitness_type == 'lsa' or fitness_type == 'dsa':
            fitness_set = []
            for i in range(test_fitness.shape[0]):
                cov = get_sc(
                    np.amin(test_fitness[i]), test_fitness[i][np.argsort(-test_fitness[i])[5]], 10, test_fitness[i]
                )
                cov[np.where(cov > 9)] = 9
                layer_fitness = to_categorical(cov)
                fitness_set.append(layer_fitness)
            test_fitness = np.concatenate(fitness_set, axis=1)
        else:
            test_fitness = test_fitness.astype('bool')

        for j in range(args.iterations):
            div_idx = random.sample(D.keys(), args.num_div)
            print('div_idx: ', div_idx)
            selected_idx = []
            for d in div_idx:
                selected_idx.extend(random.sample(D[d], args.each_div))

            print('test_fitness.shape: ', test_fitness.shape)
            selected_fitness = test_fitness[selected_idx, :]

            # print('type of selected fitness: ', type(selected_fitness))
            print('size of selected fitness: ', selected_fitness.shape)

            coverage = np.max(selected_fitness, axis=0)
            coverage_rate = np.sum(coverage) / coverage.shape[0]

            with open(save_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([fitness_type, args.num_div, args.each_div, coverage_rate])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='SVHN')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='vgg19')
    parser.add_argument('-ood_dataset', type=str, default='SUN')
    parser.add_argument('-path', help='directory where models and datasets are stored', type=str,
                        default='/mnt/hdd1/DeepSuite/')
    parser.add_argument('-save_path', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-pool_type', help='pool using training or testing data', type=str, default='test')
    parser.add_argument('-k_kmnc', help='the parameter k for kmnc', type=int, default=10)
    parser.add_argument('-k_tknc', help='the parameter k for tknc', type=int, default=1)
    parser.add_argument('-k_nc', help='the threshold for nc', type=float, default=0.75)
    parser.add_argument('-sa_layer', help='layer_num for sa, 0:1st layer; 1:inner layer ; 2:last layer; -1:all layer',
                        type=int, default=-1)
    parser.add_argument('-sa_n', type=int, default=1000)
    parser.add_argument('-idc_n', type=int, default=6)
    parser.add_argument('-idc_layer', type=str, default='all')
    parser.add_argument('-pool_iter', type=int, default=1000)
    parser.add_argument('-wrong_rate', type=float, default=0.1)
    parser.add_argument('-ns', '--each_div', help='the number of samples', type=int, default=800)
    parser.add_argument('-div', '--num_div', help='the number of diversity', type=int, default=1)
    parser.add_argument('-it', '--iterations', help='the number of interations', type=int, default=30)

    args = parser.parse_args()
    print(args)
    main()
