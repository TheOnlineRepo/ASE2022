# encoding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
sys.path.append('/home/zhiyu/DeepSuite/adversarial/')
from adv_function import *
import csv
import random
import argparse
import time
from keras.applications.vgg16 import decode_predictions
from keras.utils import to_categorical
from scipy.io import loadmat
import cv2 as cv
import gzip

def get_ood_traces(model, data, fitness):
    if args.ood_dataset == 'SVHN':
        ood_dataset = 'SVHN'
    elif args.ood_dataset == 'fashion mnist':
        ood_dataset = 'FashionMNIST'
    elif args.dataset == 'cifar10':
        a = 0
    if fitness == 'nc':
        suffix = '_nc.npy'
    else:
        suffix = '.npy'
    save_dir = os.path.join(args.path, 'ood_trace', ood_dataset)
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

    ood_data_x, ood_data_y = load_ood_data(args.path, args.dataset, args.ood_dataset)
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, args.model, args.dataset)
    # y_pred = model(x_test)
    # Image.fromarray(np.uint8(x_test[0][:, :, 0]*255)).save('temp.jpg')
    ood_trace = get_ood_traces(model, ood_data_x, args.fitness)

    if args.fitness == 'nc':
        trace_path = os.path.join(args.path, 'trace', args.model + '_nc.npy')
    else:
        trace_path = os.path.join(args.path, 'trace', args.model + '.npy')
    ori_test_set_trace = np.load(trace_path)

    if args.fitness == 'nc':
        param = args.k_nc
    elif args.fitness == 'tknc' or args.fitness == 'tknp':
        param = args.k_tknc
    elif args.fitness == 'kmnc':
        param = args.k_kmnc
    elif args.fitness == 'lsa' or args.fitness == 'dsa':
        param = 10
    elif args.fitness == 'idc':
        param = args.idc_n
    else:
        param = 'NONE'

    return ood_trace, ori_test_set_trace, param, model, ood_data_x, ood_data_y, x_train, y_train, x_test, y_test


def create_record_file(param, num_samples):
    record_model_dir = os.path.join(args.save_path, 'RQ5', args.model, args.ood_dataset)
    if not os.path.exists(record_model_dir):
        os.makedirs(record_model_dir)
    if args.fitness == 'idc' and args.idc_layer == 'all':
        record_dir = os.path.join(record_model_dir, 'idc_all_layer')
    else:
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


def trace2fitness(model, trace, traces_low, traces_high, x_train, y_train, x_test, y_test, param):
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
    elif args.fitness == 'tknp':
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
    elif args.fitness == 'idc':
        fitness = idc(model, args.model, x_train, y_train, x_test, y_test, args.idc_n, args.idc_layer)
    else:
        train_trace = np.load(os.path.join(args.path, 'trace_boundary', args.model+'_train_trace.npy'))
        y_train = y_train[:train_trace.shape[0]]
        if args.dataset == 'Udacity':
            is_classification = False
        else:
            is_classification = True
        if args.sa_layer == -1:         # cal all layer trace
            fitness = []
            layer_names = get_layers(model)
            for layer_num in range(len(layer_names)):
                start, end = get_layer_bound(model, args.sa_layer, layer_num)
                if args.fitness == 'lsa':
                    sa = lsa(trace, y_test, train_trace, y_train, args.sa_n, is_classification, start, end)

                elif args.fitness == 'dsa':
                    sa = dsa(trace, y_test, train_trace, y_train, args.sa_n, start, end)
                fitness.append(sa)
            fitness = np.stack(fitness)
        else:
            start, end = get_layer_bound(model, args.sa_layer)
            if args.fitness == 'lsa':
                sa = lsa(trace, y_test, train_trace, y_train, args.sa_n, is_classification, start, end)
            elif args.fitness == 'dsa':
                sa = dsa(trace, y_test, train_trace, y_train, args.sa_n, start, end)
            cov = get_sc(
                np.amin(sa), sa[np.argsort(-sa)[5]], n_bucket, sa
            )
            cov[np.where(cov>n_bucket-1)] = n_bucket-1
            fitness = to_categorical(fitness)
        fitness_set = []
        for i in range(fitness.shape[0]):
            cov = get_sc(
                # np.amin(fitness[i]), fitness[i][np.argsort(-fitness[i])[5]], int(param), fitness[i]
                0, 2000, int(param), fitness[i]
            )
            cov[np.where(cov > int(param) - 1)] = int(param) - 1
            layer_fitness = to_categorical(cov)
            fitness_set.append(layer_fitness)
        fitness = np.concatenate(fitness_set, axis=1)
    return fitness

from PIL import Image
def load_ood_data(path, dataset, ood_dataset):
    if ood_dataset == 'SVHN':
        ood_dataset_name = 'SVHN'
        data_dir = os.path.join(path, 'dataset', ood_dataset_name)
        train_path = os.path.join(data_dir, 'test_32x32.mat')
        train_data = loadmat(train_path)
        data_x = train_data['X'].transpose(3, 0, 1, 2)
        data_y = np.reshape(train_data['y'], -1)
        data_y[np.where(data_y == 10)] = 0          # emmm 0 is labeled as 10..
        gray_data_x = []
        for i, img in enumerate(data_x):
            gray_img = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (28, 28))
            Image.fromarray(gray_img).save(str(data_y[i])+'.jpg')
            gray_data_x.append(np.expand_dims(gray_img, 2))
        data_x = np.array(gray_data_x).astype('float32')
        data_x /= 255
    elif ood_dataset == 'fashion mnist':
        data_dir = os.path.join(path, 'dataset', 'FashionMNIST')
        labels_path = os.path.join(data_dir,
                                   '%s-labels-idx1-ubyte.gz'
                                   % 't10k')
        images_path = os.path.join(data_dir,
                                   '%s-images-idx3-ubyte.gz'
                                   % 't10k')

        with gzip.open(labels_path, 'rb') as lbpath:
            data_y = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            data_x = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(data_y), 28, 28, 1)

        a = 0
    elif dataset == 'cifar10':
        a = 0
    # ood_loader = data_x, data_y
    return data_x, data_y


def main():
    ood_trace, ori_trace, param, model, x_ood, y_ood, x_train, y_train, x_test, y_test = init()

    traces_low, traces_high = get_trace_boundary(args.path, args.model)
    ood_fitness = trace2fitness(model, ood_trace, traces_low, traces_high, x_train, y_train, x_ood, y_ood, param)
    origin_test_fitness = trace2fitness(model, ori_trace, traces_low, traces_high, x_train, y_train, x_test, y_test, param)

    for num_samples in range(50, 501, 50):
        print("-----num samples:", num_samples, "-----")
        record_path = create_record_file(param, num_samples)

        for m_iter in tqdm(range(args.pool_iter)):
            ood_idx = random.sample([i for i in range(ood_fitness.shape[0])], num_samples)
            ori_idx = random.sample([i for i in range(origin_test_fitness.shape[0])], num_samples * 2)
            ori_base_idx, ori_add_idx = ori_idx[:num_samples], ori_idx[num_samples:]
            Ori_Ori_fitness = np.sum(np.max(origin_test_fitness[ori_idx], axis=0))
            Ori_Ood_fitness = np.sum(np.max(np.concatenate((origin_test_fitness[ori_base_idx], ood_fitness[ood_idx])), axis=0))
            Ori_fitness = np.sum(np.max(origin_test_fitness[ori_base_idx], axis=0))
            Ood_fitness = np.sum(np.max(ood_fitness[ood_idx], axis=0))
            with open(record_path, 'a+') as f:
                runtime = time.clock() - start_time
                writer = csv.writer(f)
                content = [str(m_iter), Ori_fitness, Ood_fitness, Ori_Ori_fitness,
                           Ori_Ood_fitness, runtime]
                writer.writerow(content)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='mnist')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='leNet_1')
    parser.add_argument('-path', help='directory where models and datasets are stored', type=str,
                        default='/media/data1/DeepSuite/')
    parser.add_argument('-save_path', type=str, default='/media/data1/DeepSuite')
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-pool_type', help='pool using training or testing data', type=str, default='test')
    parser.add_argument('-fitness', help='the way of calculating fitness', type=str, default='snac')
    parser.add_argument('-k_kmnc', help='the parameter k for kmnc', type=int, default=10)
    parser.add_argument('-k_tknc', help='the parameter k for tknc', type=int, default=1)
    parser.add_argument('-k_nc', help='the threshold for nc', type=float, default=0.75)
    parser.add_argument('-sa_layer', help='layer_num for sa, 0:1st layer; 1:inner layer ; 2:last layer; -1:all layer', type=int, default=-1)
    parser.add_argument('-sa_n', type=int, default=1000)
    parser.add_argument('-idc_n', type=int, default=6)
    parser.add_argument('-idc_layer', type=str, default='all')
    parser.add_argument('-pool_iter', type=int, default=1000)
    parser.add_argument('-wrong_rate', type=float, default=0.1)
    parser.add_argument('-num_samples', help='the number of samples', type=int, default=100)
    parser.add_argument('-iterations', help='the number of interations', type=int, default=300)
    parser.add_argument('-ood_dataset', type=str, default='SVHN')

    args = parser.parse_args()
    print(args)
    main()
