# encoding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
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
from PIL import Image


def load_ood_data(path, dataset, ood_dataset):
    data_dir = os.path.join(path, 'dataset', args.ood_dataset)
    if ood_dataset == 'SVHN':
        train_path = os.path.join(data_dir, 'test_32x32.mat')
        train_data = loadmat(train_path)
        data_x = train_data['X'].transpose(3, 0, 1, 2)
        data_y = np.reshape(train_data['y'], -1)
        data_y[np.where(data_y == 10)] = 0          # emmm 0 is labeled as 10..
        if dataset == 'mnist':
            gray_data_x = []
            for i, img in enumerate(data_x):
                gray_img = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (28, 28))
                # Image.fromarray(gray_img).save(str(data_y[i])+'.jpg')
                gray_data_x.append(np.expand_dims(gray_img, 2))
            data_x = np.array(gray_data_x).astype('float32')
            data_x /= 255
    elif ood_dataset == 'FashionMNIST':
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
    elif ood_dataset == 'SUN':
        choosed_fig_save_path = os.path.join(data_dir, 'OOD_choosed_fig_' + dataset + '.npy')
        sun_jpg_dir = os.path.join(data_dir, 'SUN2012pascalformat/JPEGImages')
        if os.path.exists(choosed_fig_save_path):
            data_x = np.load(choosed_fig_save_path)
        else:
            choosed_fig = np.array(random.sample(os.listdir(sun_jpg_dir), 12000))
            data_x = []
            for fig in choosed_fig:
                image = Image.open(os.path.join(sun_jpg_dir, fig))
                if dataset == 'mnist':
                    image = np.expand_dims(np.array(image.resize((28, 28), Image.ANTIALIAS).convert('L')), -1)
                    data_x.append(image)
                else:
                    image = np.array(image.resize((32, 32), Image.ANTIALIAS))
                    if image.shape[-1] == 3:
                        data_x.append(image)
                if len(data_x) == 10000:
                    break
            data_x = np.array(data_x)
            np.save(choosed_fig_save_path, data_x)
        data_y = np.zeros(10000)
    else:
        _, _, _, data_x, data_y = load_model_and_testcase(args.path, args.model, ood_dataset, only_data=True)
    return data_x, data_y


def get_traces(model, data, fitness, category):
    if fitness == 'nc':
        suffix = '_nc.npy'
    else:
        suffix = '.npy'
    if args.dataset == 'cifar100' or 'SVHN':
        skip = True
    else:
        skip = False
    if category == 'ood':
        save_dir = os.path.join(args.path, 'ood_trace', args.ood_dataset+"_"+args.dataset)
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
        layer_names = get_layers(model, skip)
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
    ood_data_x, ood_data_y = load_ood_data(args.path, args.dataset, args.ood_dataset)
    if x_test.shape[0] < ood_data_x.shape[0]:
        ood_data_x, ood_data_y = ood_data_x[:y_test.shape[0]], ood_data_y[:y_test.shape[0]]
    else:
        x_test, y_test = x_test[:ood_data_y.shape[0]], y_test[:ood_data_y.shape[0]]
    return model, ood_data_x, ood_data_y, x_train, y_train, x_test, y_test


def create_record_file(param, num_samples):
    record_model_dir = os.path.join(args.save_path, 'RQ5', args.model, args.ood_dataset)
    record_dir = os.path.join(record_model_dir, args.fitness)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    record_file = 'parodam_' + str(param) + "_" + str(args.wrong_rate) + "_" + str(num_samples) + ".csv"

    record_path = os.path.join(record_dir, record_file)
    if os.path.exists(record_path):
        os.remove(record_path)
    content = ['No.', 'Fitness', 'No. Wrong Cases', 'No. Wrong Classes', 'Jaccard', 'Runtime']
    with open(record_path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(content)
    return record_path


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
        fitness = idc(model, args.dataset, args.model, x_train, y_train, x_test, y_test, args.idc_n, args.idc_layer, only_last_layer=only_last_layer)
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
        fitness = np.stack(fitness)
    return fitness


def main():
    fitness_list = ['nc', 'kmnc', 'nbc', 'snac', 'tknc', 'lsa', 'dsa', 'idc']
    # fitness_list = ['idc', 'nc', 'nbc', 'kmnc', 'tknc', 'snac', 'dsa', 'lsa']
    # fitness_list = ['idc']

    model, x_ood, y_ood, x_train, y_train, x_test, y_test = init()
    traces_low, traces_high = get_trace_boundary(args.path, args.dataset, args.model)
    save_dir = os.path.join(args.path, 'RQ5', 'RQ5_2', args.dataset, args.model, args.ood_dataset)
    if not os.path.exists(save_dir):
        os .makedirs(save_dir)

    for fitness in fitness_list:
        ood_trace = get_traces(model, x_ood, fitness, category='ood')
        test_trace = get_traces(model, x_test, fitness, category='test')

        if test_trace.shape[0] < ood_trace.shape[0]:
            ood_trace = ood_trace[:test_trace.shape[0]]
        else:
            test_trace = test_trace[:ood_trace.shape[0]]
        train_trace = get_traces(model, x_train, fitness, category='train')
        if fitness == 'lsa' or fitness == 'dsa':
            ood_fitness_save_path = os.path.join(save_dir, fitness+'.ood.npy')
            test_fitness_save_path = os.path.join(save_dir, fitness+'.test.npy')
            if not os.path.exists(ood_fitness_save_path):
                pred_ood = np.argmax(model(x_ood).numpy(),axis=1)
                ood_fitness = trace2fitness(model, ood_trace, traces_low, traces_high, fitness, x_train, y_train, x_ood, pred_ood, train_trace=train_trace)
                ood_fitness = np.nan_to_num(ood_fitness)
                np.save(ood_fitness_save_path, ood_fitness)
                test_fitness = trace2fitness(model, test_trace, traces_low, traces_high, fitness, x_train, y_train, x_test, y_test, train_trace=train_trace)
                test_fitness = np.nan_to_num(test_fitness)
                np.save(test_fitness_save_path, test_fitness)
            else:
                ood_fitness = np.load(ood_fitness_save_path)
                test_fitness = np.load(test_fitness_save_path)
        else:
            save_path = os.path.join(save_dir, fitness+'.result.csv')
            if os.path.exists(save_path):
                continue
            ood_fitness = trace2fitness(model, ood_trace, traces_low, traces_high, fitness, x_train, y_train, x_ood, y_ood).astype("bool")
            test_fitness = trace2fitness(model, test_trace, traces_low, traces_high, fitness, x_train, y_train, x_test, y_test).astype("bool")
            # if fitness == 'idc':
            #     train_fitness = np.ones_like(test_fitness)
            # else:
            train_fitness = trace2fitness(model, train_trace, traces_low, traces_high, fitness, x_train, y_train, x_train, y_train).astype("bool")

            train_coverage = np.max(train_fitness, axis=0)
            activable_neuron = np.where(train_coverage == 0)
            activable_ood_fitness = ood_fitness.T[activable_neuron]
            activable_test_fitness = test_fitness.T[activable_neuron]
            ood_samples_activable_neurons = np.sum(activable_ood_fitness, axis=0)
            test_samples_activable_neurons = np.sum(activable_test_fitness, axis=0)
            ood_coverage = np.max(ood_fitness, axis=0)
            test_coverage = np.max(test_fitness, axis=0)
            with open(save_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['train coverage', np.sum(train_coverage)])
                writer.writerow(['ood coverage', np.sum(ood_coverage)])
                writer.writerow(['test coverage', np.sum(test_coverage)])
                writer.writerow(['ood activate', np.sum(train_coverage | ood_coverage) - np.sum(train_coverage)])
                writer.writerow(['test activate', np.sum(train_coverage | test_coverage) - np.sum(train_coverage)])
                writer.writerow(['ood activable samples', np.where(ood_samples_activable_neurons > 0)[0].shape[0]])
                writer.writerow(['test activable samples', np.where(test_samples_activable_neurons > 0)[0].shape[0]])
                writer.writerow(['mean(neuron activated by ood)', np.sum(ood_samples_activable_neurons) / np.sum(ood_samples_activable_neurons > 0)])
                writer.writerow(['mean(neuron activated by test)', np.sum(test_samples_activable_neurons) / np.sum(test_samples_activable_neurons > 0)])
                writer.writerow(['sum(neuron activated by ood)', np.sum(ood_samples_activable_neurons)])
                writer.writerow(['sum(neuron activated by test)', np.sum(test_samples_activable_neurons)])


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='SVHN')
    parser.add_argument('-model', type=str, default='resnet34')
    parser.add_argument('-ood_dataset', type=str, default='cifar10')
    parser.add_argument('-path', help='directory where models and datasets are stored', type=str,
                        default='/media/data0/DeepSuite/')
    parser.add_argument('-save_path', type=str, default='/media/data1/DeepSuite')
    parser.add_argument('-batch_size', type=int, default=8192 )
    parser.add_argument('-pool_type', help='pool using training or testing data', type=str, default='test')
    parser.add_argument('-k_kmnc', help='the parameter k for kmnc', type=int, default=10)
    parser.add_argument('-k_tknc', help='the parameter k for tknc', type=int, default=1)
    parser.add_argument('-k_nc', help='the threshold for nc', type=float, default=0.75)
    parser.add_argument('-sa_layer', help='layer_num for sa, 0:1st layer; 1:inner layer ; 2:last layer; -1:all layer', type=int, default=-1)
    parser.add_argument('-sa_n', type=int, default=1000)
    parser.add_argument('-idc_n', type=int, default=6)
    parser.add_argument('-idc_layer', type=str, default='all')
    parser.add_argument('-pool_iter', type=int, default=1000)
    parser.add_argument('-wrong_rate', type=float, default=0.1)
    parser.add_argument('-iterations', help='the number of interations', type=int, default=300)

    args = parser.parse_args()
    print(args)
    main()
