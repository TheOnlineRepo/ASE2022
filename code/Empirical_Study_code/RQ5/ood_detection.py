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
import time
from keras.utils import to_categorical
from scipy.io import loadmat
import cv2 as cv
import gzip
from sklearn import metrics
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
        _, _, _, data_x, data_y = load_model_and_testcase(path, '', ood_dataset, only_data=True)
    return data_x, data_y


def get_traces(model, data, fitness, category):
    if fitness == 'nc':
        suffix = '_nc.npy'
    else:
        suffix = '.npy'
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
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, args.model, args.dataset, verbose=False)
    ood_data_x, ood_data_y = load_ood_data(args.path, args.dataset, args.ood_dataset)
    if ood_data_y.shape[0] > y_test.shape[0]:
        ood_data_x, ood_data_y = ood_data_x[:y_test.shape[0]], ood_data_y[:y_test.shape[0]]
    else:
        x_test, y_test = x_test[:ood_data_y.shape[0]], y_test[:ood_data_y.shape[0]]
    return model, ood_data_x, ood_data_y, x_train, y_train, x_test, y_test


def trace2fitness(model, trace, traces_low, traces_high, fitness_type, x_train=None, y_train=None, x_test=None, y_test=None, param=None, train_trace=None):
    fitness = None
    if fitness_type == 'nc':
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
        fitness = idc(model, args.model, x_train, y_train, x_test, y_test, args.idc_n, args.idc_layer)
    else:
        fitness = []
        layer_names = get_layers(model)
        for layer_num in range(len(layer_names)):
            start, end = get_layer_bound(model, args.sa_layer, layer_num)
            if fitness_type == 'lsa':
                sa = lsa(trace, y_test, train_trace, y_train, args.sa_n, True, start, end)

            elif fitness_type == 'dsa':
                sa = dsa(trace, y_test, train_trace, y_train, args.sa_n, start, end)
            fitness.append(sa)
            # a = 0
        fitness = np.stack(fitness)
    return fitness


def main():
    fitness_list = ['kmnc', 'tknc', 'lsa', 'nc', 'nbc', 'snac', 'idc', 'dsa']
    # fitness_list = ['lsa', 'dsa', 'nc', 'nbc', 'kmnc', 'tknc', 'snac']
    # fitness_list = ['lsa', 'dsa']
    # fitness_list = ['nbc', 'nc', 'kmnc', 'tknc', 'snac']

    model, x_ood, y_ood, x_train, y_train, x_test, y_test = init()
    save_dir = os.path.join(args.path, 'RQ5', 'OOD_Detection', args.model)
    if not os.path.exists(save_dir):
        os .makedirs(save_dir)
    save_path = os.path.join(save_dir, args.ood_dataset+'.csv')
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['fitness', 'AUC', 'AUPR'])
    for fitness in fitness_list:

        ood_trace = get_traces(model, x_ood, fitness, category='ood')
        test_trace = get_traces(model, x_test, fitness, category='test')
        train_trace = get_traces(model, x_train, fitness, category='train')
        if fitness in ['kmnc', 'nbc', 'snac']:
            traces_low, traces_high = get_trace_boundary(args.path, args.dataset, args.model)
        else:
            traces_low, traces_high = None, None
        if fitness == 'lsa' or fitness == 'dsa':
            fitness_save_dir = os.path.join(args.path, 'RQ5', 'RQ5_2', args.model, args.ood_dataset)
            if not os.path.exists(fitness_save_dir):
                os.makedirs(fitness_save_dir)
            ood_fitness_save_path = os.path.join(fitness_save_dir, fitness+'.ood.npy')
            test_fitness_save_path = os.path.join(fitness_save_dir, fitness+'.test.npy')
            ###
            if os.path.exists(ood_fitness_save_path) and os.path.exists(test_fitness_save_path):
            # if not os.path.exists(ood_fitness_save_path) and os.path.exists(test_fitness_save_path):
                ood_fitness = np.load(ood_fitness_save_path)
                test_fitness = np.load(test_fitness_save_path)
                test_fitness = np.nan_to_num(test_fitness)
                ood_fitness = np.nan_to_num(ood_fitness)
            else:
                pred_ood = np.argmax(model(x_ood).numpy(), axis=1)
                test_fitness = trace2fitness(model, test_trace, traces_low, traces_high, fitness, x_train, y_train, x_test, y_test, train_trace=train_trace)
                ood_fitness = trace2fitness(model, ood_trace, traces_low, traces_high, fitness, x_train, y_train, x_ood, pred_ood, train_trace=train_trace)
                test_fitness = np.nan_to_num(test_fitness)
                ood_fitness = np.nan_to_num(ood_fitness)
                np.save(ood_fitness_save_path, ood_fitness)
                np.save(test_fitness_save_path, test_fitness)
            ood_coverage = np.sum(ood_fitness, axis=0)
            test_coverage = np.sum(test_fitness, axis=0)
            coverage_stack = np.hstack((ood_coverage, test_coverage))
            label_stack = np.hstack((np.ones_like(ood_coverage), np.zeros_like(test_coverage)))
            prob_stack = (coverage_stack - np.min(coverage_stack)) / (np.max(coverage_stack) - np.min(coverage_stack))
        else:
            ood_fitness = trace2fitness(model, ood_trace, traces_low, traces_high, fitness, x_train, y_train, x_ood, y_ood).astype("bool")
            test_fitness = trace2fitness(model, test_trace, traces_low, traces_high, fitness, x_train, y_train, x_test, y_test).astype("bool")
            test_fitness = np.nan_to_num(test_fitness)
            ood_fitness = np.nan_to_num(ood_fitness)
            ood_coverage = np.sum(ood_fitness, axis=1)
            test_coverage = np.sum(test_fitness, axis=1)
            coverage_stack = np.hstack((ood_coverage, test_coverage))
            label_stack = np.hstack((np.ones_like(ood_coverage), np.zeros_like(test_coverage)))
            prob_stack = (coverage_stack - np.min(coverage_stack)) / (np.max(coverage_stack) - np.min(coverage_stack))
        prob_stack = np.nan_to_num(prob_stack)
        fpr, tpr, _ = metrics.roc_curve(label_stack, prob_stack)
        precision, recall, _ = metrics.precision_recall_curve(label_stack, prob_stack)
        aupr = metrics.auc(recall, precision)
        auc = metrics.auc(fpr, tpr)
        with open(save_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([fitness, auc, aupr])


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='cifar10')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='resnet20_cifar10')
    parser.add_argument('-ood_dataset', type=str, default='SVHN')
    parser.add_argument('-path', help='directory where models and datasets are stored', type=str,
                        default='/media/data0/DeepSuite/')
    parser.add_argument('-save_path', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-pool_type', help='pool using training or testing data', type=str, default='test')
    parser.add_argument('-k_kmnc', help='the parameter k for kmnc', type=int, default=10)
    parser.add_argument('-k_tknc', help='the parameter k for tknc', type=int, default=1)
    parser.add_argument('-k_nc', help='the threshold for nc', type=float, default=0.75)
    parser.add_argument('-sa_layer', help='layer_num for sa, 0:1st layer; 1:inner layer ; 2:last layer; -1:all layer', type=int, default=-1)
    parser.add_argument('-sa_n', type=int, default=10)
    parser.add_argument('-idc_n', type=int, default=6)
    parser.add_argument('-idc_layer', type=str, default='all')
    parser.add_argument('-pool_iter', type=int, default=1000)
    parser.add_argument('-wrong_rate', type=float, default=0.1)
    parser.add_argument('-iterations', help='the number of interations', type=int, default=300)

    args = parser.parse_args()
    print(args)
    main()
