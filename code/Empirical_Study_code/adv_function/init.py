# encoding=utf-8
import os
import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import pickle as p
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


def load_CIFAR_batch(filename):
    '''load cifar10'''
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR100_batch(filename):
    '''load cifar10'''
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['fine_labels']
        Y = np.array(Y)
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
        X /= 255
        return X, Y


def image2ndarray(file_dir):
    image_array = []
    label_array = []
    for column_name in os.listdir(file_dir):
        column_path = os.path.join(file_dir, column_name)
        for ind in os.listdir(column_path):
            ind_path = os.path.join(column_path, ind)
            img = image.load_img(ind_path, target_size=(224, 224))
            img = image.img_to_array(img)
            image_array.append(img)
            label_array.append(column_name)
    image_array = np.array(image_array)
    image_array = preprocess_input(image_array)
    label_array = np.array(label_array)
    return image_array, label_array


def load_t_model(directory, modelname, datasetname, verbose=True):
    modelpath = os.path.join(directory, 'trained_models', datasetname, modelname + '.h5')
    if not os.path.exists(modelpath):
        print('No ' + modelname + '.h5 exists in ' + directory + 'trained_models/!')
        return None
    if verbose:
        print('load ' + modelname + '...')

    model = load_model(modelpath)
    if verbose:
        print(modelpath)
        print(model.summary())
    return model


def load_model_and_testcase(directory, modelname, datasetname, cal_acc_pt=False, only_data=False, verbose=True):
    '''load model and training test cases'''
    if datasetname.lower() == 'mnist':
        # load training test cases
        filepath = os.path.join(directory, 'dataset', 'mnist.npz')
        if not os.path.exists(filepath):
            print('No mnist.npz in ' + directory + '/dataset/!')
            return None
        f = np.load(filepath)
        x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

        # preprocess
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_train = x_train.astype('float32')
        x_train /= 255

        x_test = x_test.reshape(-1, 28, 28, 1)
        x_test = x_test.astype('float32')
        x_test /= 255

    elif datasetname.lower() == 'cifar10':
        filepath = os.path.join(directory, 'dataset', 'cifar-10-batches-py')
        if not os.path.exists(filepath):
            print('No cifar-10-batches-py in ' + directory + '/dataset/!')
            return None

        xs = []
        ys = []
        for b in range(1, 6):
            filename = os.path.join(filepath, 'data_batch_%d' % (b,))
            X, Y = load_CIFAR_batch(filename)
            xs.append(X)
            ys.append(Y)
        x_train = np.concatenate(xs)
        y_train = np.concatenate(ys)
        del X, Y

        x_test, y_test = load_CIFAR_batch(os.path.join(filepath, 'test_batch'))

        x_train = x_train.astype('float32')
        x_train /= 255
        x_test = x_test.astype('float32')
        x_test /= 255
    elif datasetname == 'SVHN':
        data_dir = os.path.join(directory, 'dataset', 'SVHN')

        train_path = os.path.join(data_dir, 'train_32x32.mat')
        train_data = loadmat(train_path)
        x_train = train_data['X'].transpose(3, 0, 1, 2)
        y_train = np.reshape(train_data['y'], -1)
        y_train[np.where(y_train == 10)] = 0

        test_path = os.path.join(data_dir, 'test_32x32.mat')
        test_data = loadmat(test_path)
        x_test = test_data['X'].transpose(3, 0, 1, 2)
        y_test = np.reshape(test_data['y'], -1)
        y_test[np.where(y_test == 10)] = 0
    elif datasetname == 'cifar100':
        data_dir = os.path.join(directory, 'dataset', 'cifar-100-python')
        train_path = os.path.join(data_dir, 'train')
        test_path = os.path.join(data_dir, 'test')
        x_train, y_train = load_CIFAR100_batch(train_path)
        x_test, y_test = load_CIFAR100_batch(test_path)

    # load model
    if only_data:
        model = None
    else:
        model = load_t_model(directory, modelname, datasetname, verbose=verbose)

    if cal_acc_pt:
        cal_acc(model, x_train, y_train, x_test, y_test)
        sys.exit()
    return model, x_train, y_train, x_test, y_test


def cal_acc(model, x_train, y_train, x_test, y_test):
    neuron_num = 0
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        neuron_num += int(layer.output_shape[-1])
    print("Neuron num", neuron_num)
    print("Layer num", len(model.layers))
    train_pred = np.argmax(model.predict(x_train), axis=1)
    train_wrong_pred_num = np.sum(train_pred != y_train)
    train_acc = np.sum(train_pred == y_train) / x_train.shape[0]
    test_pred = np.argmax(model.predict(x_test), axis=1)
    test_wrong_pred_num = np.sum(test_pred != y_test)
    test_acc = np.sum(test_pred == y_test) / x_test.shape[0]
    print("Train acc:", train_acc)
    print("Train wrong pred", train_wrong_pred_num)
    print("Test acc:", test_acc)
    print("Test wrong pred", test_wrong_pred_num)


def create_adv_dataset_dir(path, dataset, model):
    base_dir = os.path.join(path, 'adv_dataset', dataset, model)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir


def load_data_split(path, model, x_train, y_train, x_test, y_test, test_size):
    split_dir = os.path.join(path, 'train_test_split_new')
    train_path = os.path.join(split_dir, 'train_id.npy')
    valid_path = os.path.join(split_dir, 'valid_id.npy')
    test_path = os.path.join(split_dir, 'test_id.npy')
    if os.path.exists(split_dir):
        train_id = np.load(train_path)
        valid_id = np.load(valid_path)
        test_id = np.load(test_path)
    else:
        train_pred_y = np.argmax(model.predict(x_train), axis=1)
        correct_id = np.where(train_pred_y == y_train)[0]
        train_id, valid_id, train_y, valid_y = train_test_split(correct_id, y_train[correct_id], test_size=test_size, stratify=y_train[correct_id])
        test_pred_y = np.argmax(model.predict(x_test), axis=1)
        test_id = np.where(test_pred_y == y_test)[0]
        os.mkdir(split_dir)
        np.save(train_path, train_id)
        np.save(valid_path, valid_id)
        np.save(test_path, test_id)
    return train_id, valid_id, test_id


def create_adv_record_dir(path, dataset, model, adv_method, adv_eps):
    adv_save_dir = os.path.join(path, 'adv_trace')
    dataset_dir = os.path.join(adv_save_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    model_dir = os.path.join(dataset_dir, model)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    adv_method_dir = os.path.join(model_dir, adv_method)
    if not os.path.exists(adv_method_dir):
        os.mkdir(adv_method_dir)
    return adv_method_dir


def load_adv_sample(path, dataset, model, adv_method, attack_eps):
    adv_dataset_dir = os.path.join(path, 'adv_dataset', dataset, model, adv_method, str(attack_eps))
    train_path = os.path.join(adv_dataset_dir, 'train.npy.npz')
    valid_path = os.path.join(adv_dataset_dir, 'valid.npy.npz')
    test_path = os.path.join(adv_dataset_dir, 'test.npy.npz')
    train_adv = np.load(train_path)
    valid_adv = np.load(valid_path)
    test_adv = np.load(test_path)
    train_x, train_y = train_adv['x'], train_adv['y']
    valid_x, valid_y = valid_adv['x'], valid_adv['y']
    test_x, test_y = test_adv['x'], test_adv['y']
    return train_x, train_y, valid_x, valid_y, test_x, test_y



