import os
import numpy as np
from adv_function.trace import get_layers, cov2vec
from tensorflow.keras.models import Model
from adv_function.init import load_model_and_testcase, load_adv_sample, load_t_model
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch
import gc


def load_pretrained_model(vec_dim, model_name):
    if vec_dim == 50:
        model_dir = '/media/data1/DeepSuite/trained_models/adv_pretrain_model'
        model_path = os.path.join(model_dir, model_name + '.pkl')
        model = torch.load(model_path)
    else:
        if model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            fc_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(fc_features, vec_dim)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            fc_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(fc_features, vec_dim)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
            fc_features = model.fc.in_features
            model.fc = nn.Linear(fc_features, vec_dim)
        elif model_name == 'inception_v3':
            model = models.inception_v3(pretrained=True)
            fc_features = model.fc.in_features
            model.fc = nn.Linear(fc_features, vec_dim)
        model = model.eval().cuda()
    return model


def get_attention_layer(layer_names, resnet_v=3):
    attention_layers = ['activation_1']
    for i, layer_name in enumerate(layer_names):
        split_name = layer_name.split('_')
        if split_name[0] == 'add' and int(split_name[1]) % resnet_v == 0:
            attention_layers.append(layer_names[i + 1])
    return attention_layers


class ImageDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, X):
        img_data = torch.from_numpy(np.transpose(X, [0, 3, 1, 2]))
        img_data = F.interpolate(img_data, size=[224, 224])
        self.images = img_data

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return self.images.shape[0]


def dataset_2_vec(data, model, batch_size=256):
    image_dataset = ImageDataset(data)
    image_loader = DataLoader(dataset=image_dataset,
                              batch_size=batch_size)
    image_vec = []
    for i, images in enumerate(image_loader):
        images = images.cuda()
        outputs = model(images)
        outputs = outputs.cpu().detach().numpy()
        image_vec.append(outputs)
        gc.collect()
    image_vec = np.concatenate(image_vec, axis=0)
    return image_vec


def data_average(trace, y, vec):
    origin_label = np.where(y == 0)[0]
    adv_label = np.where(y == 1)[0]
    if origin_label.shape[0] > adv_label.shape[0]:
        choose_id = np.random.choice(origin_label, adv_label.shape[0], replace=False)
        y = np.concatenate((y[choose_id], y[adv_label]))
        vec = np.concatenate((vec[choose_id], vec[adv_label]))
        for i in range(4):
            trace[i] = np.concatenate((trace[i][choose_id], trace[i][adv_label]))
    return trace, y, vec


def load_attention_traces(path, dataset, model_name, adv_method, adv_eps, with_vec=False, batch_size=128):
    save_dir = os.path.join(path, '4layer_trace', model_name, adv_method, str(adv_eps))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_path = os.path.join(save_dir, 'train_new.npz')
    valid_path = os.path.join(save_dir, 'valid_new.npz')
    test_path = os.path.join(save_dir, 'test_new.npz')
    if os.path.exists(train_path):
        # if (False):
        train = np.load(train_path)
        valid = np.load(valid_path)
        test = np.load(test_path)
        train_traces = [train['l0'], train['l1'], train['l2'], train['l3']]
        train_y = train['y']
        valid_traces = [valid['l0'], valid['l1'], valid['l2'], valid['l3']]
        valid_y = valid['y']
        test_traces = [test['l0'], test['l1'], test['l2'], test['l3']]
        test_y = test['y']
    else:
        model, train_x, train_y, valid_x, valid_y, test_x, test_y = load_detector_samples(path, dataset, model_name,
                                                                                          adv_method, adv_eps)
        layer_names = get_layers(model)
        attention_layer = get_attention_layer(layer_names)
        train_traces, valid_traces, test_traces = [], [], []
        for layer_name in attention_layer:
            layer = model.get_layer(layer_name)
            temp_model = Model(inputs=model.input, outputs=layer.output)
            layer_train_trace = temp_model.predict(train_x, batch_size=batch_size, verbose=0)
            layer_valid_trace = temp_model.predict(valid_x, batch_size=batch_size, verbose=0)
            layer_test_trace = temp_model.predict(test_x, batch_size=batch_size, verbose=0)
            train_traces.append(layer_train_trace)
            valid_traces.append(layer_valid_trace)
            test_traces.append(layer_test_trace)
        np.savez(train_path, l0=train_traces[0], l1=train_traces[1], l2=train_traces[2], l3=train_traces[3], y=train_y)
        np.savez(valid_path, l0=valid_traces[0], l1=valid_traces[1], l2=valid_traces[2], l3=valid_traces[3], y=valid_y)
        np.savez(test_path, l0=test_traces[0], l1=test_traces[1], l2=test_traces[2], l3=test_traces[3], y=test_y)
    if with_vec:
        return train_traces, valid_traces, test_traces, train_y, valid_y, test_y
    else:
        return train_traces, valid_traces, test_traces, train_y, valid_y, test_y


def load_detector_samples(path, dataset, model_name, adv_method, adv_eps, only_data=False):
    model, train_X, train_Y, test_X, test_Y = load_model_and_testcase(path, model_name, dataset, only_data=only_data)

    detector_split_dir = os.path.join(path, 'adv_dataset', dataset, model_name, 'train_test_split_new')
    train_id = np.load(os.path.join(detector_split_dir, 'train_id.npy'))
    valid_id = np.load(os.path.join(detector_split_dir, 'valid_id.npy'))
    test_id = np.load(os.path.join(detector_split_dir, 'test_id.npy'))
    train_origin, valid_origin, test_origin = train_X[train_id], train_X[valid_id], test_X[test_id]
    train_adv, valid_adv, test_adv = load_adv_sample(path, dataset, model_name, adv_method, adv_eps)

    train_x = np.concatenate((train_origin, train_adv))
    train_y = np.concatenate((np.zeros(train_origin.shape[0]), np.ones(train_adv.shape[0])))
    valid_x = np.concatenate((valid_origin, valid_adv))
    valid_y = np.concatenate((np.zeros(valid_origin.shape[0]), np.ones(valid_adv.shape[0])))
    test_x = np.concatenate((test_origin, test_adv))
    test_y = np.concatenate((np.zeros(test_origin.shape[0]), np.ones(test_adv.shape[0])))
    return model, train_x, train_y, valid_x, valid_y, test_x, test_y


def adjust_learning_rate(optimizer, epoch, lr=0.001, ch1=5, ch2=15):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch == ch1:
        lr /= 3
    elif epoch == ch2:
        lr /= 3
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_mix_samples(path, dataset, model_name, mix_method):
    save_dir = os.path.join(path, '4layer_trace', model_name, "_".join(mix_method))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_path = os.path.join(save_dir, 'train_mix.npz')
    valid_path = os.path.join(save_dir, 'valid_mix.npz')
    test_path = os.path.join(save_dir, 'test_mix.npz')
    if os.path.exists(train_path):
        train = np.load(train_path)
        valid = np.load(valid_path)
        test = np.load(test_path)
        train_x, train_y = train['x'], train['y']
        valid_x, valid_y = valid['x'], valid['y']
        test_x, test_y = test['x'], test['y']
    else:
        model, train_X, train_Y, test_X, test_Y = load_model_and_testcase(path, model_name, dataset, only_data=True)
        detector_split_dir = os.path.join(path, 'adv_dataset', dataset, model_name, 'train_test_split_new')
        train_id = np.load(os.path.join(detector_split_dir, 'train_id.npy'))
        valid_id = np.load(os.path.join(detector_split_dir, 'valid_id.npy'))
        test_id = np.load(os.path.join(detector_split_dir, 'test_id.npy'))
        train_origin, valid_origin, test_origin = train_X[train_id], train_X[valid_id], test_X[test_id]

        adv_tr, adv_val, adv_ts = [], [], []
        for label_id, adv_method in enumerate(mix_method):
            if adv_method in ['PGD', 'FGSM', 'BIM']:
                adv_eps = 0.01
            else:
                adv_eps = 0.1
            train_adv, valid_adv, test_adv = load_adv_sample(path, dataset, model_name, adv_method, adv_eps)
            train_adv = train_adv[
                np.random.choice(range(train_adv.shape[0]), int(train_origin.shape[0] / len(mix_method)),
                                 replace=False)]
            valid_adv = valid_adv[
                np.random.choice(range(valid_adv.shape[0]), int(valid_origin.shape[0] / len(mix_method)),
                                 replace=False)]
            test_adv = test_adv[
                np.random.choice(range(test_adv.shape[0]), int(test_origin.shape[0] / len(mix_method)), replace=False)]
            adv_tr.append(train_adv)
            adv_val.append(valid_adv)
            adv_ts.append(test_adv)
        train_adv = np.concatenate(adv_tr)
        valid_adv = np.concatenate(adv_val)
        test_adv = np.concatenate(adv_ts)

        train_x = np.concatenate((train_origin, train_adv))
        train_y = np.concatenate((np.zeros(train_origin.shape[0]), np.ones(train_adv.shape[0])))
        valid_x = np.concatenate((valid_origin, valid_adv))
        valid_y = np.concatenate((np.zeros(valid_origin.shape[0]), np.ones(valid_adv.shape[0])))
        test_x = np.concatenate((test_origin, test_adv))
        test_y = np.concatenate((np.zeros(test_origin.shape[0]), np.ones(test_adv.shape[0])))
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_mix_data(path, dataset, model_name, mix_method, batch_size=128):
    save_dir = os.path.join(path, '4layer_trace', model_name, "_".join(mix_method))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_path = os.path.join(save_dir, 'train_mix.npz')
    valid_path = os.path.join(save_dir, 'valid_mix.npz')
    test_path = os.path.join(save_dir, 'test_mix.npz')
    if os.path.exists(train_path):
        train = np.load(train_path)
        valid = np.load(valid_path)
        test = np.load(test_path)
        train_traces = [train['l0'], train['l1'], train['l2'], train['l3']]
        train_y = train['y']
        valid_traces = [valid['l0'], valid['l1'], valid['l2'], valid['l3']]
        valid_y = valid['y']
        test_traces = [test['l0'], test['l1'], test['l2'], test['l3']]
        test_y = test['y']
    else:
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_mix_samples(path, dataset, model_name, mix_method)
        model = load_t_model(path, model_name, dataset)
        layer_names = get_layers(model)
        attention_layer = get_attention_layer(layer_names)
        train_traces, valid_traces, test_traces = [], [], []
        for layer_name in attention_layer:
            layer = model.get_layer(layer_name)
            temp_model = Model(inputs=model.input, outputs=layer.output)
            layer_train_trace = temp_model.predict(train_x, batch_size=batch_size, verbose=0)
            layer_valid_trace = temp_model.predict(valid_x, batch_size=batch_size, verbose=0)
            layer_test_trace = temp_model.predict(test_x, batch_size=batch_size, verbose=0)
            train_traces.append(layer_train_trace)
            valid_traces.append(layer_valid_trace)
            test_traces.append(layer_test_trace)
        np.savez(train_path, l0=train_traces[0], l1=train_traces[1], l2=train_traces[2], l3=train_traces[3], y=train_y,
                 x=train_x)
        np.savez(valid_path, l0=valid_traces[0], l1=valid_traces[1], l2=valid_traces[2], l3=valid_traces[3], y=valid_y,
                 x=valid_x)
        np.savez(test_path, l0=test_traces[0], l1=test_traces[1], l2=test_traces[2], l3=test_traces[3], y=test_y,
                 x=test_x)
    return train_traces, valid_traces, test_traces, train_y, valid_y, test_y


def load_pretrained_vec(path, dataset, model_name, vec_dim, mix_method, batch_size=256):
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mix_samples(path, dataset, model_name, mix_method)
    pre_trained_model_list = ['alexnet', 'vgg16', 'resnet101', 'inception_v3']
    train_vec, valid_vec, test_vec = [], [], []
    for pre_model_name in pre_trained_model_list:
        save_dir = os.path.join(path, '4layer_trace', model_name, "_".join(mix_method))
        vec_path = os.path.join(save_dir, f'vec_{pre_model_name}_{vec_dim}.npz')
        if os.path.exists(vec_path):
            vector = np.load(vec_path)
            m_train_vec, m_valid_vec, m_test_vec = vector['train'], vector['valid'], vector['test']
        else:
            model = load_pretrained_model(vec_dim, pre_model_name)
            model_save_path = os.path.join(save_dir, f'{pre_model_name}.pkl')
            torch.cuda.empty_cache()
            m_train_vec = dataset_2_vec(train_x, model, batch_size)
            m_valid_vec = dataset_2_vec(valid_x, model, batch_size)
            m_test_vec = dataset_2_vec(test_x, model, batch_size)
            torch.save(model, model_save_path)
            np.savez(vec_path, train=m_train_vec, valid=m_valid_vec, test=m_test_vec)
        train_vec.append(m_train_vec)
        valid_vec.append(m_valid_vec)
        test_vec.append(m_test_vec)
    train_vec = np.stack(train_vec, axis=2)
    valid_vec = np.stack(valid_vec, axis=2)
    test_vec = np.stack(test_vec, axis=2)
    return train_vec, valid_vec, test_vec


def load_test_trace(path, dataset, model_name, adv_method, adv_eps, mix_method, vec_dim=50):
    save_dir = os.path.join(path, 'check_4layer', model_name, adv_method, str(adv_eps))
    test_path = os.path.join(save_dir, 'test.npz')
    test = np.load(test_path)
    test_traces = [test['l0'], test['l1'], test['l2'], test['l3']]
    test_y = test['y']
    _, _, _, _, _, test_x, _ = load_detector_samples(path, dataset, model_name, adv_method, adv_eps, True)
    pre_train_model_dir = os.path.join(path, 'check_4layer', model_name, "_".join(mix_method))
    pre_trained_model_list = ['alexnet', 'vgg16', 'resnet101', 'inception_v3']
    test_vec = []
    for pre_model_name in pre_trained_model_list:
        vec_path = os.path.join(pre_train_model_dir, f'{pre_model_name}_{adv_method}.npy')
        if os.path.exists(vec_path):
            m_test_vec = np.load(vec_path)
        else:
            pre_train_model_path = os.path.join(pre_train_model_dir, f'{pre_model_name}.pkl')
            model = torch.load(pre_train_model_path)
            m_test_vec = dataset_2_vec(test_x, model, batch_size=256)
            np.save(vec_path, m_test_vec)
        test_vec.append(m_test_vec)
    test_vec = np.stack(test_vec, axis=2)
    return test_traces, test_y, test_vec


def load_l_test_trace(path, model_name, adv_method, adv_eps, vec_dim=50):
    save_dir = os.path.join(path, '4layer_trace', model_name, adv_method, str(adv_eps))
    test_path = os.path.join(save_dir, 'test.npz')
    test = np.load(test_path)
    test_traces = [test['l0'], test['l1'], test['l2'], test['l3']]
    test_y = test['y']

    pre_trained_model_list = ['alexnet', 'vgg16', 'resnet101', 'inception_v3']
    test_vec = []
    for pre_model_name in pre_trained_model_list:
        save_dir = os.path.join(path, 'check_4layer', model_name, adv_method, str(adv_eps))
        vec_path = os.path.join(save_dir, f'vec_{pre_model_name}_{vec_dim}.npz')
        vector = np.load(vec_path)
        m_test_vec = vector['test']
        test_vec.append(m_test_vec)
    test_vec = np.stack(test_vec, axis=2)
    return test_traces, test_y, test_vec
