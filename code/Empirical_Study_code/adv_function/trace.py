from pathlib import Path
from adv_function.sa import _get_kdes, _get_lsa, get_sc, find_closest_at
from adv_function.idc.idc import *
from adv_function.idc.utils import *


def get_layers(model, skip=False):
    if skip:
        layer_names = [layer.name for layer in model.layers if 'bn' not in layer.name and 'relu' not in layer.name and 'concat' not in layer.name and 'batch' not in layer.name and 'activation' not in layer.name and 'flatten' not in layer.name and 'input' not in layer.name]
    else:
        layer_names = [layer.name for layer in model.layers if 'flatten' not in layer.name and 'input' not in layer.name]
    return layer_names


def scale_output(layer_output, rmax=1, rmin=0):
    '''given the outputs of a layer, scale the output of each sample into [0,1]'''
    for i, opt in enumerate(layer_output):
        if not opt.max() == opt.min:
            std = (opt - opt.min()) / (opt.max() - opt.min())
            layer_output[i] = std * (rmax - rmin) + rmin
        else:
            layer_output[i] -= opt.min()
    return layer_output


def get_trace_boundary(path, dataset, model_name):
    '''get the boundary of training test cases'''
    filepath1 = os.path.join(path, 'trace_boundary', dataset, model_name + '_traces_low.npy')
    filepath2 = os.path.join(path, 'trace_boundary', dataset, model_name + '_traces_high.npy')
    traces_low = np.load(filepath1)
    traces_high = np.load(filepath2)
    return traces_low, traces_high


def cov2vec(output):
    '''convert the output of convulutional layer into a 1-D vector'''
    vector = []
    # from (28,28,4) to (4,)
    for j in range(output.shape[-1]):
        vector.append(np.mean(output[..., j]))
    vector = np.array(vector)
    return vector


def cal_samples_trace(model, dataset, fitness, batch_size=128):
    layer_names = get_layers(model)
    traces = []
    for layer_name in tqdm(layer_names):
        layer = model.get_layer(layer_name)
        temp_model = Model(inputs=model.input, outputs=layer.output)
        layer_output = temp_model.predict(dataset, batch_size=batch_size, verbose=0)
        if fitness == 'nc':
            layer_output = scale_output(layer_output)

        if layer_output[0].ndim == 3:
            layer_vector = list(map(cov2vec, layer_output))
        else:
            layer_vector = layer_output
        layer_vector = np.array(layer_vector)
        traces.append(layer_vector)
    traces = np.concatenate(traces, axis=1)
    return traces


def load_traces(path, dataset, model, adv_method, fitness, attack_eps=0.3):
    if fitness == 'nc':
        fitness_suf = '_nc.npy'
    else:
        fitness_suf = '.npy'
    if attack_eps == 0.3:
        attack_suf = '.npy'
    else:
        attack_suf = f'_{attack_eps}.npy'
    split_dir = os.path.join(path, 'adv_dataset', dataset, model, 'train_test_split_new')
    train_origin_id = np.load(os.path.join(split_dir, 'train_id.npy'))
    test_origin_id = np.load(os.path.join(split_dir, 'test_id.npy'))
    origin_trace_path = os.path.join(path, 'trace', model + fitness_suf)
    origin_trace = np.load(origin_trace_path)
    train_origin = origin_trace[train_origin_id]
    test_origin = origin_trace[test_origin_id]

    adv_trace_dir = os.path.join(path, 'adv_trace', dataset, model, adv_method)
    train_adv = np.load(os.path.join(adv_trace_dir, 'train' + fitness_suf[:-4] + attack_suf))
    test_adv = np.load(os.path.join(adv_trace_dir, 'test' + fitness_suf[:-4] + attack_suf))

    train_x = np.concatenate((train_origin, train_adv))
    train_y = np.concatenate((np.zeros(train_origin.shape[0]), np.ones(train_adv.shape[0])))
    test_x = np.concatenate((test_origin, test_adv))
    test_y = np.concatenate((np.zeros(test_origin.shape[0]), np.ones(test_adv.shape[0])))
    return train_x, train_y, test_x, test_y


def nc(trace, threshold):
    return trace > threshold


def kmnc(trace, k, traces_low, traces_high):
    fitness = []
    unit = (traces_high - traces_low) / float(k)
    for i in range(trace.shape[1]):
        neuron_value = trace[:, i]
        neuron_value = (neuron_value - traces_low[i]) / unit[i]
        neuron_value = np.nan_to_num(neuron_value)
        fall = np.zeros((trace.shape[0], k)).astype("bool")
        neuron_value[np.where(neuron_value < 0)] = 0
        neuron_value[np.where(neuron_value >= k)] = k - 1
        for j in range(trace.shape[0]):
            fall[j][int(neuron_value[j])] = True
        fitness.append(fall)
    fitness = np.concatenate(fitness, axis=1)
    return fitness


def nbc(trace, traces_low, traces_high):
    fitness = np.zeros((trace.shape[0], trace.shape[1] * 2))
    for i in range(trace.shape[1]):
        fitness[:, i * 2] = trace[:, i] > traces_high[i]
        fitness[:, i * 2 + 1] = trace[:, i] < traces_low[i]
    return fitness


def snac(trace, traces_high):
    fitness = np.zeros_like(trace)
    for i in range(trace.shape[1]):
        fitness[:, i] = trace[:, i] > traces_high[i]
    return fitness


def tknc(trace, t, model, skip):
    layers = get_layers(model, skip=skip)
    # layers = [layer for layer in model.layers if 'flatten' not in layer.name and 'input' not in layer.name]
    fitness = np.zeros_like(trace)
    check = 0
    for i in range(len(layers)):
        layer_neuron_num = int(model.get_layer(layers[i]).output.shape[-1])
        layer_split_start = check
        layer_split_end = check + layer_neuron_num
        layer_trace = trace[:, layer_split_start:layer_split_end]
        layer_trace_rank = np.argsort(-layer_trace, axis=1)
        for j in range(trace.shape[0]):
            k = min(t, layer_neuron_num)
            fitness[j][layer_split_start + layer_trace_rank[j][:k]] = 1
        check += layer_neuron_num
    return fitness


def get_param_list(fitness):
    if fitness == 'nc':
        return [0.75, 0.5]
    if fitness == 'kmnc':
        return [10, 20, 50]
    if fitness == 'tknc':
        return [1, 2, 3]
    else:
        return [None]


def load_fitness(path, dataset, model, adv_method, fitness, param=None, attack_eps=0.3):
    if fitness == 'trace':
        train_x, train_y, test_x, test_y = load_traces(path, dataset, model, adv_method, fitness, attack_eps)
    else:
        if attack_eps == 0.3:
            attack_suf = '.npy'
        else:
            attack_suf = f'_{attack_eps}.npy'
        param_dir = Path(os.path.join(path, 'adv_fitness', dataset, model, adv_method, fitness, str(param)))
        train_x = np.load(param_dir.joinpath("train_x"+attack_suf))
        train_y = np.load(param_dir.joinpath("train_y"+attack_suf))
        test_x = np.load(param_dir.joinpath("test_x"+attack_suf))
        test_y = np.load(param_dir.joinpath("test_y"+attack_suf))
    return train_x, train_y, test_x, test_y


def load_fitness_combine(path, dataset, model, adv_method, fitness, param=None, attack_eps=0.3):
    train_x, train_y, test_x, test_y = load_fitness(path, dataset, model, adv_method, fitness, param, attack_eps)
    tknc_train_x, tknc_train_y, tknc_test_x, tknc_test_y = load_fitness(path, dataset, model, adv_method, "tknc", 1, attack_eps)
    tknc_train_x = tknc_train_x.astype('bool')
    tknc_test_x = tknc_test_x.astype('bool')
    train_x = np.concatenate((train_x, tknc_train_x), axis=1)
    test_x = np.concatenate((test_x, tknc_test_x), axis=1)
    return train_x, train_y, test_x, test_y


def get_layer_bound(model, sa_layer, layer_num=None):
    layer_names = get_layers(model)
    if layer_num == None:
        if sa_layer == 0:
            layer_name = layer_names[0]
        elif sa_layer == 1:
            layer_name = layer_names[int(len(layer_names) / 2)]
        elif sa_layer == 2:
            layer_name = layer_names[-1]
    else:
        layer_name = layer_names[layer_num]
    start, end = 0, 0
    for layer in layer_names:
        if layer == layer_name:
            end = start + model.get_layer(layer).output_shape[-1]
            break
        else:
            start += model.get_layer(layer).output_shape[-1]
    return start, end


def lsa(trace, y_test, train_trace, y_train, n_bucket, is_classification, start, end, num_classes=10):
    class_matrix = {}
    trace = trace[:, start:end]
    train_trace = train_trace[:, start:end]
    if is_classification:
        for i, label in enumerate(y_train):
            if label not in class_matrix:
                class_matrix[label] = []
            class_matrix[label].append(i)
    kdes, removed_cols = _get_kdes(train_trace, y_train, class_matrix, is_classification, num_classes=num_classes)
    if kdes is not None:
        lsa = []
        if is_classification:
            for i, at in enumerate(tqdm(trace)):
                label = y_test[i]
                kde = kdes[label]
                neuron_lsa = _get_lsa(kde, at, removed_cols)
                lsa.append(neuron_lsa)
        else:
            kde = kdes[0]
            for at in tqdm(trace):
                lsa.append(_get_lsa(kde, at, removed_cols))
        lsa = np.array(lsa)
        return np.array(lsa)
    else:
        return None


def dsa(trace, y_test, train_trace, y_train, n_bucket, start, end):
    trace = trace[:, start:end]
    train_trace = train_trace[:, start:end]
    class_matrix = {}
    all_idx = []
    for i, label in enumerate(y_train):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)
    dsa = []
    for i, at in enumerate(tqdm(trace)):
        label = y_test[i]
        a_dist, a_dot = find_closest_at(at, train_trace[class_matrix[label]])
        b_dist, _ = find_closest_at(
            a_dot, train_trace[list(set(all_idx) - set(class_matrix[label]))]
        )
        dsa.append(a_dist / b_dist)
    return np.array(dsa)


def idc(model, dataset, model_name, x_train, y_train, x_test, y_test, num_relevant, selected_class=-1, only_last_layer=False):

    trainable_layers = get_trainable_layers(model)
    non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
    print('Trainable layers: ' + str(trainable_layers))
    print('Non trainable layers: ' + str(non_trainable_layers))
    subject_layer = trainable_layers[-1]
    if model_name == 'MobileNet':
        subject_layer -= 1

    skip_layers = [] #SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)
    skip_layers = np.array(skip_layers)
    print(skip_layers)

    X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model, x_train, y_train)
    cc = ImportanceDrivenCoverage(model, dataset, model_name, num_relevant, selected_class, subject_layer, skip_layers, X_train_corr, Y_train_corr)
    test_fitness = cc.test(x_test, only_last_layer)
    return test_fitness