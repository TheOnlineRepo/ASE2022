import h5py
import datetime
import random
import numpy as np
from keras import backend as K
from keras import models
from adv_function.idc.lrp_toolbox.model_io import read

random.seed(123)
np.random.seed(123)

def get_layer_outs_old(model, class_specific_test_set):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    # Testing
    layer_outs = [func([class_specific_test_set, 1.]) for func in functors]

    return layer_outs


def get_layer_outs(model, test_input, skip=[]):
    inp = model.input  # input placeholder
    outputs = [layer.output for index, layer in enumerate(model.layers) \
               if index not in skip]

    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layer_outs = [func([test_input]) for func in functors]

    return layer_outs


def get_layer_outs_new(model, inputs, skip=[]):
    # TODO: FIX LATER. This is done for solving incompatibility in Simos' computer
    # It is a shortcut.
    # skip.append(0)
    evaluater = models.Model(inputs=model.input,
                             outputs=[layer.output for index, layer in enumerate(model.layers) \
                                      if index not in skip])

    # Insert some dummy value in the beginning to avoid messing with layer index
    # arrangements in the main flow
    # outs = evaluater.predict(inputs)
    # outs.insert(0, inputs)

    # return outs

    return evaluater.predict(inputs)


def calc_major_func_regions(model, train_inputs, skip=None):
    if skip is None:
        skip = []

    outs = get_layer_outs_new(model, train_inputs, skip=skip)

    major_regions = []

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        layer_out = layer_out.mean(axis=tuple(i for i in range(1, layer_out.ndim - 1)))

        major_regions.append((layer_out.min(axis=0), layer_out.max(axis=0)))

    return major_regions


def get_layer_outputs_by_layer_name(model, test_input, skip=None):
    if skip is None:
        skip = []

    inp = model.input  # input placeholder
    outputs = {layer.name: layer.output for index, layer in enumerate(model.layers)
               if (index not in skip and 'input' not in layer.name)}  # all layer outputs (except input for functionals)
    functors = {name: K.function([inp], [out]) for name, out in outputs.items()}  # evaluation functions

    layer_outs = {name: func([test_input]) for name, func in functors.items()}
    return layer_outs


def get_layer_inputs(model, test_input, skip=None, outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs(model, test_input)

    inputs = []

    for i in range(len(outs)):
        weights, biases = model.layers[i].get_weights()

        inputs_for_layer = []

        for input_index in range(len(test_input)):
            inputs_for_layer.append(
                np.add(np.dot(outs[i - 1][0][input_index] if i > 0 else test_input[input_index], weights), biases))

        inputs.append(inputs_for_layer)

    return [inputs[i] for i in range(len(inputs)) if i not in skip]


def filter_correct_classifications(model, X, Y):
    X_corr = []
    Y_corr = []
    X_misc = []
    Y_misc = []
    preds = model.predict(X)  # np.expand_dims(x,axis=0))

    for idx, pred in enumerate(preds):
        if np.argmax(pred) == np.argmax(Y[idx]):
            X_corr.append(X[idx])
            Y_corr.append(Y[idx])
        else:
            X_misc.append(X[idx])
            Y_misc.append(Y[idx])

    '''
    for x, y in zip(X, Y):
        if np.argmax(p) == np.argmax(y):
            X_corr.append(x)
            Y_corr.append(y)
        else:
            X_misc.append(x)
            Y_misc.append(y)
    '''

    return np.array(X_corr), np.array(Y_corr), np.array(X_misc), np.array(Y_misc)


def filter_val_set(desired_class, X, Y):
    """
    Filter the given sets and return only those that match the desired_class value
    :param desired_class:
    :param X:
    :param Y:
    :return:
    """
    X_class = []
    Y_class = []
    for x, y in zip(X, Y):
        if y[desired_class] == 1:
            X_class.append(x)
            Y_class.append(y)
    print("Validation set filtered for desired class: " + str(desired_class))
    return np.array(X_class), np.array(Y_class)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def get_trainable_layers(model):
    trainable_layers = []
    for idx, layer in enumerate(model.layers):
        try:
            if 'input' not in layer.name and 'softmax' not in layer.name and \
                    'pred' not in layer.name and 'drop' not in layer.name :
                weights = layer.get_weights()[0]
                trainable_layers.append(model.layers.index(layer))
        except:
            pass

    return trainable_layers


def weight_analysis(model):
    threshold_weight = 0.1
    deactivatables = []
    for i in range(2, target_layer + 1):
        for k in range(model.layers[i - 1].output_shape[1]):
            neuron_weights = model.layers[i].get_weights()[0][k]
            deactivate = True
            for j in range(len(neuron_weights)):
                if neuron_weights[j] > threshold_weight:
                    deactivate = False

            if deactivate:
                deactivatables.append((i, k))

    return deactivatables


def percent_str(part, whole):
    return "{0}%".format(float(part) / whole * 100)


def find_relevant_pixels(inputs, model_path, lrpmethod, relevance_percentile):
    lrpmodel = read(model_path + '.txt', 'txt')  # 99.16% prediction accuracy
    lrpmodel.drop_softmax_output_layer()  # drop softnax output layer for analysis

    all_relevant_pixels = []

    for inp in inputs:
        ypred = lrpmodel.forward(np.expand_dims(inp, axis=0))

        mask = np.zeros_like(ypred)
        mask[:, np.argmax(ypred)] = 1
        Rinit = ypred * mask

        if lrpmethod == 'simple':
            R_inp, R_all = lrpmodel.lrp(Rinit)  # as Eq(56) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'epsilon':
            R_inp, R_all = lrpmodel.lrp(Rinit, 'epsilon', 0.01)  # as Eq(58) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'alphabeta':
            R_inp, R_all = lrpmodel.lrp(Rinit, 'alphabeta', 3)  # as Eq(60) from DOI: 10.1371/journal.pone.0130140

        if 'lenet' in model_path.lower():
            R_inp_flat = R_inp.reshape(28 * 28)
        elif 'cifar' in model_path.lower():
            R_inp_flat = R_inp.reshape(32 * 32 * 3)
        else:
            R_inp_flat = R_inp.reshape(100 * 100 * 3)

        abs_R_inp_flat = np.absolute(R_inp_flat)

        relevance_threshold = np.percentile(abs_R_inp_flat, relevance_percentile)
        # if relevance_threshold < 0: relevance_threshold = 0

        s = datetime.datetime.now()
        if 'lenet' in model_path.lower():
            R_inp = np.absolute(R_inp.reshape(28, 28))
        elif 'cifar' in model_path.lower():
            R_inp = np.absolute(R_inp.reshape(32, 32, 3))
        else:
            R_inp = np.absolute(R_inp.reshape(100, 100, 3))

        relevant_pixels = np.where(R_inp > relevance_threshold)
        all_relevant_pixels.append(relevant_pixels)
    return all_relevant_pixels


def save_relevant_pixels(filename, relevant_pixels):
    with h5py.File(filename + '_relevant_pixels.h5', 'a') as hf:
        group = hf.create_group('gg')
        for i in range(len(relevant_pixels)):
            group.create_dataset("relevant_pixels_" + str(i), data=relevant_pixels[i])

    print("Relevant pixels saved to %s" % (filename))
    return


def load_relevant_pixels(filename):
    try:
        with h5py.File(filename + '_relevant_pixels.h5', 'r') as hf:
            group = hf.get('gg')
            i = 0
            relevant_pixels = []
            while True:
                relevant_pixels.append(group.get('relevant_pixels_' + str(i)).value)
                i += 1
    except (AttributeError) as error:
        # because we don't know the exact number of inputs in each class
        # we leave it to iterate until it throws an attribute error, and then return
        # return relevant pixels to the caller function

        print("Relevant pixels loaded from %s" % (filename))

        return relevant_pixels
