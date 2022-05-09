import numpy as np
from sklearn import cluster
from tqdm import tqdm
from adv_function.idc.utils import get_layer_outs_new
from adv_function.idc.lrp_toolbox.model_io import write, read
import keras

from sklearn.metrics import silhouette_score
import os
experiment_folder = '/media/data0/DeepSuite'
model_folder      = '/media/data0/DeepSuite/trained_models'

class ImportanceDrivenCoverage:
    def __init__(self,model, dataset, model_name, num_relevant_neurons, selected_class, subject_layer, skip_layers, train_inputs, train_labels):
        self.covered_combinations = ()

        self.model = model
        self.dataset = dataset
        self.model_name = model_name
        self.num_relevant_neurons = num_relevant_neurons
        self.selected_class = selected_class
        self.subject_layer = subject_layer
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.skip_layers = skip_layers


    def get_measure_state(self):
        return self.covered_combinations

    def set_measure_state(self, covered_combinations):
        self.covered_combinations = covered_combinations

    def test(self, test_inputs, only_last_layer):
        if isinstance(self.model.layers[self.subject_layer], keras.layers.convolutional.Conv2D): is_conv = True
        else: is_conv = False
        choosed_layer = self.subject_layer -np.sum(self.skip_layers < self.subject_layer)
        #########################
        #1.Find Relevant Neurons#
        #########################
        relevant_neurons_save_dir = os.path.join(experiment_folder, 'trace', 'idc_neuron', self.dataset)
        relevant_neurons_save_path = os.path.join(relevant_neurons_save_dir, self.model_name+'.npy')
        if os.path.exists(relevant_neurons_save_path):
            choosed_layer_R = np.load(relevant_neurons_save_path)
            relevant_neurons, least_relevant_neurons = np.argsort(choosed_layer_R)[0][::-1][:self.num_relevant_neurons], np.argsort(choosed_layer_R)[0][:self.num_relevant_neurons]
        else:
            print("Calculating Relevance scores now!")
            # Convert keras model into txt
            model_path = model_folder + '/' + self.model_name
            write(self.model, model_path, num_channels=test_inputs[0].shape[-1], fmt='keras_txt', only_last_layer=only_last_layer)

            lrpmodel = read(model_path + '.txt', 'txt')  # 99.16% prediction accuracy
            drop_layer = lrpmodel.drop_softmax_output_layer()  # drop softnax output layer for analysis
            drop_layer = 0 ## trick for resnet34
            relevant_neurons, least_relevant_neurons, choosed_layer_R, total_R = find_relevant_neurons(self.model, lrpmodel, self.train_inputs, self.train_labels, choosed_layer, self.num_relevant_neurons, is_conv, only_last_layer, drop_layer)
            if not os.path.exists(relevant_neurons_save_dir):
                os.makedirs(relevant_neurons_save_dir)
            np.save(relevant_neurons_save_path, choosed_layer_R)

        ####################################
        #2.Quantize Relevant Neuron Outputs#
        ####################################
        choosed_layer_model = Model(inputs=self.model.input, outputs=self.model.layers[choosed_layer].output)
        print("Calculating Clustering results now!")
        train_layer_outs = choosed_layer_model.predict(self.train_inputs, batch_size=32)
        qtized = quantizeSilhouette(train_layer_outs, is_conv, relevant_neurons)

        ####################
        #3.Measure coverage#
        ####################
        print("Calculating IDC coverage")
        test_layer_outs = choosed_layer_model.predict(test_inputs, batch_size=32)
        test_fitness = measure_idc(self.model, self.model_name, test_inputs, choosed_layer, relevant_neurons, self.selected_class, test_layer_outs, qtized, is_conv, self.covered_combinations)

        return test_fitness


def quantize(out_vectors, conv, relevant_neurons, n_clusters=3):
    #if conv: n_clusters+=1
    quantized_ = []

    for i in range(out_vectors.shape[-1]):
        out_i = []
        for l in out_vectors:
            if conv: #conv layer
                out_i.append(np.mean(l[...,i]))
            else:
                out_i.append(l[i])

        #If it is a convolutional layer no need for 0 output check
        if not conv: out_i = filter(lambda elem: elem != 0, out_i)
        values = []
        if not len(out_i) < 10: #10 is threshold of number positives in all test input activations
            kmeans = cluster.KMeans(n_clusters=n_clusters)
            kmeans.fit(np.array(out_i).reshape(-1, 1))
            values = kmeans.cluster_centers_.squeeze()
        values = list(values)
        values = limit_precision(values)

        #if not conv: values.append(0) #If it is convolutional layer we dont add  directly since thake average of whole filter.

        quantized_.append(values)

    quantized_ = [quantized_[rn] for rn in relevant_neurons]

    return quantized_


def quantizeSilhouette(out_vectors, conv, relevant_neurons):
    #if conv: n_clusters+=1
    quantized_ = []

    for i in range(out_vectors.shape[-1]):
        if i not in relevant_neurons: continue
        out_i = []
        for l in out_vectors:
            if conv: #conv layer
                out_i.append(np.mean(l[..., i]))
            else:
                out_i.append(l[i])

        #If it is a convolutional layer no need for 0 output check
        #if not conv: out_i = [item for item in out_i if item != 0]
        out_i = list(filter(lambda elem: elem != 0, out_i))
        values = []

        if not len(out_i) < 10: #10 is threshold of number positives in all test input activations

            clusterSize = range(2, 5)#[2, 3, 4, 5]
            clustersDict = {}
            for clusterNum in clusterSize:
                kmeans          = cluster.KMeans(n_clusters=clusterNum)
                clusterLabels   = kmeans.fit_predict(np.array(out_i).reshape(-1, 1))
                silhouetteAvg   = silhouette_score(np.array(out_i).reshape(-1, 1), clusterLabels)
                clustersDict [silhouetteAvg] = kmeans

            maxSilhouetteScore = max(clustersDict.keys())
            bestKMean          = clustersDict[maxSilhouetteScore]

            values = bestKMean.cluster_centers_.squeeze()
        values = list(np.sort(values))
        values = limit_precision(values)
        #if not conv: values.append(0) #If it is convolutional layer we dont add  directly since thake average of whole filter.
        if len(values) == 0: values.append(0)

        quantized_.append(values)
    #quantized_ = [quantized_[rn] for rn in relevant_neurons]
    return quantized_

def limit_precision(values, prec=2):
    limited_values = []
    for v in values:
        limited_values.append(round(v,prec))

    return limited_values


def determine_quantized_cover(lout, quantized):
    covered_comb = []
    for idx, l in enumerate(lout):
        #if l == 0:
        #    covered_comb.append(0)
        #else:
        closest_q = min(quantized[idx], key=lambda x:abs(x-l))
        covered_comb.append(closest_q)

    return covered_comb


def measure_idc(model, model_name, test_inputs, subject_layer,
                                   relevant_neurons, sel_class,
                                   test_layer_outs, qtized, is_conv,
                                   covered_combinations=()):
    if is_conv:
        test_relevant_outs = test_layer_outs[..., relevant_neurons]
    else:
        test_relevant_outs = test_layer_outs[:, relevant_neurons]
    test_fitness = []
    for i in range(len(qtized)):
        test_layer_fitness = np.zeros(test_inputs.shape[0])
        test_layer_traces = test_relevant_outs[:, i]
        for j in range(len(qtized[i]) - 1, 0, -1):
            m_mean = np.mean((qtized[i][j], qtized[i][j-1]))
            condition = (test_layer_fitness == 0) * (test_layer_traces > m_mean)
            test_layer_fitness[condition] = j
        test_layer_fitness = keras.utils.to_categorical(test_layer_fitness, num_classes=len(qtized[i]))
        test_fitness.append(test_layer_fitness)
    # subject_layer = subject_layer - 1
    test_fitness = np.concatenate(test_fitness, axis=1)
    return test_fitness


from keras.models import Model
def find_relevant_neurons(kerasmodel, lrpmodel, inps, outs, choosed_layer, num_rel, is_conv, only_last_layer, drop_layer):

    totalR = None
    cnt = 0
    traces = [inps]
    if only_last_layer:
        end = len(kerasmodel.layers)
        choosed_layer = 0
    else:
        end = -1
    for layer in kerasmodel.layers[:end]:
        layer_model = Model(inputs=kerasmodel.input, outputs=layer.output)
        trace = layer_model.predict(inps)
        traces.append(trace)
    for id, inp in enumerate(tqdm(inps)):
        cnt += 1
        if only_last_layer:
            ypred = kerasmodel.predict(inp[np.newaxis, ...])
        else:
            ypred = lrpmodel.forward(np.expand_dims(inp, axis=0))
        inp_trace = [traces[layer_num][id] for layer_num in range(len(traces) - drop_layer)]

        mask = np.zeros_like(ypred)
        mask[:, np.argmax(ypred)] = 1
        Rinit = ypred*mask

        R_inp, R_all = lrpmodel.lrp(Rinit, inp_trace, only_last_layer=only_last_layer)

        if totalR:
            for idx, elem in enumerate(totalR):
                totalR[idx] = elem + R_all[idx]

        else: totalR = R_all

    #      THE MOST RELEVANT                               THE LEAST RELEVANT'
    if is_conv:
        choosed_layer_R = np.mean(totalR[choosed_layer], axis=(1,2))
    else:
        choosed_layer_R = totalR[choosed_layer]
    return np.argsort(choosed_layer_R)[0][::-1][:num_rel], np.argsort(choosed_layer_R)[0][:num_rel], choosed_layer_R, totalR
    # return np.argsort(final_relevants)[0][::-1][:num_rel], np.argsort(final_relevants)[0][:num_rel], totalR



