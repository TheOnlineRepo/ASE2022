import torch.nn as nn
import numpy as np
import math
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn.functional as F


class LayerDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, X, Y):
        self.images = np.transpose(X, [0, 3, 1, 2])
        self.labels = Y.astype('long')

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


class LayerModel(nn.Module):
    def __init__(self, layer_num, num_classes=1):
        super(LayerModel, self).__init__()
        channel = [16, 16, 32, 64][layer_num]
        out_channel = min(channel * 2, 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, out_channel, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())
        self.fc1 = nn.Linear(64 * 4 * 4, num_classes)
        self.fc2 = nn.Linear(64 * 3 * 3, num_classes)
        self.fc3 = nn.Linear(64 * 2 * 2, num_classes)

    def forward(self, layer_num, x):
        out = self.conv1(x)
        if layer_num < 3:
            out = self.mp1(out)
        out = self.conv2(out)
        if layer_num < 2:
            out = self.mp2(out)
        out = self.conv3(out)
        # out = self.conv4(out)
        out = out.reshape(out.size(0), -1)
        if layer_num < 2:
            out = self.fc1(out)
        elif layer_num == 2:
            out = self.fc2(out)
        else:
            out = self.fc3(out)
        return out


def layer_model_evalute(model, test_loader, layer_num, criterion):
    outputs, labels = [], []
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(test_loader):
            images, batch_labels = images.cuda(), batch_labels.cuda()
            batch_outputs = model(layer_num, images)
            outputs.append(batch_outputs)
            labels.append(batch_labels)
    valid_outputs = torch.cat(outputs, dim=0)
    valid_label = torch.cat(labels, dim=0)
    valid_acc = int(torch.sum(torch.argmax(valid_outputs, dim=1) == valid_label).cpu()) / valid_label.shape[0]
    valid_loss = criterion(valid_outputs, valid_label)
    return valid_acc, valid_loss


def attention_model_evalute(model, test_loader, criterion):
    outputs, labels = [], []
    with torch.no_grad():
        for i, (trace0, trace1, trace2, trace3, vector, batch_labels) in enumerate(test_loader):
            trace0, trace1, trace2, trace3 = trace0.cuda(), trace1.cuda(), trace2.cuda(), trace3.cuda()
            batch_labels = batch_labels.cuda()
            vector = vector.cuda()
            batch_outputs = model(trace0, trace1, trace2, trace3, vector)
            outputs.append(batch_outputs)
            labels.append(batch_labels)
    valid_outputs = torch.cat(outputs, dim=0)
    valid_label = torch.cat(labels, dim=0)
    valid_acc = int(torch.sum(torch.argmax(valid_outputs, dim=1) == valid_label).cpu()) / valid_label.shape[0]
    valid_loss = criterion(valid_outputs, valid_label)
    return valid_acc, valid_loss


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
        w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp


class AttentionDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, traces, vec, label):
        self.trace_0 = np.transpose(traces[0], [0, 3, 1, 2])
        self.trace_1 = np.transpose(traces[1], [0, 3, 1, 2])
        self.trace_2 = np.transpose(traces[2], [0, 3, 1, 2])
        self.trace_3 = np.transpose(traces[3], [0, 3, 1, 2])
        self.labels = label.astype('long')
        self.vec = vec

    def __getitem__(self, index):
        return self.trace_0[index], self.trace_1[index], self.trace_2[index], self.trace_3[index], self.vec[index], \
               self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


class AttentionModel(nn.Module):

    def __init__(self, spp_pool_size, num_classes=1000):
        super(AttentionModel, self).__init__()
        spp_dim = np.sum(np.power(np.array(spp_pool_size), 2)) * 64
        self.spp_pool_size = spp_pool_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.fc1 = nn.Linear(spp_dim, num_classes)
        self.fc2 = nn.Linear(num_classes, 2)

    def forward(self, trace0, trace1, trace2, trace3, img_vec, batch_size=128):
        vec0 = self.conv1(trace0)
        vec0 = spatial_pyramid_pool(vec0, vec0.shape[0], [int(vec0.size(2)), int(vec0.size(3))], self.spp_pool_size)
        vec0 = self.fc1(vec0)

        vec1 = self.conv1(trace1)
        vec1 = spatial_pyramid_pool(vec1, vec0.shape[0], [int(vec1.size(2)), int(vec1.size(3))], self.spp_pool_size)
        vec1 = self.fc1(vec1)

        vec2 = self.conv2(trace2)
        vec2 = spatial_pyramid_pool(vec2, vec0.shape[0], [int(vec2.size(2)), int(vec2.size(3))], self.spp_pool_size)
        vec2 = self.fc1(vec2)

        vec3 = self.conv3(trace3)
        vec3 = spatial_pyramid_pool(vec3, vec0.shape[0], [int(vec3.size(2)), int(vec3.size(3))], self.spp_pool_size)
        vec3 = self.fc1(vec3)
        vector = torch.stack([vec0, vec1, vec2, vec3], 2)

        w0 = torch.sum(vec0 * img_vec, dim=1)
        w1 = torch.sum(vec1 * img_vec, dim=1)
        w2 = torch.sum(vec2 * img_vec, dim=1)
        w3 = torch.sum(vec3 * img_vec, dim=1)
        weight = torch.stack([w0, w1, w2, w3], 1) / np.sqrt(img_vec.shape[1])
        weight = F.softmax(weight, dim=1).unsqueeze(dim=2)

        vector = torch.matmul(vector, weight).squeeze()
        out = self.fc2(vector)
        return out


class LinerModel(nn.Module):

    def __init__(self, spp_pool_size, num_classes=1000):
        super(LinerModel, self).__init__()
        spp_dim = np.sum(np.power(np.array(spp_pool_size), 2)) * 64
        self.spp_pool_size = spp_pool_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.fc1 = nn.Linear(spp_dim, num_classes)
        self.fc2 = nn.Linear(num_classes * 4, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, trace0, trace1, trace2, trace3, img_vec, batch_size=128):
        vec0 = self.conv1(trace0)
        vec0 = spatial_pyramid_pool(vec0, vec0.shape[0], [int(vec0.size(2)), int(vec0.size(3))], self.spp_pool_size)
        vec0 = self.fc1(vec0)

        vec1 = self.conv1(trace1)
        vec1 = spatial_pyramid_pool(vec1, vec0.shape[0], [int(vec1.size(2)), int(vec1.size(3))], self.spp_pool_size)
        vec1 = self.fc1(vec1)

        vec2 = self.conv2(trace2)
        vec2 = spatial_pyramid_pool(vec2, vec0.shape[0], [int(vec2.size(2)), int(vec2.size(3))], self.spp_pool_size)
        vec2 = self.fc1(vec2)

        vec3 = self.conv3(trace3)
        vec3 = spatial_pyramid_pool(vec3, vec0.shape[0], [int(vec3.size(2)), int(vec3.size(3))], self.spp_pool_size)
        vec3 = self.fc1(vec3)
        vector = torch.cat([vec0, vec1, vec2, vec3], 1)

        out = self.fc2(vector)
        out = self.fc3(out)
        return out


class MultiAttentionModel(nn.Module):

    def __init__(self, spp_pool_size, num_classes=1000):
        super(MultiAttentionModel, self).__init__()
        spp_dim = np.sum(np.power(np.array(spp_pool_size), 2)) * 64
        self.spp_pool_size = spp_pool_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.fc1 = nn.Linear(spp_dim, num_classes)
        self.fc2 = nn.Linear(num_classes * 4, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, trace0, trace1, trace2, trace3, img_vec, batch_size=128):
        vec0 = self.conv1(trace0)
        vec0 = spatial_pyramid_pool(vec0, vec0.shape[0], [int(vec0.size(2)), int(vec0.size(3))], self.spp_pool_size)
        vec0 = self.fc1(vec0)

        vec1 = self.conv1(trace1)
        vec1 = spatial_pyramid_pool(vec1, vec0.shape[0], [int(vec1.size(2)), int(vec1.size(3))], self.spp_pool_size)
        vec1 = self.fc1(vec1)

        vec2 = self.conv2(trace2)
        vec2 = spatial_pyramid_pool(vec2, vec0.shape[0], [int(vec2.size(2)), int(vec2.size(3))], self.spp_pool_size)
        vec2 = self.fc1(vec2)

        vec3 = self.conv3(trace3)
        vec3 = spatial_pyramid_pool(vec3, vec0.shape[0], [int(vec3.size(2)), int(vec3.size(3))], self.spp_pool_size)
        vec3 = self.fc1(vec3)
        vector = torch.stack([vec0, vec1, vec2, vec3], 2)
        out = []
        for i in range(img_vec.shape[2]):
            w0 = torch.sum(vec0 * img_vec[:, :, i], dim=1)
            w1 = torch.sum(vec1 * img_vec[:, :, i], dim=1)
            w2 = torch.sum(vec2 * img_vec[:, :, i], dim=1)
            w3 = torch.sum(vec3 * img_vec[:, :, i], dim=1)
            weight = torch.stack([w0, w1, w2, w3], 1) / np.sqrt(img_vec.shape[1])
            weight = F.softmax(weight, dim=1).unsqueeze(dim=2)
            out.append(torch.matmul(vector, weight).squeeze())
        out = torch.cat(out, dim=1)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
