import os
from importlib import import_module
from os.path import join as opjoin

import numpy as np
import torch
from data_loader import LungNodule3Ddetector
from layers_se import *
from patient_data_loader import PatientDataLoader
from torch.utils.data import DataLoader
from abc import abstractmethod, ABC

from config_training import config as config_training

default_data_path = os.path.join('/content/drive/My Drive/DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection',
                                   config_training['preprocess_result_path'])
base_path = '/content/drive/My Drive/DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection'
luna_path = opjoin(base_path, 'luna_detector')


class AbstractTest:
    def __init__(self, data_path=None, path_to_model='', start=0, end=0):
        self.data_path = default_data_path if data_path is None else data_path
        model = import_module('res18_se')
        print('creating model')
        config, net, loss, get_pbb = model.get_model()
        net = net.cuda()
        loss = loss.cuda()
        checkpoint = torch.load(opjoin(luna_path, 'test_results', path_to_model))
        net.load_state_dict(checkpoint['state_dict'])
        self.net = net
        self.config = config
        self.loss = loss
        self.gp = GetPBB(config)
        self.start = start
        self.end = end
        dataset = self.create_dataset()
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True)

        self.outputs, self.targets = self.predict_on_data(data_loader)

    @abstractmethod
    def create_dataset(self):
        pass

    def predict_on_data(self, data_loader):
        outputs, targets = [], []
        for i, (data, target, coord) in enumerate(data_loader):
            data, target, coord = data.cuda(), target.cuda(), coord.cuda()
            data = data.type(torch.cuda.FloatTensor)
            coord = coord.type(torch.cuda.FloatTensor)
            print(data.shape)
            print(coord.shape)

            output = self.net(data, coord)
            outputs.append(output.cpu().detach().numpy()[0])
            targets.append(target)
        return outputs, [self.transform_target(target) for target in targets]

    @abstractmethod
    def transform_target(self, target):
        pass

    @abstractmethod
    def is_positive(self, target):
        pass

    def common_test(self, threshold):
        tn, tp, n, p = 0, 0, 0, 0
        for output, target in zip(self.outputs, self.targets):
            pred = self.gp(output, threshold)
            true = self.gp(target, 0.8)
            if self.is_positive(target):
                p += 1
                if len(pred) > 0:
                    tp += 1
            else:
                n += 1
                if len(pred) == 0:
                    tn += 1
            print('pred: {}'.format(pred))
            print('true: {}'.format(true))
            print(tp, tn, p, n)
        return [tp, tn, p, n]



class SimpleTest(AbstractTest):

    def create_dataset(self):
        luna_test = np.load('./luna_test.npy')
        dataset = LungNodule3Ddetector(self.data_path, luna_test, self.config, start=0, end=0, r_rand=0.9)
        return dataset


    def is_positive(self, target):
        return len(self.gp(target, 0.8)) > 0

    def transform_target(self, target):
        return target.cpu().detach().numpy()[0]

    def test_luna(self, threshold):
        return self.common_test(threshold=threshold)


class PatientTest(AbstractTest):

    def create_dataset(self):
        luna_test = np.load('./luna_test.npy')
        dataset = PatientDataLoader(self.data_path, luna_test, self.config, start=0, end=0)
        return dataset

    def transform_target(self, target):
        return target.cpu().detach().numpy()[0]


    def is_positive(self, target):
        return target

    def test_luna(self, threshold):
        return self.common_test(threshold=threshold)
