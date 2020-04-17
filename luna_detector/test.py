os.chdir(luna_path)
import glob
from importlib import import_module

import numpy as np
from lidc_dataset import LIDCDataset
from torch.utils.data import DataLoader
import torch
from layers_se import *

from config_training import config as config_training
from scipy.spatial.distance import euclidean


class Test:
    def __init__(self, thr, data_path, model_path):
        model = import_module('res18_se')
        print('creating model')
        config, net, loss, get_pbb = model.get_model()
        net = net.cuda()
        loss = loss.cuda()
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['state_dict'])
        self.net = net
        self.config = config
        self.loss = loss
        self.gp = GetPBB(config)
        self.thr = thr
        self.data_path = data_path

    def test(self, start=0, end=50):
        dataset = LIDCDataset(self.data_path, self.config, start, end, load=True, isRand=True, phase='test')
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True)
        tn = 0
        tp = 0
        n = 0
        p = 0
        for i, (data, target, coord, tgt) in enumerate(data_loader):
            data = torch.autograd.Variable(data.cuda())
            target = torch.autograd.Variable(target.cuda())
            coord = torch.autograd.Variable(coord.cuda())
            data = data.type(torch.cuda.FloatTensor)
            coord = coord.type(torch.cuda.FloatTensor)

            output = self.net(data, coord)
            pred = self.gp(output.cpu().detach().numpy()[0], self.thr)
            true = self.gp(target.cpu().detach().numpy()[0], 0.8)
            if len(true) > 0:
                p += 1
                correct_positive = False
                for bbox in pred:
                    if euclidean(bbox[1:], true[0][1:]) < 2:
                        tp += 1
            else:
                n += 1
                if len(pred) == 0:
                    tn += 1

            print('pred: {}'.format(pred))
            print('true: {}'.format(true))
            print(tp, tn, p, n)
        return [tp, tn, p, n]
