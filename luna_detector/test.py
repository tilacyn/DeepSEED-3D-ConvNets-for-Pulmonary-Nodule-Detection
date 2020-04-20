import os
from importlib import import_module

import numpy as np
from lidc_dataset import LIDCDataset
from data_loader import LungNodule3Ddetector
from torch.utils.data import DataLoader
import torch
from layers_se import *
import time

from config_training import config as config_training
from scipy.spatial.distance import euclidean
from PIL import Image
import math
from os.path import join as opjoin


class Test:
    def __init__(self, data_path=None, thr=0, path_to_model=''):
        if data_path is None:
            self.data_path = os.path.join('/content/drive/My Drive/DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection',
                                   config_training['preprocess_result_path'])
        else:
            self.data_path = data_path
        self.base_path = '/content/drive/My Drive/DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection'
        self.luna_path = opjoin(self.base_path, 'luna_detector')
        model = import_module('res18_se')
        print('creating model')
        config, net, loss, get_pbb = model.get_model()
        net = net.cuda()
        loss = loss.cuda()
        checkpoint = torch.load(opjoin(self.luna_path, 'test_results', path_to_model))
        net.load_state_dict(checkpoint['state_dict'])
        self.net = net
        self.config = config
        self.loss = loss
        self.gp = GetPBB(config)
        self.thr = thr


    def test(self):
        dataset = LIDCDataset(self.data_path, self.config, 0, 50, phase='test')
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True)
        tn = 0
        tp = 0
        n = 0
        p = 0
        for i, (data, target, coord) in enumerate(data_loader):
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
                    if iou(bbox[1:], true[0][1:]) < 3:
                        tp += 1
                        break
            else:
                n += 1
                if len(pred) == 0:
                    tn += 1

            print('pred: {}'.format(pred))
            print('true: {}'.format(true))
            print(tp, tn, p, n)
        return [tp, tn, p, n]

    def test_luna(self):
        luna_train = np.load('./luna_train.npy')
        dataset = LungNodule3Ddetector(self.data_path, luna_train, self.config)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True)
        tn = 0
        tp = 0
        n = 0
        p = 0
        fp = 0

        # for i, (data, target, coord) in enumerate([dataset[0], dataset[0], dataset[0]]):
        #     # data = torch.autograd.Variable(data.cuda())
        #     # target = torch.autograd.Variable(target.cuda())
        #     # coord = torch.autograd.Variable(coord.cuda())
        #
        #     # print(self.gp(target.cpu().detach().numpy()[0], 0.8))
        #     print(target.shape)
        #     print(self.gp(target, 0.8))
        # return

        for i, (data, target, coord) in enumerate(data_loader):
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
                    if iou(bbox[1:], true[0][1:]) > 0.5:
                        correct_positive = True
                    else:
                        fp +=1
                tp += int(correct_positive)
            else:
                n += 1
                if len(pred) == 0:
                    tn += 1

            print('pred: {}'.format(pred))
            print('true: {}'.format(true))
            print(tp, tn, p, n, fp)
        return [tp, tn, p, n, fp]

    def validate(self, start, end):
        net = self.net
        loss = self.loss
        start_time = time.time()
        dataset = LIDCDataset(data_path, self.config, start, end, load=True, phase='train')
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True)

        net.eval()

        metrics = []
        for i, (data, target, coord, _) in enumerate(data_loader):
            data = torch.autograd.Variable(data.cuda(non_blocking=True))
            target = torch.autograd.Variable(target.cuda(non_blocking=True))
            coord = torch.autograd.Variable(coord.cuda(non_blocking=True))

            data = data.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
            coord = coord.type(torch.cuda.FloatTensor)

            output = net(data, coord)
            loss_output = loss(output, target, train=False)

            loss_output[0] = loss_output[0].item()
            metrics.append(loss_output)
        end_time = time.time()

        metrics = np.asarray(metrics, np.float32)
        print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            end_time - start_time))
        print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5])))
        print('\n')
        return np.mean(metrics[:, 0])

    def visualize(self, start, end):
        net = self.net
        loss = self.loss
        start_time = time.time()
        dataset = LIDCDataset(data_path, self.config, start, end, load=True, phase='train')
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True)

        net.eval()

        metrics = []
        ds = dataset
        for i, (data, target, coord, _) in enumerate([ds[1], ds[1], ds[1]]):
            data = torch.autograd.Variable(data.cuda(non_blocking=True))
            target = torch.autograd.Variable(target.cuda(non_blocking=True))
            # coord = torch.autograd.Variable(coord.cuda(non_blocking=True))

            self.visualize_imgs(data.cpu().detach().numpy()[0], i)
            self.visualize_masks(data.cpu().detach().numpy()[0], target.cpu().detach().numpy(), i)

            # data = data.type(torch.cuda.FloatTensor)
            # target = target.type(torch.cuda.FloatTensor)
            # coord = coord.type(torch.cuda.FloatTensor)

            # output = net(data, coord)
            # loss_output = loss(output, target, train=False)

            # loss_output[0] = loss_output[0].item()
            # metrics.append(loss_output)
        end_time = time.time()

    def visualize_imgs(self, imgs, idx):
        print(imgs.shape)
        imgs = imgs * 128 + 128
        imgs = np.uint8(imgs)
        # print(imgs)
        imgs = [np.reshape(img, (img.shape[0], img.shape[1], 1)) for img in imgs]
        imgs = [np.repeat(img, 3, axis=2) for img in imgs]
        images = [Image.fromarray(img) for img in imgs]
        images[0].save(opjoin(base_path, 'test_imgs_{}.gif'.format(idx)),
                       save_all=True, append_images=images[1:], optimize=False, duration=80, loop=0)

    def visualize_masks(self, imgs, target, idx):
        print(imgs.shape)
        imgs = imgs * 128 + 128
        imgs = np.uint8(imgs)
        bbox = self.gp(target, 0.8)[0]
        print('bbox {}'.format(bbox))
        # print(imgs)
        imgs = [np.reshape(img, (img.shape[0], img.shape[1], 1)) for img in imgs]
        imgs = [np.repeat(img, 3, axis=2) for img in imgs]
        r = bbox[-1] // 2
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    if abs(i - bbox[1]) < r and abs(j - bbox[2]) < r and abs(k - bbox[3]) < r:
                        imgs[i][j][k] = (255, 0, 0)
        images = [Image.fromarray(img) for img in imgs]
        images[0].save(opjoin(base_path, 'test_masks_{}.gif'.format(idx)),
                       save_all=True, append_images=images[1:], optimize=False, duration=80, loop=0)
