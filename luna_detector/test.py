import os
from importlib import import_module
from os.path import join as opjoin

import numpy as np
import torch
from data_loader import LungNodule3Ddetector
from layers_se import *
from lidc_dataset import LIDCDataset
from torch.utils.data import DataLoader

from config_training import config as config_training


class Test:
    def __init__(self, data_path=None, thr=0, path_to_model='', start=0, end=0, iou_threshold=0.5, r_rand=0.9):
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
        self.iou_threshold = iou_threshold
        self.config = config
        self.loss = loss
        self.gp = GetPBB(config)
        self.thr = thr
        self.start = start
        self.end = end

        luna_test = np.load('./luna_test.npy')
        dataset = LungNodule3Ddetector(self.data_path, luna_test, self.config, start=0, end=0, r_rand=r_rand)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True)
        self.outputs = []
        self.targets = []
        for i, tpl in enumerate(data_loader):
            data = torch.autograd.Variable(tpl[0].cuda())
            target = torch.autograd.Variable(tpl[1].cuda())
            coord = torch.autograd.Variable(tpl[2].cuda())
            data = data.type(torch.cuda.FloatTensor)
            coord = coord.type(torch.cuda.FloatTensor)
            print(data.shape)
            print(coord.shape)

            output = self.net(data, coord)
            self.outputs.append(output.cpu().detach().numpy()[0])
            self.targets.append(target.cpu().detach().numpy()[0])



    def test_luna(self, thr, iou_threshold):
        return self.common_test(thr=thr, iou_threshold=iou_threshold)

    def common_test(self, thr, iou_threshold):
        tn = 0
        tp = 0
        n = 0
        p = 0
        fp_bbox = 0
        total_pred_len = 0
        dices = []
        target_volumes = []
        output_volumes = []
        for output, target in zip(self.outputs, self.targets):

            pred = self.gp(output, thr)
            true = self.gp(target, 0.8)
            current_dice = 0
            if len(true) > 0:
                p += 1
                correct_positive = 0
                current_dices = []
                for bbox in pred:
                    cdice = dice(bbox[1:], true[0][1:])
                    if cdice > iou_threshold:
                        correct_positive = 1
                    else:
                        fp_bbox += 1
                    # print('dice: {}'.format(cdice))
                    current_dices.append([bbox[4], dice(bbox[1:], true[0][1:])])
                if len(pred) == 0:
                    current_dices.append([-1, 0])
                current_dices = np.array(current_dices)
                if len(pred) > 0:
                    tp += 1
                max_dice_idx = np.argmax(current_dices[:, 1])
                current_dice = current_dices[max_dice_idx][1]
                r = current_dices[max_dice_idx][0]
                dices.append(current_dice)
                target_volumes.append(true[0][4])
                output_volumes.append(r)
                total_pred_len += len(pred)
            else:
                n += 1
                if len(pred) == 0:
                    tn += 1

            print('pred: {}'.format(pred))
            print('true: {}'.format(true))
            print(tp, tn, p, n, fp_bbox, current_dice)
        return [tp, tn, p, n, fp_bbox, total_pred_len], dices