import glob
import os
from os.path import join as opjoin

import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import pydicom as dicom
import numpy as np
from data_loader import LabelMapping
from data_loader import Crop
from numpy import array as na
import cv2

from PIL import Image, ImageDraw

from matplotlib import pyplot as plt


def parseXML(scan_path):
    '''
    parse xml file
    args:
    xml file path
    output:
    nodule list
    [{nodule_id, roi:[{z, sop_uid, xy:[[x1,y1],[x2,y2],...]}]}]
    '''
    file_list = os.listdir(scan_path)
    xml_file = None
    for file in file_list:
        if '.' in file and file.split('.')[1] == 'xml':
            xml_file = file
            break
    prefix = "{http://www.nih.gov}"
    if xml_file is None:
        print('SCAN PATH: {}'.format(scan_path))
    tree = ET.parse(scan_path + '/' + xml_file)
    root = tree.getroot()
    readingSession_list = root.findall(prefix + "readingSession")
    nodules = []

    for session in readingSession_list:
        # print(session)
        unblinded_list = session.findall(prefix + "unblindedReadNodule")
        for unblinded in unblinded_list:
            nodule_id = unblinded.find(prefix + "noduleID").text
            edgeMap_num = len(unblinded.findall(prefix + "roi/" + prefix + "edgeMap"))
            if edgeMap_num >= 1:
                # it's segmentation label
                nodule_info = {}
                nodule_info['nodule_id'] = nodule_id
                nodule_info['roi'] = []
                roi_list = unblinded.findall(prefix + "roi")
                for roi in roi_list:
                    roi_info = {}
                    # roi_info['z'] = float(roi.find(prefix + "imageZposition").text)
                    roi_info['sop_uid'] = roi.find(prefix + "imageSOP_UID").text
                    roi_info['xy'] = []
                    edgeMap_list = roi.findall(prefix + "edgeMap")
                    for edgeMap in edgeMap_list:
                        x = float(edgeMap.find(prefix + "xCoord").text)
                        y = float(edgeMap.find(prefix + "yCoord").text)
                        xy = [x, y]
                        roi_info['xy'].append(xy)
                    nodule_info['roi'].append(roi_info)
                nodules.append(nodule_info)
    return nodules


def make_mask(image, image_id, nodules):
    height, width = image.shape
    # print(image.shape)
    filled_mask = np.full((height, width), 0, np.uint8)
    contoured_mask = np.full((height, width), 0, np.uint8)
    # todo OR for all masks
    for nodule in nodules:
        for roi in nodule['roi']:
            if roi['sop_uid'] == image_id:
                edge_map = roi['xy']
                cv2.fillPoly(filled_mask, np.int32([np.array(edge_map)]), 255)
                # cv2.polylines(contoured_mask, np.int32([np.array(edge_map)]), color=255, isClosed=False)

    # mask = np.swapaxes(np.array([contoured_mask, filled_mask]), 0, 2)
    # cv2.imwrite('kek0.jpg', image)
    # cv2.imwrite('kek1.jpg', filled_mask)
    return np.reshape(filled_mask, (height, width, 1))


def create_map_from_nodules(nodules):
    id2roi = {}
    for nodule in nodules:
        for roi in nodule['roi']:
            id = roi['sop_uid']
            if id not in id2roi:
                id2roi[id] = []
            id2roi[id].append(roi['xy'])
    return id2roi


def get_mask(image, image_id, nodules)


def make_mask_for_rgb(image, image_id, nodules):
    filled_mask = image
    for nodule in nodules:
        for roi in nodule['roi']:
            if roi['sop_uid'] == image_id:
                edge_map = roi['xy']
                cv2.fillPoly(filled_mask, np.int32([np.array(edge_map)]), (255, 0, 0))
    return filled_mask


def imread(image_path):
    ds = dicom.dcmread(image_path)
    img = ds.pixel_array
    img_2d = img.astype(float)
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
    img_2d_scaled = np.uint8(img_2d_scaled)
    image = img_2d_scaled
    return image, ds


def resolve_bbox(dcms, id2roi):
    nodule_coordinates = []
    for i, image, dcm_data in enumerate(dcms):
        rois = id2roi[dcm_data.SOPInstanceUID]
        roi = rois[0]
        mean = np.mean(roi, axis=0)
        nodule_coordinates.append([i, mean[0], mean[1]])
    return np.concatenate((np.mean(nodule_coordinates, axis=0), [5.0]))


class LIDCDataset(Dataset):
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.ids = []
        self.create_ids()
        self.label_mapping = LabelMapping(config, 'test')
        self.crop = Crop(config)

    def __getitem__(self, idx):
        dcms = []
        parent_path = self.ids[idx]
        for file in os.listdir(parent_path):
            if not file.endswith('dcm'):
                continue
            image, dcm_data = imread(opjoin(parent_path, file))
            # image = np.reshape(image, (image.shape[0], image.shape[1], 1))
            # image = np.repeat(image, 3, axis=2)
            if not has_slice_location(dcm_data):
                continue
            dcms.append((image, dcm_data))
        dcms.sort(key=lambda dcm: dcm[1].SliceLocation)
        nodules = parseXML(parent_path)
        id2roi = create_map_from_nodules(nodules)
        imgs = na([dcm[0] for dcm in dcms])
        bbox = resolve_bbox(na(dcms), id2roi)
        sample, target, bboxes, coord = self.crop(imgs, bbox, [bbox], isScale=False, isRand=True)

        label = self.label_mapping(sample.shape[1:], target, bboxes)
        sample = (sample.astype(np.float32) - 128) / 128

        return torch.from_numpy(sample), \
               torch.from_numpy(label), \
               coord

    def __len__(self):
        return len(self.ids)

    def create_ids(self):
        for root, _, files in os.walk(self.data_path):
            if glob.glob(opjoin(self.data_path, root, '*xml')):
                self.ids.append(root)
            if len(self.ids) > 30:
                break


def has_slice_location(dcm_data):
    try:
        slice_location = dcm_data.SliceLocation
        return True
    except:
        # print('No Slice Location')
        return False
