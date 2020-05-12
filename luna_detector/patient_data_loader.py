from torch.utils.data import Dataset
from data_loader import get_filenames_and_labels, LabelMapping
import os
import numpy as np


class PatientDataLoader(Dataset):
    def __init__(self, data_dir, split_path, config, split_comber=None, start=0, end=0, r_rand=None):
        sizelim = config['sizelim'] / config['reso']
        sizelim2 = config['sizelim2'] / config['reso']
        sizelim3 = config['sizelim3'] / config['reso']
        self.split_comber = split_comber
        # idcs = np.load(split_path)
        idcs = split_path
        self.filenames, self.sample_bboxes = get_filenames_and_labels(idcs, start, end, data_dir)
        self.bboxes = []

        for i, l in enumerate(self.sample_bboxes):
            if len(l) > 0:
                for t in l:
                    if t[3] > sizelim:
                        self.bboxes += [[np.concatenate([[i], t])]]
                    if t[3] > sizelim2:
                        self.bboxes += [[np.concatenate([[i], t])]] * 2
                    if t[3] > sizelim3:
                        self.bboxes += [[np.concatenate([[i], t])]] * 4
        self.bboxes = np.concatenate(self.bboxes, axis=0)
        self.label_mapping = LabelMapping(config, 'train')


    def __getitem__(self, idx):
        bbox = self.bboxes[idx]
        filename = self.filenames[int(bbox[0])]
        imgs = np.load(filename)
        bboxes = self.sample_bboxes[int(bbox[0])]
        return imgs, bboxes, bbox

