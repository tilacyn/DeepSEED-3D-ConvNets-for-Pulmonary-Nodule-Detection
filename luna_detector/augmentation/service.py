import numpy as np
from config_training import config as config
from os.path import join as opjoin
import os
import pandas as pd
from prepare import savenpy_luna, load_itk_image

class AugmentationService:
    def __init__(self, augmented_data_path, annos_path=None):
        self.augmented_data_path = augmented_data_path
        # annos_path = opjoin(self.)
        annos_path = config['luna_label'] if annos_path is None else annos_path
        self.annos = pd.read_csv(annos_path).to_numpy()


    def load_np(self, scan_id):
        path2scan = opjoin(self.augmented_data_path, 'generated_{}.mhd.npy'.format(scan_id))
        scan = np.load(path2scan)
        # base_prp = config['preprocess-result-path']
        # get_property_path = lambda prop : opjoin(base_prp, '{}_{}.npy'.format(scan_id, prop))
        # spacing = np.load(get_property_path('spacing'))
        # origin = np.load(get_property_path('origin'))
        # is_flip = False
        mask, origin, spacing, is_flip  = load_itk_image(opjoin(config['luna_segment'], '{}.mhd'.format(scan_id)))
        return scan, origin, spacing, is_flip, mask


    # preprocess augmented
    def preprocess_augmented_data(self, augmented_prp=None):
        augmented_scan_paths = [path for path in os.listdir(self.augmented_data_path) if path.endswith('mhd.npy')]
        scan_ids = [path[-11:-8] for path in augmented_scan_paths]
        augmented_prp = 'data/augmented-preprocess-result-path' if augmented_prp is None else augmented_prp
        for scan_id in scan_ids:
            savenpy_luna(scan_id, self.annos, None, None, None, augmented_prp, self.load_np)
