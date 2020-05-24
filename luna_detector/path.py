from os.path import join as opjoin

mydrive_path = '/content/drive/My Drive'
base_path = opjoin(mydrive_path, 'DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection')
ctgan_base_path = opjoin(mydrive_path, 'CT-GAN')
augmented_data_path = opjoin(ctgan_base_path, 'generated')
data_path = '/content/drive/My Drive/dsb2018_topcoders/data'
luna_path = opjoin(base_path, 'luna_detector')
augmented_prp = opjoin(base_path, 'data/augmented-preprocess-result-path')
segment_path = opjoin(base_path, 'data/luna-segment/seg-lungs-LUNA16')
annos_path = opjoin(luna_path, 'labels/annos.csv')