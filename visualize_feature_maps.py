from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, RafDBConf

from cnn_model import CNNModel
from custom_loss import CustomLosses
from data_helper import DataHelper

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import csv
from skimage.io import imread
import pickle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
from keras.models import Model


class FeatureMapVisualizer:
    def __init__(self, weight_path, dataset_name):
        self.weight_path = weight_path
        if dataset_name == DatasetName.rafdb:
            self.img_path = RafDBConf.aug_train_img_path
            self.annotation_path = RafDBConf.aug_train_annotation_path
        elif dataset_name == DatasetName.affectnet:
            self.img_path = AffectnetConf.aug_train_img_path_7
            self.annotation_path = AffectnetConf.aug_train_annotation_path_7

    def visualize(self):
        model = tf.keras.models.load_model(self.weight_path)
        model.summary()

        model_face = Model(inputs=model.inputs, outputs=model.get_layer('face_block_3_expand').output)
        model_eyes = Model(inputs=model.inputs, outputs=model.get_layer('eyes_block_3_expand').output)
        model_nose = Model(inputs=model.inputs, outputs=model.get_layer('nose_block_3_expand').output)
        model_mouth = Model(inputs=model.inputs, outputs=model.get_layer('mouth_block_3_expand').output)

        for i, file in tqdm(enumerate(os.listdir(self.img_path))):
            img_orig = np.array(Image.open(self.img_path + file))
            up_mask = np.array(Image.open(self.annotation_path + 'spm/' + file[:-4] + '_spm_up.jpg'))
            md_mask = np.array(Image.open(self.annotation_path + 'spm/' + file[:-4] + '_spm_md.jpg'))
            bo_mask = np.array(Image.open(self.annotation_path + 'spm/' + file[:-4] + '_spm_bo.jpg'))

            img_orig = tf.expand_dims(img_orig, axis=0)
            up_mask = tf.expand_dims(up_mask, axis=0)
            md_mask = tf.expand_dims(md_mask, axis=0)
            bo_mask = tf.expand_dims(bo_mask, axis=0)

            '''predicting'''
            face_feature_maps = model_face.predict([img_orig, up_mask, md_mask, bo_mask])
            eyes_feature_maps = model_eyes.predict([img_orig, up_mask, md_mask, bo_mask])
            nose_feature_maps = model_nose.predict([img_orig, up_mask, md_mask, bo_mask])
            mouth_feature_maps = model_mouth.predict([img_orig, up_mask, md_mask, bo_mask])
            '''visualizing'''
            self.print_feature_map(img=img_orig, index=i, fmap=face_feature_maps[:,:,:,:10], prefix='face')
            self.print_feature_map(img=up_mask, index=i, fmap=eyes_feature_maps[:,:,:,:10], prefix='eyes')
            self.print_feature_map(img=md_mask, index=i, fmap=nose_feature_maps[:,:,:,:10], prefix='nose')
            self.print_feature_map(img=bo_mask, index=i, fmap=mouth_feature_maps[:,:,:,:10], prefix='mouth')
            # if i > 0 : break

    def print_feature_map(self, img, index, fmap, prefix):
        # fmap = fmap[-1, :, :, :]
        f_min, f_max = fmap.min(), fmap.max()

        filters = (fmap - f_min) / (f_max - f_min)
        # filters = fmap

        dpi = 500
        width = 100  # filters.shape[1]
        height = 100  # filters.shape[2]

        figsize = width * 10 * (filters.shape[3]+1)/float(dpi), (height * 10) / float(dpi)
        fig, axs = plt.subplots(1, filters.shape[3]+1, figsize=figsize)
        axs[0].imshow(img[-1, :, :], vmin=np.amin(img), vmax=np.amax(img))
        axs[0].axis('off')

        for i in range(filters.shape[3]):
            filt = filters[-1, :, :, i]
            # axs[i+1].imshow(filt, vmin=np.amin(filt), vmax=np.amax(filt))
            axs[i+1].imshow(filt)
            axs[i+1].axis('off')
        plt.tight_layout()
        plt.savefig('z_' + prefix + '_' + str(index) + '.png')
        # plt.clf()

        # img_mean = np.mean(abs(fmap[-1, :, :, :]), axis=2)
        # plt.figure()
        # plt.imshow(img_mean, vmin=np.amin(img_mean), vmax=np.amax(img_mean))
        # plt.tight_layout(pad=0)
        # plt.rcParams["figure.figsize"] = (48, 48)
        # plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
        #
        # plt.savefig('z_' + prefix + '_' + str(index) + '.png', bbox_inches='tight', dpi=400, pad_inches=0)
        # plt.clf()




