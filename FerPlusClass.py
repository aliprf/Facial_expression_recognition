from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, ExpressionCodesAffectnet
from config import LearningConfig, InputDataSize, DatasetName, FerPlusConf, DatasetType

import numpy as np
import os
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray, savez_compressed
import csv
from skimage.io import imread
import pickle
import csv
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
import random
import cv2
from skimage.feature import hog
from skimage import data, exposure
from matplotlib.path import Path
from scipy import ndimage, misc
from data_helper import DataHelper
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from shutil import copyfile
from dataset_class import CustomDataset
from dataset_dynamic import DynamicDataset


class FerPlus:

    def __init__(self, ds_type):
        """we set the parameters needed during the whole class:
        """
        self.ds_type = ds_type
        if ds_type == DatasetType.train:
            self.img_path = FerPlusConf.no_aug_train_img_path
            self.anno_path = FerPlusConf.no_aug_train_annotation_path
            self.img_path_aug = FerPlusConf.aug_train_img_path
            self.anno_path_aug = FerPlusConf.aug_train_annotation_path
            self.masked_img_path = FerPlusConf.aug_train_masked_img_path
            self.orig_image_path = FerPlusConf.orig_image_path_train

        elif ds_type == DatasetType.test:
            self.img_path = FerPlusConf.test_img_path
            self.anno_path = FerPlusConf.test_annotation_path
            self.img_path_aug = FerPlusConf.test_img_path
            self.anno_path_aug = FerPlusConf.test_annotation_path
            self.masked_img_path = FerPlusConf.test_masked_img_path
            self.orig_image_path = FerPlusConf.orig_image_path_train

    def create_from_orig(self):
        print('create_from_orig & relabel to affectNetLike--->')
        """
        labels are from 1-7, but we save them from 0 to 6

        :param ds_type:
        :return:
        """

        '''read the text file, and save exp, and image'''
        dhl = DataHelper()

        exp_affectnet_like_lbls = [6, 5, 4, 1, 0, 2, 3]
        lbl_affectnet_like_lbls = ['angry/', 'disgust/', 'fear/', 'happy/', 'neutral/', 'sad/', 'surprise/']

        for exp_index in range(len(lbl_affectnet_like_lbls)):
            exp_prefix = lbl_affectnet_like_lbls[exp_index]
            for i, file in tqdm(enumerate(os.listdir(self.orig_image_path + exp_prefix))):
                if file.endswith(".jpg") or file.endswith(".png"):
                    img_source_address = self.orig_image_path + exp_prefix + file
                    img_dest_address = self.img_path + file
                    exp_dest_address = self.anno_path + file[:-4]
                    exp = exp_affectnet_like_lbls[exp_index]

                    img = np.array(Image.open(img_source_address))
                    res_img = resize(img, (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                                     anti_aliasing=True)

                    im = Image.fromarray(np.round(res_img * 255.0).astype(np.uint8))
                    '''save image'''
                    im.save(img_dest_address)
                    '''save annotation'''
                    np.save(exp_dest_address + '_exp', exp)

    def create_synthesized_landmarks(self, model_file, test_print=False):
        dhl = DataHelper()
        model = tf.keras.models.load_model(model_file)
        for i, file in tqdm(enumerate(os.listdir(self.img_path))):
            if file.endswith(".jpg") or file.endswith(".png"):
                dhl.create_synthesized_landmarks_path(img_path=self.img_path,
                                                      anno_path=self.anno_path, file=file,
                                                      model=model,
                                                      test_print=test_print)

    def upsample_data_fix_rate(self):
        """  [4965 7215  4830     3171     4097      463     3955]
            Augmentation: sample_count_by_category:      ====>>
            [2,      1,     2,     3,       2,      16,     3]
            [9930,  7215,   9660,  9513,  8194,    7408,  11865]
        """

        dhl = DataHelper()
        ''''''
        aug_factor_by_class = [2, 1, 3, 4, 17, 7, 8]
        sample_count_by_class = np.zeros([7])
        img_addr_by_class = [[] for i in range(7)]
        anno_addr_by_class = [[] for i in range(7)]
        lnd_addr_by_class = [[] for i in range(7)]

        for i, file in tqdm(enumerate(os.listdir(self.anno_path))):
            if file.endswith("_exp.npy"):
                exp = int(np.load(os.path.join(self.anno_path, file)))
                sample_count_by_class[exp] += 1
                '''adding ex'''
                anno_addr_by_class[exp].append(os.path.join(self.anno_path, file))
                img_addr_by_class[exp].append(os.path.join(self.img_path, file[:-8] + '.jpg'))
                lnd_addr_by_class[exp].append(os.path.join(self.anno_path, file[:-8] + '_slnd.npy'))

        print("sample_count_by_category: ====>>")
        print(sample_count_by_class)

        for i in range(len(anno_addr_by_class)):
            dhl.do_random_augment(img_addrs=img_addr_by_class[i],
                                  anno_addrs=anno_addr_by_class[i],
                                  lnd_addrs=lnd_addr_by_class[i],
                                  aug_factor=aug_factor_by_class[i],
                                  aug_factor_freq=None,
                                  img_save_path=self.img_path_aug,
                                  anno_save_path=self.anno_path_aug,
                                  class_index=i
                                  )

    def report(self, aug=True):
        dhl = DataHelper()
        if aug:
            anno_path = self.anno_path_aug
            img_path = self.img_path_aug
        else:
            anno_path = self.anno_path
            img_path = self.img_path

        sample_count_by_class = np.zeros([7])
        for i, file in tqdm(enumerate(os.listdir(anno_path))):
            if file.endswith("_exp.npy"):
                exp = np.load(os.path.join(anno_path, file))
                sample_count_by_class[exp] += 1
        print("sample_count_by_category: ====>>")
        print(sample_count_by_class)
