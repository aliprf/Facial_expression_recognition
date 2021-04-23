from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, ExpressionCodesAffectnet
from config import LearningConfig, InputDataSize, DatasetName, RafDBConf, DatasetType

import numpy as np
import os
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
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


class RafDB:
    def __init__(self, ds_type):
        """we set the parameters needed during the whole class:
        """
        self.ds_type = ds_type
        if ds_type == DatasetType.train:
            self.img_path = RafDBConf.no_aug_train_img_path
            self.anno_path = RafDBConf.no_aug_train_annotation_path
            self.img_path_aug = RafDBConf.aug_train_img_path
            self.anno_path_aug = RafDBConf.aug_train_annotation_path

        elif ds_type == DatasetType.test:
            self.img_path = RafDBConf.test_img_path
            self.anno_path = RafDBConf.test_img_path

    def create_from_orig(self, ds_type):
        """
        labels are from 1-7, but we save them from 0 to 6

        :param ds_type:
        :return:
        """
        if ds_type == DatasetType.train:
            txt_path = RafDBConf.orig_annotation_txt_path
            load_img_path = RafDBConf.orig_image_path
            save_img_path = RafDBConf.no_aug_train_img_path
            save_anno_path = RafDBConf.no_aug_train_annotation_path
            prefix = 'train'
        elif ds_type == DatasetType.test:
            txt_path = RafDBConf.orig_annotation_txt_path
            load_img_path = RafDBConf.orig_image_path
            save_img_path = RafDBConf.test_img_path
            save_anno_path = RafDBConf.test_annotation_path
            prefix = 'test'

        '''read the text file, and save exp, and image'''
        file1 = open(txt_path, 'r')
        while True:
            line = file1.readline()
            if not line:
                break
            f_name = line.split(' ')[0]
            img_source_address = load_img_path + f_name[:-4] + '_aligned.jpg'
            img_dest_address = save_img_path + f_name
            exp = int(line.split(' ')[1]) - 1
            '''padd, resize image and save'''
            img = np.array(Image.open(img_source_address))
            # padd
            fix_pad = InputDataSize.image_input_size * 0.1
            img = np.pad(img, ((fix_pad, fix_pad), (fix_pad, fix_pad), (0, 0)), 'wrap')
            # resize
            res_img = resize(img, (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                             anti_aliasing=True)
            im = Image.fromarray(np.round(res_img*255.0).astype(np.uint8))
            im.save(img_dest_address)
            '''save annotation'''
            np.save(save_anno_path + f_name, exp)

        file1.close()
