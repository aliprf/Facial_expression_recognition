from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, ExpressionCodesAffectnet
from config import LearningConfig, InputDataSize, DatasetName, AffectnetConf, DatasetType

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
from skimage import transform
from skimage.transform import resize
import tensorflow as tf
import random
import cv2
from skimage.feature import hog
from skimage import data, exposure
from matplotlib.path import Path
from scipy import ndimage, misc
from skimage.transform import SimilarityTransform, AffineTransform
from skimage.draw import rectangle
from skimage.draw import line, set_color
from tf_augmnetation import TFAugmentation
from data_helper import DataHelper

class DynamicDataset:

    def create_dataset(self, img_filenames, anno_names, lnd_filenames, is_validation=False,
                       ds=DatasetName.affectnet):

        dhl = DataHelper()

        def get_img(file_name, lnd):
            path = bytes.decode(file_name)
            image_raw = tf.io.read_file(path)
            img = tf.image.decode_image(image_raw, channels=3)
            img = tf.cast(img, tf.float32)/255.0

            up_mask, mid_mask, bot_mask = dhl.create_spatial_mask(img=img, lnd=lnd[0])

            up_mask = tf.cast(up_mask, dtype=tf.float32)
            mid_mask = tf.cast(mid_mask, dtype=tf.float32)
            bot_mask = tf.cast(bot_mask, dtype=tf.float32)

            up_mask = img * tf.stack([up_mask, up_mask, up_mask], -1)
            mid_mask = img * tf.stack([mid_mask, mid_mask, mid_mask], -1)
            bot_mask = img * tf.stack([bot_mask, bot_mask, bot_mask], -1)

            '''as we don't flip or rotate the landmark points, we should first create the mask.
                then we augment all independently. I know if we rotate or flip the parts independently,
                 they might not be the same as the main image, but we want each branch to be trained independently.'''
            if is_validation or tf.random.uniform([]) <= 0.5:
                return img, up_mask, mid_mask, bot_mask

            '''main image'''
            img = self._do_augment(img)
            up_mask = self._do_augment(up_mask)
            mid_mask = self._do_augment(mid_mask)
            bot_mask = self._do_augment(bot_mask)
            ''''''
            return img, up_mask, mid_mask, bot_mask

        def get_lbl(anno_name):
            path = bytes.decode(anno_name)
            lbl = load(path)
            return lbl

        def get_lnd(lnd_name):
            path = bytes.decode(lnd_name)
            lnd = load(path)
            return lnd

        def wrap_get_img(img_filename, anno_name, lnd_name):
            lnd = tf.numpy_function(get_lnd, [lnd_name], [tf.double])

            img, up_mask, mid_mask, bot_mask = tf.numpy_function(get_img, [img_filename, lnd],
                                                                 [tf.float32, tf.float32, tf.float32, tf.float32])

            if is_validation and ds == DatasetName.affectnet:
                lbl = tf.numpy_function(get_lbl, [anno_name], [tf.string])
            else:
                lbl = tf.numpy_function(get_lbl, [anno_name], [tf.int64])

            return img, up_mask, mid_mask, bot_mask, lbl

        epoch_size = len(img_filenames)

        img_filenames = tf.convert_to_tensor(img_filenames, dtype=tf.string)
        anno_names = tf.convert_to_tensor(anno_names)
        lnd_filenames = tf.convert_to_tensor(lnd_filenames)

        dataset = tf.data.Dataset.from_tensor_slices((img_filenames, anno_names, lnd_filenames))
        dataset = dataset.shuffle(epoch_size)
        dataset = dataset.map(wrap_get_img, num_parallel_calls=32)\
            .batch(LearningConfig.batch_size)\
            .prefetch(10)
        return dataset

    def _do_augment(self, img):
        tf_aug = TFAugmentation()
        img = tf_aug.color(img)
        img = tf_aug.random_invert_img(img)
        img = tf_aug.random_quality(img)
        img = tf_aug.random_zoom(img)
        img = tf_aug.flip(img)
        return img