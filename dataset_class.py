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

class CustomDataset:

    def create_dataset(self, file_names_face, file_names_eyes, file_names_nose, file_names_mouth, anno_names,
                       is_validation=False, ds=DatasetName.affectnet):
        tf_aug = TFAugmentation()

        def get_img(file_name):
            path = bytes.decode(file_name)  # called when use dataset since dataset is generator
            img = load(path)['arr_0']
            if is_validation or tf.random.uniform([]) <= 0.6:
                return img

            '''split'''
            img_main = img[:, :, :3]
            # img_mask = img[:, :, 3:]
            img_mask_1 = tf.image.grayscale_to_rgb(tf.expand_dims(img[:, :, 3], axis=2))
            img_mask_2 = tf.image.grayscale_to_rgb(tf.expand_dims(img[:, :, 4], axis=2))
            '''main image'''
            img_main = tf_aug.color(img_main)
            img_main = tf_aug.random_invert_img(img_main)
            '''masks'''
            img_mask_1 = tf_aug.random_invert_img(img_mask_1, p=0.2)
            img_mask_1 = tf.image.rgb_to_grayscale(tf_aug.color(img_mask_1))
            #
            img_mask_2 = tf_aug.random_invert_img(img_mask_2, p=0.5)
            img_mask_2 = tf.image.rgb_to_grayscale(tf_aug.color(img_mask_2))
            ''''''
            # img = tf.concat([img_main, img_mask], axis=2)
            img = tf.concat([img_main, img_mask_1, img_mask_2], axis=2)
            return img

        def get_lbl(anno_name):
            path = bytes.decode(anno_name)
            lbl = load(path)
            return lbl

        def wrap_get_img(file_name_face, file_name_eyes, file_name_nose,
                         file_name_mouth, anno_name):
            img_face = tf.numpy_function(get_img, [file_name_face], [tf.double])
            img_eyes = tf.numpy_function(get_img, [file_name_eyes], [tf.double])
            img_nose = tf.numpy_function(get_img, [file_name_nose], [tf.double])
            img_mouth = tf.numpy_function(get_img, [file_name_mouth], [tf.double])

            if is_validation and ds == DatasetName.affectnet:
                lbl = tf.numpy_function(get_lbl, [anno_name], [tf.string])
            else:
                lbl = tf.numpy_function(get_lbl, [anno_name], [tf.int64])

            return img_face, img_eyes, img_nose, img_mouth, lbl

        epoch_size = len(file_names_face)

        file_names_face = tf.convert_to_tensor(file_names_face, dtype=tf.string)
        file_names_eyes = tf.convert_to_tensor(file_names_eyes, dtype=tf.string)
        file_names_nose = tf.convert_to_tensor(file_names_nose, dtype=tf.string)
        file_names_mouth = tf.convert_to_tensor(file_names_mouth, dtype=tf.string)
        anno_names = tf.convert_to_tensor(anno_names)

        dataset = tf.data.Dataset.from_tensor_slices((file_names_face, file_names_eyes, file_names_nose,
                                                      file_names_mouth, anno_names))
        dataset = dataset.shuffle(epoch_size)
        dataset = dataset.map(wrap_get_img, num_parallel_calls=32)\
            .batch(LearningConfig.batch_size)\
            .prefetch(10)
        return dataset

    # def create_dataset(self, imgs_path, annotations_path):
    #     def map_fn(image_path, anno_path):
    #         label = load(anno_path)
    #         image = load(image_path)['arr_0']
    #         return image, label
    #
    #     epoch_size = len(imgs_path)
    #     imgs_path = tf.convert_to_tensor(imgs_path, dtype=tf.string)
    #     annotations_path = tf.convert_to_tensor(annotations_path, dtype=tf.string)
    #
    #     dataset = tf.data.Dataset.from_tensor_slices((imgs_path, annotations_path))
    #     dataset = dataset.repeat().shuffle(epoch_size)
    #     #'''''''''''''
    #     dataset = dataset.map(map_fn, num_parallel_calls=8)
    #     dataset = dataset.batch(LearningConfig.batch_size)
    #     # try one of the following
    #     dataset = dataset.prefetch(5)
    #     # dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))
    #
    #     images, labels = dataset.make_one_shot_iterator().get_next()
    #     return dataset.make_one_shot_iterator().get_next()


    # def create_dataset(self, image_list, label_list, epochs):
    #     def _parse_function(filename, label):
    #         image_string = tf.read_file(filename, "file_reader")
    #         image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    #         image = tf.cast(image_decoded, tf.float32)
    #         return image, label
    #
    #     dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_list), tf.constant(label_list)))
    #     dataset = dataset.shuffle(len(image_list))
    #     dataset = dataset.repeat(epochs)
    #     dataset = dataset.map(_parse_function).batch(batch_size)

