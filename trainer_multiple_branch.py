from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, DatasetType
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


class TrainMultiple:
    def __init__(self, dataset_name, ds_type):
        self.dataset_name = dataset_name
        self.num_of_branches = 4

        if dataset_name == DatasetName.affectnet:
            if ds_type == DatasetType.train:
                self.img_path = AffectnetConf.aug_train_img_path
                self.annotation_path = AffectnetConf.aug_train_annotation_path
            if ds_type == DatasetType.train_7:
                self.img_path = AffectnetConf.aug_train_img_path_7
                self.annotation_path = AffectnetConf.aug_train_annotation_path_7

    def train(self, arch, weight_paths):
        """"""
        c_loss = CustomLosses()
        dhp = DataHelper()

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        '''making models'''
        _lr = 1e-4
        models = self.make_models(arch=arch, w_paths=weight_paths, num_of_branches=)
        '''create optimizer'''
        optimizer = self._get_optimizer(lr=_lr)
        '''create sample generator'''

    def make_models(self, arch, w_paths, num_of_branches):
        """
        return #num_of_input_branches
        :param arch:
        :param w_paths: [] of weight paths or None
        :param num_of_branches: number of input branches
        :return: array of models
        """
        models = []
        cnn = CNNModel()
        for i in range(num_of_branches):
            model = cnn.get_model(arch=arch)
            if w_paths is not None:
                model.load_weights(w_paths[i])
            models.append(model)

        return models

    def _get_optimizer(self, lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-4):
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
