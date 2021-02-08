from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig
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

class Test:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if dataset_name == DatasetName.affectnet:
            self.img_path = AffectnetConf.revised_eval_img_path
            self.annotation_path = AffectnetConf.revised_eval_annotation_path

        # self.img_path = AffectnetConf.revised_train_img_path
        # self.annotation_path = AffectnetConf.revised_train_annotation_path


    def test_reg(self, model_file):
        dhp = DataHelper()
        model = tf.keras.models.load_model(model_file)
        filenames, val_labels = dhp.create_test_gen(img_path=self.img_path, annotation_path=self.annotation_path)
        loss_eval = self._eval_model_reg(filenames, val_labels, model)
        print('=============Evaluation=================')
        print(loss_eval)
        print('========================================')

    def test(self, model_file):
        dhp = DataHelper()
        model = tf.keras.models.load_model(model_file)

        filenames, val_labels = dhp.create_test_gen(img_path=self.img_path,
                                                    annotation_path=self.annotation_path)

        loss_eval = self._eval_model(filenames, val_labels, model)
        print('=============Evaluation=================')
        print(loss_eval)
        print('========================================')

    def _eval_model_reg(self, img_filenames, labels_filenames, model):
        dhl = DataHelper()

        gt_lbls = []
        pr_lbls = []

        for i, file_name in tqdm(enumerate(img_filenames)):
            img = np.expand_dims(np.array(imread(self.img_path + file_name)) / 255.0, axis=0)
            gt_lbls.append(dhl.load_and_categorize_valence(self.annotation_path + labels_filenames[i]))
            prediction = model(img)[0]
            pr_lbls.append(dhl.categorize_valence(prediction))
            # pr_lbls.append(np.argmax(score))
            print('Gt => ' + str(gt_lbls[i]) + ' : ' + str(pr_lbls[i]) + ' <= Pr')

        print(confusion_matrix(gt_lbls, pr_lbls))
        acc = accuracy_score(gt_lbls, pr_lbls)

        return acc

    def _eval_model(self, img_filenames, labels_filenames, model):
        dhl = DataHelper()

        gt_lbls = []
        pr_lbls = []
        i =0
        for file_name in tqdm(img_filenames):
            lbl = int(load(self.annotation_path + labels_filenames[i]))
            # print(lbl)
            # print(type(lbl))
            if lbl == 0 or lbl == 1 or lbl == 2 or lbl == 6:
                img = np.expand_dims(np.array(imread(self.img_path + file_name)) / 255.0, axis=0)
                gt_lbls.append(dhl.load_and_relabel_exp(self.annotation_path + labels_filenames[i]))
                # gt_lbls.append(dhl.load_and_categorize_valence(self.annotation_path + labels_filenames[i]))
                prediction = model(img)[0]
                score = tf.nn.softmax(prediction)
                pr_lbls.append(np.argmax(score))
                print('Gt => ' + str(gt_lbls[i]) + ' : ' + str(pr_lbls[i]) + ' <= Pr')
                i +=1

        print(confusion_matrix(gt_lbls, pr_lbls))
        acc = accuracy_score(gt_lbls, pr_lbls)

        return acc
