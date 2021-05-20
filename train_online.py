from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, DatasetType, RafDBConf
from cnn_model import CNNModel
from custom_loss import CustomLosses
from data_helper import DataHelper

import tensorflow as tf
# import tensorflow.keras as keras
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
import os
from AffectNetClass import AffectNet
from RafdbClass import RafDB
from dataset_dynamic import DynamicDataset


class TrainOnline:
    def __init__(self, dataset_name, ds_type, weights='imagenet', lr=1e-2):
        self.dataset_name = dataset_name
        self.ds_type = ds_type
        self.weights = weights
        self.lr = lr

        if dataset_name == DatasetName.rafdb:
            self.drop = 0.5
            self.epochs_drop = 20

            self.img_path = RafDBConf.aug_train_img_path
            self.annotation_path = RafDBConf.aug_train_annotation_path
            self.masked_img_path = RafDBConf.aug_train_masked_img_path
            self.val_img_path = RafDBConf.test_img_path
            self.val_annotation_path = RafDBConf.test_annotation_path
            self.eval_masked_img_path = RafDBConf.test_masked_img_path
            self.num_of_classes = 7
            self.num_of_samples = None

        elif dataset_name == DatasetName.affectnet:
            self.drop = 0.5
            self.epochs_drop = 5
            if ds_type == DatasetType.train:
                self.img_path = AffectnetConf.aug_train_img_path
                self.annotation_path = AffectnetConf.aug_train_annotation_path
                self.masked_img_path = AffectnetConf.aug_train_masked_img_path
                self.val_img_path = AffectnetConf.eval_img_path
                self.val_annotation_path = AffectnetConf.eval_annotation_path
                self.eval_masked_img_path = AffectnetConf.eval_masked_img_path
                self.num_of_classes = 8
                self.num_of_samples = None
            elif ds_type == DatasetType.train_7:
                self.img_path = AffectnetConf.aug_train_img_path_7
                self.annotation_path = AffectnetConf.aug_train_annotation_path_7
                self.masked_img_path = AffectnetConf.aug_train_masked_img_path_7
                self.val_img_path = AffectnetConf.eval_img_path_7
                self.val_annotation_path = AffectnetConf.eval_annotation_path_7
                self.eval_masked_img_path = AffectnetConf.eval_masked_img_path_7
                self.num_of_classes = 7
                self.num_of_samples = None

    def train(self, arch, weight_path):
        """"""

        '''create loss'''
        c_loss = CustomLosses()

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        start_train_date = datetime.now().strftime("%Y%m%d-%H%M%S")

        '''making models'''

        model = self.make_model(arch=arch, w_path=weight_path)
        '''create save path'''

        if self.dataset_name == DatasetName.affectnet:
            save_path = AffectnetConf.weight_save_path + start_train_date + '/'
        elif self.dataset_name == DatasetName.rafdb:
            save_path = RafDBConf.weight_save_path + start_train_date + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        '''create sample generator'''
        dhp = DataHelper()
        '''     Train   Generator'''
        '''     Train   Generator'''
        img_filenames, exp_filenames, spm_up_filenames, spm_md_filenames, spm_bo_filenames = \
            dhp.create_generator_full_path_with_spm(img_path=self.img_path,
                                                    annotation_path=self.annotation_path)
        dds = DynamicDataset()
        ds = dds.create_dataset(img_filenames=img_filenames,
                                spm_up_filenames=spm_up_filenames,
                                spm_md_filenames=spm_md_filenames,
                                spm_bo_filenames=spm_bo_filenames,
                                anno_names=exp_filenames)

        # global_accuracy, conf_mat = self._eval_model(model=model)

        '''create train configuration'''
        step_per_epoch = len(img_filenames) // LearningConfig.batch_size
        gradients = None
        virtual_step_per_epoch = LearningConfig.virtual_batch_size // LearningConfig.batch_size

        '''create optimizer'''
        '''create optimizer'''
        learning_rate = MyLRSchedule(initial_learning_rate=self.lr, drop=self.drop, epochs_drop=self.epochs_drop)
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)
        # optimizer = tf.keras.optimizers.Adam(learning_rate)

        '''start train:'''
        for epoch in range(LearningConfig.epochs):
            batch_index = 0
            for global_bunch, upper_bunch, middle_bunch, bottom_bunch, exp_batch in ds:
                exp_batch = exp_batch[:, -1]
                global_bunch = global_bunch[:, -1, :, :]
                upper_bunch = upper_bunch[:, -1, :, :]
                middle_bunch = middle_bunch[:, -1, :, :]
                bottom_bunch = bottom_bunch[:, -1, :, :]

                # self.test_print_batch(global_bunch, upper_bunch, middle_bunch, bottom_bunch, batch_index)

                '''train step'''
                step_gradients = self.train_step(epoch=epoch, step=batch_index, total_steps=step_per_epoch,
                                                 global_bunch=global_bunch,
                                                 upper_bunch=upper_bunch,
                                                 middle_bunch=middle_bunch,
                                                 bottom_bunch=bottom_bunch,
                                                 anno_exp=exp_batch,
                                                 model=model, optimizer=optimizer, c_loss=c_loss,
                                                 summary_writer=summary_writer)
                batch_index += 1

                # '''apply gradients'''
                # if batch_index > 0 and batch_index % virtual_step_per_epoch == 0:
                #     '''apply gradient'''
                #     print("===============apply gradient================= ")
                #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                #     gradients = None
                # else:
                #     '''accumulate gradient'''
                #     if gradients is None:
                #         gradients = [self._flat_gradients(g) / LearningConfig.virtual_batch_size for g in
                #                      step_gradients]
                #     else:
                #         for i, g in enumerate(step_gradients):
                #             gradients[i] += self._flat_gradients(g) / LearningConfig.virtual_batch_size

            '''evaluating part'''
            global_accuracy, avg_accuracy, acc_per_label, conf_mat = self._eval_model(model=model)
            '''save weights'''
            model.save(save_path + '_' + str(epoch) + '_' + self.dataset_name +
                       '_ACglob-' + str(global_accuracy) +
                       '_ACavg-' + str(avg_accuracy) +
                       '.h5')

    def calc_learning_rate(self, iterations, step_size, base_lr, max_lr, gamma=0.99994):
        """"""
        '''reducing triangle'''
        cycle = np.floor(1 + iterations / (2 * step_size))
        x = np.abs(iterations / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))

        '''exp'''
        # cycle = np.floor(1 + iterations / (2 * step_size))
        # x = np.abs(iterations / step_size - 2 * cycle + 1)
        # lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma ** iterations

        print('LR is: ' + str(lr))
        return lr

    def train_step(self, epoch, step, total_steps, model,
                   global_bunch, upper_bunch, middle_bunch, bottom_bunch,
                   anno_exp, optimizer, summary_writer, c_loss):
        with tf.GradientTape() as tape:
            '''create annotation_predicted'''
            # annotation_predicted = model(images, training=True)
            # val_pr, exp_pr = model(images, training=True)
            exp_pr, emb_face, emb_eyes, emb_nose, emb_mouth = model([global_bunch, upper_bunch,
                                                                     middle_bunch, bottom_bunch],
                                                                    training=True)  # todo

            '''calculate loss'''
            loss_exp, accuracy = c_loss.cross_entropy_loss(y_pr=exp_pr, y_gt=anno_exp,
                                                           num_classes=self.num_of_classes,
                                                           ds_name=self.dataset_name)

            loss_face = c_loss.triplet_loss(y_pr=emb_face, y_gt=anno_exp)
            loss_eyes = c_loss.triplet_loss(y_pr=emb_eyes, y_gt=anno_exp)
            loss_nose = c_loss.triplet_loss(y_pr=emb_nose, y_gt=anno_exp)
            loss_mouth = c_loss.triplet_loss(y_pr=emb_mouth, y_gt=anno_exp)

            loss_total = loss_exp + loss_face + loss_eyes + loss_nose + loss_mouth
        '''calculate gradient'''
        gradients_of_model = tape.gradient(loss_total, model.trainable_variables)
        # '''apply Gradients:'''
        optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps),
                 ' -> : loss_total: ', loss_total,
                 ' -> : accuracy: ', accuracy,
                 ' -> : loss_exp: ', loss_exp,
                 ' -> : loss_face: ', loss_face,
                 ' -> : loss_eyes: ', loss_eyes,
                 ' -> : loss_nose: ', loss_nose,
                 ' -> : loss_mouth: ', loss_mouth)
        with summary_writer.as_default():
            tf.summary.scalar('loss_total', loss_total, step=epoch)
            tf.summary.scalar('loss_exp', loss_exp, step=epoch)
            tf.summary.scalar('loss_face', loss_face, step=epoch)
            tf.summary.scalar('loss_eyes', loss_eyes, step=epoch)
            tf.summary.scalar('loss_nose', loss_nose, step=epoch)
            tf.summary.scalar('loss_mouth', loss_mouth, step=epoch)
        return gradients_of_model

    def _eval_model(self, model):
        """"""
        '''first we need to create the 4 bunch here: '''

        '''for Affectnet, we need to calculate accuracy of each label and then total avg accuracy:'''
        global_accuracy = 0
        conf_mat = []
        if self.dataset_name == DatasetName.affectnet:
            if self.ds_type == DatasetType.train:
                affn = AffectNet(ds_type=DatasetType.eval)
            else:
                affn = AffectNet(ds_type=DatasetType.eval_7)
            global_accuracy, conf_mat = affn.test_accuracy_dynamic(model=model)
        if self.dataset_name == DatasetName.rafdb:
            rafdb = RafDB(ds_type=DatasetType.test)
            global_accuracy, conf_mat = rafdb.test_accuracy_dynamic(model=model)

        # else:
        #     predictions = model(img_batch_eval)
        #     scores = np.array([tf.nn.softmax(predictions[i]) for i in range(len(pn_batch_eval))])
        #     predicted_lbls = np.array([np.argmax(scores[i]) for i in range(len(pn_batch_eval))])
        #
        #     acc = accuracy_score(pn_batch_eval, predicted_lbls)
        print("================== Total Accuracy =====================")
        print(global_accuracy)
        print("================== Confusion Matrix =====================")
        print(conf_mat)
        return global_accuracy, conf_mat

    def make_model(self, arch, w_path):
        cnn = CNNModel()
        model = cnn.get_model(arch=arch, num_of_classes=self.num_of_classes)
        if w_path is not None:
            model.load_weights(w_path)
        return model

    def _get_optimizer(self, lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-6):
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
        # return tf.keras.optimizers.SGD(lr=lr)

    def _flat_gradients(self, grads_or_idx_slices):
        if type(grads_or_idx_slices) == tf.IndexedSlices:
            return tf.scatter_nd(
                tf.expand_dims(grads_or_idx_slices.indices, 1),
                grads_or_idx_slices.values,
                grads_or_idx_slices.dense_shape
            )
        return grads_or_idx_slices

    def test_print_batch(self, global_bunch, upper_bunch, middle_bunch, bottom_bunch, _index):
        dhl = DataHelper()
        global_bunch = np.array(global_bunch)
        upper_bunch = np.array(upper_bunch)
        middle_bunch = np.array(middle_bunch)
        bottom_bunch = np.array(bottom_bunch)

        bs = np.array(global_bunch).shape[0]
        for i in range(bs):
            dhl.test_image_print(img_name=str((_index + 1) * (i + 1)) + '_glob_img', img=global_bunch[i, :, :, :],
                                 landmarks=[])
            dhl.test_image_print(img_name=str((_index + 1) * (i + 1)) + '_up_img', img=upper_bunch[i, :, :, :],
                                 landmarks=[])
            dhl.test_image_print(img_name=str((_index + 1) * (i + 1)) + '_mid_dr', img=middle_bunch[i, :, :, :],
                                 landmarks=[])
            dhl.test_image_print(img_name=str((_index + 1) * (i + 1)) + '_bot_dr', img=bottom_bunch[i, :, :, :],
                                 landmarks=[])
        pass


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, drop, epochs_drop):
        self.initial_learning_rate = initial_learning_rate
        self.drop = drop
        self.epochs_drop = epochs_drop

    def __call__(self, step):
        return self.initial_learning_rate * math.pow(self.drop, math.floor((1 + step) / self.epochs_drop))
