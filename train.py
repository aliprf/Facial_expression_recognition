from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, DatasetType
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


class Train:
    def __init__(self, dataset_name, ds_type):
        self.dataset_name = dataset_name
        self.ds_type = ds_type
        if dataset_name == DatasetName.affectnet:
            if ds_type == DatasetType.train:
                self.img_path = AffectnetConf.aug_train_img_path
                self.annotation_path = AffectnetConf.aug_train_annotation_path
                self.masked_img_path = AffectnetConf.aug_train_masked_img_path
                self.val_img_path = AffectnetConf.eval_img_path
                self.val_annotation_path = AffectnetConf.eval_annotation_path
                self.eval_masked_img_path = AffectnetConf.eval_masked_img_path
                self.num_of_classes = 8
                self.num_of_samples = AffectnetConf.num_of_samples_train
            elif ds_type == DatasetType.train_7:
                self.img_path = AffectnetConf.aug_train_img_path_7
                self.annotation_path = AffectnetConf.aug_train_annotation_path_7
                self.masked_img_path = AffectnetConf.aug_train_masked_img_path_7
                self.val_img_path = AffectnetConf.eval_img_path_7
                self.val_annotation_path = AffectnetConf.eval_annotation_path_7
                self.eval_masked_img_path = AffectnetConf.eval_masked_img_path_7
                self.num_of_classes = 7
                self.num_of_samples = AffectnetConf.num_of_samples_train_7

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
        # elif self.dataset_name == DatasetName.:
        #     save_path = '/media/data2/alip/HM_WEIGHTs/wflw/efn_1d/11_april/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        '''create sample generator'''
        dhp = DataHelper()
        '''     Train   Generator'''
        face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames = \
            dhp.create_masked_generator(img_path=self.masked_img_path, annotation_path=self.annotation_path,
                                        num_of_samples=self.num_of_samples)

        global_accuracy, avg_accuracy, acc_per_label, conf_mat = self._eval_model(model=model)

        '''create train configuration'''
        step_per_epoch = len(face_img_filenames) // LearningConfig.batch_size
        gradients = None
        virtual_step_per_epoch = LearningConfig.virtual_batch_size // LearningConfig.batch_size

        '''create optimizer'''
        _lr = 1e-4
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            _lr,
            decay_steps=step_per_epoch * 10,  # will be 0.5 every 5 spoch
            decay_rate=1,
            staircase=False)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)

        '''start train:'''
        for epoch in range(LearningConfig.epochs):
            face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames = \
                shuffle(face_img_filenames, eyes_img_filenames,
                        nose_img_filenames, mouth_img_filenames, exp_filenames)

            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                # print('get_batch_sample ->')
                global_bunch, upper_bunch, middle_bunch, bottom_bunch, exp_batch = dhp.get_batch_sample_masked(
                    batch_index=batch_index, img_path=self.masked_img_path,
                    annotation_path=self.annotation_path,
                    exp_filenames=exp_filenames,
                    face_img_filenames=face_img_filenames,
                    eyes_img_filenames=eyes_img_filenames,
                    nose_img_filenames=nose_img_filenames,
                    mouth_img_filenames=mouth_img_filenames)

                '''convert to tensor'''
                # print('convert to tensor->')
                global_bunch = tf.cast(global_bunch, tf.float32)
                upper_bunch = tf.cast(upper_bunch, tf.float32)
                middle_bunch = tf.cast(middle_bunch, tf.float32)
                bottom_bunch = tf.cast(bottom_bunch, tf.float32)
                exp_batch = tf.cast(exp_batch, tf.uint8)

                '''train step'''
                step_gradients = self.train_step(epoch=epoch, step=batch_index, total_steps=step_per_epoch,
                                                 global_bunch=global_bunch,
                                                 upper_bunch=upper_bunch,
                                                 middle_bunch=middle_bunch,
                                                 bottom_bunch=bottom_bunch,
                                                 anno_exp=exp_batch,
                                                 model=model, optimizer=optimizer, c_loss=c_loss,
                                                 summary_writer=summary_writer)
                '''apply gradients'''
                if batch_index > 0 and batch_index % virtual_step_per_epoch == 0:
                    '''apply gradient'''
                    print("===============apply gradient================= ")
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    gradients = None
                else:
                    '''accumulate gradient'''
                    if gradients is None:
                        gradients = [self._flat_gradients(g) / LearningConfig.virtual_batch_size for g in
                                     step_gradients]
                    else:
                        for i, g in enumerate(step_gradients):
                            gradients[i] += self._flat_gradients(g) / LearningConfig.virtual_batch_size

            '''evaluating part'''
            global_accuracy, avg_accuracy, acc_per_label, conf_mat = self._eval_model(model=model)
            '''save weights'''
            model.save(save_path + '_' + str(epoch) + '_' + self.dataset_name +
                       '_ACglob-' + str(global_accuracy) +
                       '_ACavg-' + str(avg_accuracy) +
                       '.h5')

            '''calculate Learning rate'''
            # _lr = self.calc_learning_rate(iterations=epoch, step_size=10, base_lr=1e-5, max_lr=1e-2)
            # optimizer = self._get_optimizer(lr=_lr)

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
            loss_exp = 2 * c_loss.cross_entropy_loss(y_pr=exp_pr, y_gt=anno_exp, num_classes=self.num_of_classes)
            loss_face = 3 * c_loss.triplet_loss(y_pr=emb_face, y_gt=anno_exp)
            loss_eyes = c_loss.triplet_loss(y_pr=emb_eyes, y_gt=anno_exp)
            loss_nose = c_loss.triplet_loss(y_pr=emb_nose, y_gt=anno_exp)
            loss_mouth = c_loss.triplet_loss(y_pr=emb_mouth, y_gt=anno_exp)
            loss_total = loss_exp + loss_face + loss_eyes + loss_nose + loss_mouth
        '''calculate gradient'''
        gradients_of_model = tape.gradient(loss_total, model.trainable_variables)
        # '''apply Gradients:'''
        # optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps),
                 ' -> : loss_total: ', loss_total,
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
        if self.dataset_name == DatasetName.affectnet:
            if self.ds_type == DatasetType.train:
                affn = AffectNet(ds_type=DatasetType.eval)
            else:
                affn = AffectNet(ds_type=DatasetType.eval_7)
            global_accuracy, avg_accuracy, acc_per_label, conf_mat = affn.test_accuracy(model=model)

        # else:
        #     predictions = model(img_batch_eval)
        #     scores = np.array([tf.nn.softmax(predictions[i]) for i in range(len(pn_batch_eval))])
        #     predicted_lbls = np.array([np.argmax(scores[i]) for i in range(len(pn_batch_eval))])
        #
        #     acc = accuracy_score(pn_batch_eval, predicted_lbls)
        print("================== Total Accuracy =====================")
        print(global_accuracy)
        print("================== Average Accuracy =====================")
        print(avg_accuracy)
        print("==== per Label :")
        print(acc_per_label)
        print("================== Confusion Matrix =====================")
        print(conf_mat)
        return global_accuracy, avg_accuracy, acc_per_label, conf_mat

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
