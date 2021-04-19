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
import os


class Train:
    def __init__(self, dataset_name, ds_type):
        self.dataset_name = dataset_name
        if dataset_name == DatasetName.affectnet:
            if ds_type == DatasetType.train:
                self.img_path = AffectnetConf.aug_train_img_path
                self.annotation_path = AffectnetConf.aug_train_annotation_path
                self.val_img_path = AffectnetConf.eval_img_path
                self.val_annotation_path = AffectnetConf.eval_annotation_path
                self.num_of_classes = 8
            elif ds_type == DatasetType.train_7:
                self.img_path = AffectnetConf.aug_train_img_path_7
                self.annotation_path = AffectnetConf.aug_train_annotation_path_7
                self.val_img_path = AffectnetConf.eval_img_path_7
                self.val_annotation_path = AffectnetConf.eval_annotation_path_7
                self.num_of_classes = 7

    def train(self, arch, weight_path):
        """"""

        '''create loss'''
        c_loss = CustomLosses()

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        start_train_date = datetime.now().strftime("%Y%m%d-%H%M%S")

        '''making models'''
        _lr = 5e-3
        model = self.make_model(arch=arch, w_path=weight_path)
        '''create optimizer'''
        optimizer = self._get_optimizer(lr=_lr)

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
        img_filenames, exp_filenames, lnd_filenames, dr_mask_filenames, au_mask_filenames, \
        up_mask_filenames, md_mask_filenames, bo_mask_filenames = dhp.create_generators_with_mask(
            img_path=self.img_path, annotation_path=self.annotation_path)

        # '''     Validation   Generator'''
        # val_img_filenames, val_exp_filenames, val_lnd_filenames, val_dr_mask_filenames, \
        # val_au_mask_filenames, val_up_mask_filenames, val_md_mask_filenames, \
        # val_bo_mask_filenames = dhp.create_generators_with_mask(
        #     img_path=self.val_img_path,
        #     annotation_path=self.val_annotation_path)

        # accuracy, confusion_matrix = self._eval_model(model, val_img_filenames, val_exp_filenames,
        #                                               val_lnd_filenames, val_dr_mask_filenames,
        #                                               val_au_mask_filenames, val_up_mask_filenames,
        #                                               val_md_mask_filenames, val_bo_mask_filenames)

        '''create train configuration'''
        step_per_epoch = len(img_filenames) // LearningConfig.batch_size
        '''start train:'''
        for epoch in range(LearningConfig.epochs):
            img_filenames, exp_filenames, lnd_filenames, dr_mask_filenames, au_mask_filenames, \
            up_mask_filenames, md_mask_filenames, bo_mask_filenames = dhp.shuffle_data(img_filenames,
                                                                                       exp_filenames, lnd_filenames,
                                                                                       dr_mask_filenames,
                                                                                       au_mask_filenames,
                                                                                       up_mask_filenames,
                                                                                       md_mask_filenames,
                                                                                       bo_mask_filenames)
            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                images, anno_exp = dhp.get_batch_sample(batch_index=batch_index, img_path=self.img_path,
                                                        annotation_path=self.annotation_path,
                                                        img_filenames=img_filenames,
                                                        exp_filenames=exp_filenames,
                                                        lnd_filenames=lnd_filenames,
                                                        dr_mask_filenames=dr_mask_filenames,
                                                        au_mask_filenames=au_mask_filenames,
                                                        up_mask_filenames=up_mask_filenames,
                                                        md_mask_filenames=md_mask_filenames,
                                                        bo_mask_filenames=bo_mask_filenames)
                '''convert to tensor'''
                images = tf.cast(images, tf.float32)
                # anno_val = tf.cast(anno_val, tf.int8)
                anno_exp = tf.cast(anno_exp, tf.uint8)
                '''train step'''
                self.train_step(epoch=epoch, step=batch_index, total_steps=step_per_epoch, images=images,
                                model=model, anno_exp=anno_exp, optimizer=optimizer, c_loss=c_loss,
                                summary_writer=summary_writer)

            '''evaluating part'''
            accuracy, confusion_matrix = self._eval_model(model, val_img_filenames, val_exp_filenames,
                                                          val_lnd_filenames, val_dr_mask_filenames,
                                                          val_au_mask_filenames, val_up_mask_filenames,
                                                          val_md_mask_filenames, val_bo_mask_filenames)
            with summary_writer.as_default():
                tf.summary.scalar('Eval-LOSS', accuracy, step=epoch)

            '''save weights'''
            model.save(save_path + '_' + str(epoch) + '_' + self.dataset_name + '_' + str(accuracy) + '.h5')

            '''calculate Learning rate'''
            # _lr = self.calc_learning_rate(iterations=epoch, step_size=10, base_lr=1e-5, max_lr=1e-2)
            # optimizer = self._get_optimizer(lr=_lr)

    def calc_learning_rate(self, iterations, step_size, base_lr, max_lr, gamma=0.99994):
        '''reducing triangle'''
        # cycle = np.floor(1 + iterations / (2 * step_size))
        # x = np.abs(iterations / step_size - 2 * cycle + 1)
        # lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
        '''exp'''
        cycle = np.floor(1 + iterations / (2 * step_size))
        x = np.abs(iterations / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma ** iterations

        print('LR is: ' + str(lr))
        return lr

    def train_step(self, epoch, step, total_steps, images, model, anno_exp, optimizer, summary_writer, c_loss):
        with tf.GradientTape() as tape:
            '''create annotation_predicted'''
            # annotation_predicted = model(images, training=True)
            # val_pr, exp_pr = model(images, training=True)
            exp_pr = model([images, images, images, images], training=True)  # todo
            '''calculate loss'''
            # if mode == 0:
            # loss_val = c_loss.cross_entropy_loss(y_pr=val_pr, y_gt=anno_val)
            # print('======exp_pr=======')
            # tf.print(exp_pr)
            # print('======anno_exp=======')
            # tf.print(anno_exp)
            loss_exp = c_loss.cross_entropy_loss(y_pr=exp_pr, y_gt=anno_exp)
            # loss_total = loss_exp
            # loss_total = loss_val + loss_exp
            # else:
            #     loss_total = c_loss.regressor_loss(y_pr=annotation_predicted, y_gt=anno_val)
        '''calculate gradient'''
        gradients_of_model = tape.gradient(loss_exp, model.trainable_variables)
        '''apply Gradients:'''
        optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps), ' -> : LOSS: ', loss_exp)
        # ' -> : Loss-EXP: ', loss_exp)
        # ' -> : Loss-EXP: ', loss_exp, ' -> : Loss-Val: ', loss_val)
        with summary_writer.as_default():
            tf.summary.scalar('LOSS', loss_exp, step=epoch)
            # tf.summary.scalar('Loss-EXP', loss_exp, step=epoch)
            # tf.summary.scalar('Loss-Val', loss_val, step=epoch)

    def _eval_model(self, model, val_img_filenames, val_exp_filenames, val_lnd_filenames, val_dr_mask_filenames,
                    val_au_mask_filenames, val_up_mask_filenames, val_md_mask_filenames, val_bo_mask_filenames):
        """
        :param model:
        :param val_img_filenames:
        :param val_exp_filenames:
        :param val_lnd_filenames:
        :param val_dr_mask_filenames:
        :param val_au_mask_filenames:
        :param val_up_mask_filenames:
        :param val_md_mask_filenames:
        :param val_bo_mask_filenames:
        :return: accuracy,
        :return: confusion matrix #todo
        """
        '''first we need to create the 4 bunch here: '''

        '''for Affectnet, we need to calculate accuracy of each label and then total avg accuracy:'''
        if self.dataset_name == DatasetName.affectnet:
            pass  # todo
        else:
            predictions = model(img_batch_eval)
            scores = np.array([tf.nn.softmax(predictions[i]) for i in range(len(pn_batch_eval))])
            predicted_lbls = np.array([np.argmax(scores[i]) for i in range(len(pn_batch_eval))])

            acc = accuracy_score(pn_batch_eval, predicted_lbls)
        return acc, 0

    def make_model(self, arch, w_path):
        cnn = CNNModel()
        model = cnn.get_model(arch=arch, num_of_classes=self.num_of_classes)
        if w_path is not None:
            model.load_weights(w_path)
        return model

    def _get_optimizer(self, lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-4):
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
