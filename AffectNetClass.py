from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, ExpressionCodesAffectnet
from config import LearningConfig, InputDataSize, DatasetName, AffectnetConf, DatasetType

import numpy as np
import os
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray, savez_compressed, savez
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
from dataset_class import CustomDataset


class AffectNet:
    def __init__(self, ds_type):
        """we set the parameters needed during the whole class:
        """
        self.ds_type = ds_type
        if ds_type == DatasetType.train:
            self.img_path = AffectnetConf.no_aug_train_img_path
            self.anno_path = AffectnetConf.no_aug_train_annotation_path
            self.img_path_aug = AffectnetConf.aug_train_img_path
            self.masked_img_path = AffectnetConf.aug_train_masked_img_path
            self.anno_path_aug = AffectnetConf.aug_train_annotation_path

        elif ds_type == DatasetType.eval:
            self.img_path_aug = AffectnetConf.eval_img_path
            self.anno_path_aug = AffectnetConf.eval_annotation_path
            self.img_path = AffectnetConf.eval_img_path
            self.anno_path = AffectnetConf.eval_annotation_path
            self.masked_img_path = AffectnetConf.eval_masked_img_path

        elif ds_type == DatasetType.train_7:
            self.img_path = AffectnetConf.no_aug_train_img_path_7
            self.anno_path = AffectnetConf.no_aug_train_annotation_path_7
            self.img_path_aug = AffectnetConf.aug_train_img_path_7
            self.masked_img_path = AffectnetConf.aug_train_masked_img_path_7
            self.anno_path_aug = AffectnetConf.aug_train_annotation_path_7

        elif ds_type == DatasetType.eval_7:
            self.img_path_aug = AffectnetConf.eval_img_path_7
            self.anno_path_aug = AffectnetConf.eval_annotation_path_7
            self.img_path = AffectnetConf.eval_img_path_7
            self.anno_path = AffectnetConf.eval_annotation_path_7
            self.masked_img_path = AffectnetConf.eval_masked_img_path_7

        # elif ds_type == DatasetType.test:
        #     self.img_path = AffectnetConf.revised_test_img_path
        #     self.anno_path = AffectnetConf.revised_test_annotation_path

    def upsample_data_fix_rate(self):
        dhl = DataHelper()
        if self.ds_type == DatasetType.train:
            aug_factor_by_class = [0.5, 0.3, 1, 2, 4, 6, 1, 6]
            sample_count_by_class = np.zeros([8])
            img_addr_by_class = [[] for i in range(8)]
            anno_addr_by_class = [[] for i in range(8)]
            lnd_addr_by_class = [[] for i in range(8)]
        elif self.ds_type == DatasetType.train_7:
            aug_factor_by_class = [0.5, 0.3, 1, 2, 4, 6, 1]
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
        # [ 74874. 134415.  25459.  14090.   6378.   3803.  24882.]
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

    def upsample_data(self):
        """we generate some samples so that all classes will have equal number of training samples"""
        dhl = DataHelper()

        '''count samples & categorize their address based on their category'''
        if self.ds_type == DatasetType.train:
            sample_count_by_class = np.zeros([8])
            img_addr_by_class = [[] for i in range(8)]
            anno_addr_by_class = [[] for i in range(8)]
            lnd_addr_by_class = [[] for i in range(8)]
        else:
            sample_count_by_class = np.zeros([7])
            img_addr_by_class = [[] for i in range(7)]
            anno_addr_by_class = [[] for i in range(7)]
            lnd_addr_by_class = [[] for i in range(7)]

        """"""
        print("counting classes:")
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

        '''calculate augmentation factor for each class:'''
        aug_factor_by_class, aug_factor_by_class_freq = dhl.calculate_augmentation_rate(
            sample_count_by_class=sample_count_by_class,
            base_aug_factor=AffectnetConf.augmentation_factor)

        '''after we have calculated those two array, we will augment samples '''
        for i in range(len(anno_addr_by_class)):
            dhl.do_random_augment(img_addrs=img_addr_by_class[i],
                                  anno_addrs=anno_addr_by_class[i],
                                  lnd_addrs=lnd_addr_by_class[i],
                                  aug_factor=int(aug_factor_by_class[i]),
                                  aug_factor_freq=int(aug_factor_by_class_freq[i]),
                                  img_save_path=self.img_path_aug,
                                  anno_save_path=self.anno_path_aug,
                                  class_index=i
                                  )

    def create_synthesized_landmarks(self, model_file):
        dhl = DataHelper()
        model = tf.keras.models.load_model(model_file)
        for i, file in tqdm(enumerate(os.listdir(self.img_path))):
            if file.endswith(".jpg") or file.endswith(".png"):
                dhl.create_synthesized_landmarks_path(img_path=self.img_path,
                                                      anno_path=self.anno_path, file=file,
                                                      model=model,
                                                      test_print=False)

    def create_masked_image(self):
        dhl = DataHelper()
        for i, file in tqdm(enumerate(os.listdir(self.img_path_aug))):
            if file.endswith(".jpg") or file.endswith(".png"):
                if os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_exp.npy")) \
                        and os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_slnd.npy")):
                    '''load data'''
                    lnd = np.load(os.path.join(self.anno_path_aug, file[:-4] + "_slnd.npy"))
                    img_file_name = os.path.join(self.img_path_aug, file)
                    img = np.float32(Image.open(img_file_name)) / 255.0
                    '''create masks'''
                    dr_mask = np.expand_dims(dhl.create_derivative(img=img, lnd=lnd), axis=-1)
                    au_mask = np.expand_dims(dhl.create_AU_mask(img=img, lnd=lnd), axis=-1)
                    up_mask, mid_mask, bot_mask = dhl.create_spatial_mask(img=img, lnd=lnd)
                    up_mask = np.expand_dims(up_mask, axis=-1)
                    mid_mask = np.expand_dims(mid_mask, axis=-1)
                    bot_mask = np.expand_dims(bot_mask, axis=-1)
                    '''fuse images'''
                    face_fused = dhl.create_input_bunches(img_batch=img, dr_mask_batch=dr_mask, au_mask_batch=au_mask,
                                                          spatial_mask=None)
                    eyes_fused = dhl.create_input_bunches(img_batch=img, dr_mask_batch=dr_mask, au_mask_batch=au_mask,
                                                          spatial_mask=up_mask)
                    nose_fused = dhl.create_input_bunches(img_batch=img, dr_mask_batch=dr_mask, au_mask_batch=au_mask,
                                                          spatial_mask=mid_mask)
                    mouth_fused = dhl.create_input_bunches(img_batch=img, dr_mask_batch=dr_mask, au_mask_batch=au_mask,
                                                           spatial_mask=bot_mask)
                    '''save fused'''
                    savez_compressed(self.masked_img_path + file[:-4] + "_face", face_fused)
                    savez_compressed(self.masked_img_path + file[:-4] + "_eyes", eyes_fused)
                    savez_compressed(self.masked_img_path + file[:-4] + "_nose", nose_fused)
                    savez_compressed(self.masked_img_path + file[:-4] + "_mouth", mouth_fused)

                    # im_face.save(self.masked_img_path + file[:-4] + "_face.jpg")
                    # im_eyes.save(self.masked_img_path + file[:-4] + "_eyes.jpg")
                    # im_nose.save(self.masked_img_path + file[:-4] + "_nose.jpg")
                    # im_mouth.save(self.masked_img_path + file[:-4] + "_mouth.jpg")

    def create_derivative_mask(self):
        dhl = DataHelper()
        for i, file in tqdm(enumerate(os.listdir(self.img_path_aug))):
            if file.endswith(".jpg") or file.endswith(".png"):
                if os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_exp.npy")) \
                        and os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_slnd.npy")):
                    # check if we have already created it:
                    if os.path.exists(os.path.join(self.anno_path_aug + 'dmg/', file[:-4] + "_dmg.jpg")) or \
                            os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_dmg.jpg")): continue

                    dhl.create_derivative_path(img_path=self.img_path_aug,
                                               anno_path=self.anno_path_aug, file=file, test_print=False)

    def create_au_mask(self):
        dhl = DataHelper()
        for i, file in tqdm(enumerate(os.listdir(self.img_path_aug))):
            if file.endswith(".jpg") or file.endswith(".png"):
                if os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_exp.npy")) \
                        and os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_slnd.npy")):
                    if os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_im.jpg")) or \
                            os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_im.jpg")): continue
                    dhl.create_AU_mask_path(img_path=self.img_path_aug,
                                            anno_path=self.anno_path_aug, file=file, test_print=False)

    def create_spatial_masks(self):
        dhl = DataHelper()
        for i, file in tqdm(enumerate(os.listdir(self.img_path_aug))):
            if file.endswith(".jpg") or file.endswith(".png"):
                if os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_exp.npy")) \
                        and os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_slnd.npy")):
                    if os.path.exists(os.path.join(self.anno_path_aug + 'spm/', file[:-4] + "_spm_up.jpg")) and \
                            os.path.exists(os.path.join(self.anno_path_aug + 'spm/', file[:-4] + "_spm_md.jpg")) and \
                            os.path.exists(
                                os.path.join(self.anno_path_aug + 'spm/', file[:-4] + "_spm_bo.jpg")): continue
                    dhl.create_spatial_mask_path(img_path=self.img_path_aug,
                                                 anno_path=self.anno_path_aug, file=file, test_print=False)

    def read_csv(self, ds_name, ds_type, FLD_model_file_name, is_7=False):
        if ds_name == DatasetName.affectnet:
            if ds_type == DatasetType.train:
                csv_path = AffectnetConf.orig_csv_train_path
                load_img_path = AffectnetConf.orig_img_path_prefix
                save_img_path = AffectnetConf.aug_train_img_path
                save_anno_path = AffectnetConf.aug_train_annotation_path
                do_aug = True
            elif ds_type == DatasetType.eval:
                csv_path = AffectnetConf.orig_csv_evaluate_path
                load_img_path = AffectnetConf.orig_img_path_prefix
                save_img_path = AffectnetConf.eval_img_path
                save_anno_path = AffectnetConf.eval_annotation_path
                do_aug = False

            elif ds_type == DatasetType.train_7:
                csv_path = AffectnetConf.orig_csv_train_path
                load_img_path = AffectnetConf.orig_img_path_prefix
                save_img_path = AffectnetConf.aug_train_img_path_7
                save_anno_path = AffectnetConf.aug_train_annotation_path_7
                do_aug = True
            elif ds_type == DatasetType.eval_7:
                csv_path = AffectnetConf.orig_csv_evaluate_path
                load_img_path = AffectnetConf.orig_img_path_prefix
                save_img_path = AffectnetConf.eval_img_path_7
                save_anno_path = AffectnetConf.eval_annotation_path_7
                do_aug = False

            elif ds_type == DatasetType.test:
                csv_path = AffectnetConf.orig_csv_test_path
                load_img_path = AffectnetConf.orig_test_path_prefix
                save_img_path = AffectnetConf.revised_test_img_path
                save_anno_path = AffectnetConf.revised_test_annotation_path
                do_aug = False

        csv_data = None
        with open(csv_path, 'r') as f:
            csv_data = list(csv.reader(f, delimiter="\n"))
        if csv_data is not None:
            img_path_arr, bbox_arr, landmarks_arr, expression_lbl_arr, valence_arr, arousal_arr = \
                self.decode_affectnet(csv_data[1:])
            self.create_affectnet(load_img_path=load_img_path, save_img_path=save_img_path,
                                  save_anno_path=save_anno_path,
                                  img_path_arr=img_path_arr, bbox_arr=bbox_arr, landmarks_arr=landmarks_arr,
                                  expression_lbl_arr=expression_lbl_arr, valence_arr=valence_arr,
                                  arousal_arr=arousal_arr, FLD_model_file_name=FLD_model_file_name, do_aug=do_aug,
                                  is_7=is_7)
        return 0

    def decode_affectnet(self, csv_data):
        img_path_arr = []
        bbox_arr = []
        landmarks_arr = []
        expression_lbl_arr = []
        valence_arr = []
        arousal_arr = []
        index = 0
        for line in tqdm(csv_data):
            img_path, x, y, width, height, lnds, expression_lbl, valence, arousal = line[0].split(',')
            landmarks = lnds.split(';')
            img_path_arr.append(img_path)
            bbox_arr.append([x, y, width, height])
            landmarks_arr.append(landmarks)
            expression_lbl_arr.append(expression_lbl)
            valence_arr.append(valence)
            arousal_arr.append(arousal)
            index += 1
            # if index >10:
            #     break
        return img_path_arr, bbox_arr, landmarks_arr, expression_lbl_arr, valence_arr, arousal_arr

    def create_affectnet(self, load_img_path, save_img_path, save_anno_path, img_path_arr, bbox_arr, landmarks_arr,
                         expression_lbl_arr, valence_arr, arousal_arr, FLD_model_file_name, do_aug, is_7):

        # model = tf.keras.models.load_model(FLD_model_file_name)
        dhl = DataHelper()
        model = None
        if is_7:
            print('777777777777777777777777777777777777')
            print('++++++++++++| 7 labels |++++++++++++')
            print('777777777777777777777777777777777777')
        else:
            print('888888888888888888888888888888888888')
            print('++++++++++++| 8 labels |++++++++++++')
            print('888888888888888888888888888888888888')

        print('len(img_path_arr)')
        print(len(img_path_arr))

        for i in tqdm(range(len(img_path_arr))):
            if is_7:
                if int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.none or \
                        int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.uncertain or \
                        int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.contempt or \
                        int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.noface:
                    continue
            else:
                if int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.none or \
                        int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.uncertain or \
                        int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.noface:
                    continue
            '''crop, resize, augment image'''
            dhl.crop_resize_aug_img(load_img_name=load_img_path + img_path_arr[i],
                                    save_img_name=save_img_path + str(i) + '.jpg',
                                    bbox=bbox_arr[i], landmark=landmarks_arr[i],
                                    save_anno_name=save_anno_path + str(i) + '_lnd',
                                    synth_save_anno_name=save_anno_path + str(i) + '_slnd',
                                    model=model, do_aug=do_aug)
            '''save annotation: exp_lbl, valence, arousal, landmark '''
            # print(str(int(expression_lbl_arr[i])))
            # save(save_anno_path + str(i) + '_exp', str(int(expression_lbl_arr[i])-1))
            save(save_anno_path + str(i) + '_exp', str(int(expression_lbl_arr[i])))
            save(save_anno_path + str(i) + '_val', valence_arr[i])
            save(save_anno_path + str(i) + '_aro', arousal_arr[i])

    def test_accuracy(self, model):
        dhp = DataHelper()
        if self.ds_type == DatasetType.eval:
            num_lbls = 8
        else:
            num_lbls = 7

        batch_size = LearningConfig.batch_size
        exp_pr_glob = []
        exp_gt_glob = []
        acc_per_label = []
        '''create batches'''
        face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames = \
            dhp.create_masked_generator_full_path(
                img_path=self.masked_img_path,
                annotation_path=self.anno_path, label=None, num_of_samples=None)
        print(len(face_img_filenames))
        step_per_epoch = int(len(face_img_filenames) // batch_size)
        exp_pr_lbl = []
        exp_gt_lbl = []

        cds = CustomDataset()
        ds = cds.create_dataset(file_names_face=face_img_filenames,
                                file_names_eyes=eyes_img_filenames,
                                file_names_nose=nose_img_filenames,
                                file_names_mouth=mouth_img_filenames,
                                anno_names=exp_filenames,
                                is_validation=True)

        batch_index = 0
        for global_bunch, upper_bunch, middle_bunch, bottom_bunch, exp_gt_b in ds:
            '''predict on batch'''
            exp_gt_b = exp_gt_b[:, -1]
            global_bunch = global_bunch[:, -1, :, :]
            upper_bunch = upper_bunch[:, -1, :, :]
            middle_bunch = middle_bunch[:, -1, :, :]
            bottom_bunch = bottom_bunch[:, -1, :, :]

            probab_exp_pr_b, _, _, _, _ = model.predict_on_batch([global_bunch, upper_bunch,
                                                                  middle_bunch, bottom_bunch])
            scores_b = np.array([tf.nn.softmax(probab_exp_pr_b[i]) for i in range(len(probab_exp_pr_b))])
            exp_pr_b = np.array([np.argmax(scores_b[i]) for i in range(len(probab_exp_pr_b))])

            # print(exp_pr_b)
            # print(exp_gt_b)
            # print('================')

            exp_pr_lbl += np.array(exp_pr_b).tolist()
            exp_gt_lbl += np.array(exp_gt_b).tolist()
            batch_index += 1
        exp_pr_lbl = np.float64(np.array(exp_pr_lbl))
        exp_gt_lbl = np.float64(np.array(exp_gt_lbl))

        global_accuracy = accuracy_score(exp_gt_lbl, exp_pr_lbl)
        conf_mat = confusion_matrix(exp_gt_lbl, exp_pr_lbl)
        conf_mat = tf.math.confusion_matrix(exp_gt_lbl, exp_pr_lbl, num_classes=7)

        ds = None
        face_img_filenames = None
        eyes_img_filenames = None
        nose_img_filenames = None
        mouth_img_filenames = None
        exp_filenames = None

        return global_accuracy, conf_mat

    def _test_accuracy(self, model):
        """"""
        dhp = DataHelper()
        ''' we need to get samples by category first. Then, predict accuracy on batch for each class and
                then calculate the average '''
        if self.ds_type == DatasetType.eval:
            num_lbls = 8
        else:
            num_lbls = 7

        batch_size = LearningConfig.batch_size
        exp_pr_glob = []
        exp_gt_glob = []
        acc_per_label = []
        for lbl in range(num_lbls):
            val_img_filenames, val_exp_filenames, val_lnd_filenames = dhp.create_generators_with_mask_online(
                img_path=self.img_path,
                annotation_path=self.anno_path, label=lbl, num_of_samples=None)

            '''create batches'''
            step_per_epoch = int(len(val_img_filenames) // batch_size)
            exp_pr_lbl = []
            exp_gt_lbl = []
            for batch_index in tqdm(range(step_per_epoch)):
                global_bunch, upper_bunch, middle_bunch, bottom_bunch, exp_gt_b = dhp.get_batch_sample_online(
                    batch_index=batch_index, img_path=self.img_path,
                    annotation_path=self.anno_path,
                    img_filenames=val_img_filenames,
                    exp_filenames=val_exp_filenames,
                    lnd_filenames=val_lnd_filenames,
                    batch_size=batch_size)
                '''predict on batch'''
                probab_exp_pr_b, _, _, _, _ = model.predict_on_batch([global_bunch, upper_bunch,
                                                                      middle_bunch, bottom_bunch])

                # scores_b = np.array([tf.nn.softmax(probab_exp_pr_b[i]) for i in range(len(probab_exp_pr_b))])
                # exp_pr_b = np.array([np.argmax(scores_b[i]) for i in range(len(probab_exp_pr_b))])
                exp_pr_b = np.array([np.argmax(probab_exp_pr_b[i]) for i in range(len(probab_exp_pr_b))])
                '''append to label-global'''
                exp_gt_lbl += exp_gt_b.tolist()
                exp_pr_lbl += exp_pr_b.tolist()
                '''append to label-global'''
                exp_gt_glob += exp_gt_b.tolist()
                exp_pr_glob += exp_pr_b.tolist()
                ''''''

            '''calculate per label acuracy'''
            acc_per_label.append(accuracy_score(exp_gt_lbl, exp_pr_lbl))
        '''print per=label accuracy and calculate the average'''
        avg_accuracy = np.mean(np.array(acc_per_label))
        global_accuracy = accuracy_score(exp_gt_glob, exp_pr_glob)
        '''calculate confusion matrix'''
        conf_mat = confusion_matrix(exp_gt_glob, exp_pr_glob)
        '''clean memory '''
        exp_pr_glob = None
        exp_gt_glob = None
        exp_pr_lbl = None
        exp_gt_lbl = None
        global_bunch = None
        upper_bunch = None
        middle_bunch = None
        bottom_bunch = None
        exp_gt_b = None
        val_img_filenames = None
        val_exp_filenames = None
        val_lnd_filenames = None
        val_dr_mask_filenames = None
        val_au_mask_filenames = None
        val_up_mask_filenames = None
        val_md_mask_filenames = None
        val_bo_mask_filenames = None
        scores_b = None
        '''return'''
        return global_accuracy, avg_accuracy, acc_per_label, conf_mat
