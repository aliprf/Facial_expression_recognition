from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, ExpressionCodesAffectnet
from config import LearningConfig, InputDataSize, DatasetName, RafDBConf, DatasetType

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
            self.masked_img_path = RafDBConf.aug_train_masked_img_path

        elif ds_type == DatasetType.test:
            self.img_path = RafDBConf.test_img_path
            self.anno_path = RafDBConf.test_annotation_path
            self.img_path_aug = RafDBConf.test_img_path
            self.anno_path_aug = RafDBConf.test_annotation_path
            self.masked_img_path = RafDBConf.test_masked_img_path

    def create_from_orig(self, ds_type):
        print('create_from_orig & relabel to affectNetLike--->')
        """
        labels are from 1-7, but we save them from 0 to 6

        :param ds_type:
        :return:
        """
        if ds_type == DatasetType.train:
            txt_path = RafDBConf.orig_annotation_txt_path
            load_img_path = RafDBConf.orig_image_path
            load_bbox_path = RafDBConf.orig_bounding_box
            save_img_path = RafDBConf.no_aug_train_img_path
            save_anno_path = RafDBConf.no_aug_train_annotation_path
            prefix = 'train'
        elif ds_type == DatasetType.test:
            txt_path = RafDBConf.orig_annotation_txt_path
            load_img_path = RafDBConf.orig_image_path
            load_bbox_path = RafDBConf.orig_bounding_box
            save_img_path = RafDBConf.test_img_path
            save_anno_path = RafDBConf.test_annotation_path
            prefix = 'test'

        '''read the text file, and save exp, and image'''
        file1 = open(txt_path, 'r')
        dhl = DataHelper()
        affectnet_like_lbls = [3, 4, 5, 1, 2, 6, 0]
        while True:
            line = file1.readline()
            if not line:
                break
            f_name = line.split(' ')[0]
            if prefix not in f_name: continue

            img_source_address = load_img_path + f_name[:-4] + '.jpg'
            img_dest_address = save_img_path + f_name

            exp = int(line.split(' ')[1]) - 1
            # '''relabel to affectNet'''
            exp = affectnet_like_lbls[exp]

            img = np.array(Image.open(img_source_address))

            '''padd, resize image and save'''
            x_min, y_min, x_max, y_max = self.get_bounding_box(load_bbox_path + f_name[:-4] + '_boundingbox.txt')
            img = dhl.crop_image_bbox(img, x_min, y_min, x_max, y_max)
            '''resize'''
            res_img = resize(img, (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                             anti_aliasing=True)
            im = Image.fromarray(np.round(res_img * 255.0).astype(np.uint8))
            im.save(img_dest_address)
            '''save annotation'''
            np.save(save_anno_path + f_name[:-4] + '_exp', exp)
        file1.close()

    def create_synthesized_landmarks(self, model_file, test_print=False):
        dhl = DataHelper()
        model = tf.keras.models.load_model(model_file)
        for i, file in tqdm(enumerate(os.listdir(self.img_path))):
            if file.endswith(".jpg") or file.endswith(".png"):
                dhl.create_synthesized_landmarks_path(img_path=self.img_path,
                                                      anno_path=self.anno_path, file=file,
                                                      model=model,
                                                      test_print=test_print)
    def get_bounding_box(self, file_name):
        file1 = open(file_name, 'r')
        line = file1.readline()
        x_min, y_min, x_max, y_max = line.split(' ')[0:4]
        file1.close()
        return int(float(x_min)), int(float(y_min)), int(float(x_max)), int(float(y_max))

    def report(self, aug=True):
        dhl = DataHelper()
        if aug:
            anno_path = self.anno_path_aug
        else:
            anno_path = self.anno_path

        sample_count_by_class = np.zeros([7])
        for i, file in tqdm(enumerate(os.listdir(anno_path))):
            if file.endswith("_exp.npy"):
                exp = np.load(os.path.join(anno_path, file))
                sample_count_by_class[exp] += 1
                # print(exp, end="-")
        print("sample_count_by_category: ====>>")
        ''' testset : [ 680. 1185.  478.  329.   74.  160.  162.]
            train :   [7,572. 4,772. 7,928. 6,450. 5,058. 5,736. 6,345.]
                      [7572. 4772. 7928. 6450. 5058. 5736. 6345.]
             '''

        print(sample_count_by_class)


    def upsample_data_fix_rate(self):
        """  [2524. 4772.   1982.  1290.   281.   717.  705.]
            Augmentation: sample_count_by_category:      ====>>
            [2,      1,     3,    4,       17,      7,     8]
            [5048,  4772,   5946,  5160,  4777, 5019,  5640]
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

    def upsample_data(self):
        """we generate some samples so that all classes will have equal number of training samples"""
        dhl = DataHelper()

        '''count samples & categorize their address based on their category'''
        sample_count_by_class = np.zeros([7])
        img_addr_by_class = [[] for i in range(7)]
        anno_addr_by_class = [[] for i in range(7)]
        lnd_addr_by_class = [[] for i in range(7)]

        """"""
        print("counting classes:")
        count = 0
        for i, file in tqdm(enumerate(os.listdir(self.anno_path))):
            if file.endswith("_exp.npy"):
                exp = int(np.load(os.path.join(self.anno_path, file)))
                sample_count_by_class[exp] += 1
                '''adding ex'''
                anno_addr_by_class[exp].append(os.path.join(self.anno_path, file))
                img_addr_by_class[exp].append(os.path.join(self.img_path, file[:-8] + '.jpg'))
                lnd_addr_by_class[exp].append(os.path.join(self.anno_path, file[:-8] + '_slnd.npy'))
                count += 1

        print("sample_count_by_category: ====>>")
        print(sample_count_by_class)
        # {Surprise 1290}===={ Fear 281.}===[Disgust 717}===[Happiness 4772]
        # ==={ Sadness 1982}=={Anger 705}===.[ Neutral 2524}

        '''calculate augmentation factor for each class:'''
        aug_factor_by_class, aug_factor_by_class_freq = dhl.calculate_augmentation_rate(
            sample_count_by_class=sample_count_by_class,
            base_aug_factor=RafDBConf.augmentation_factor)

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

    def relabel(self):
        # affectnet_like_lbls = [3, 4, 5, 1, 2, 6, 0]
        raf_lbls = [6, 3, 4, 0, 1, 2, 5]
        for i, file in tqdm(enumerate(os.listdir(self.img_path_aug))):
            if file.endswith(".jpg") or file.endswith(".png"):
                if os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_exp.npy")) \
                        and os.path.exists(os.path.join(self.anno_path_aug, file[:-4] + "_slnd.npy")):
                    lbl = np.int64(np.load(os.path.join(self.anno_path_aug, file[:-4] + "_exp.npy")))
                    # lbl = affectnet_like_lbls[lbl]
                    lbl = raf_lbls[lbl]
                    save(os.path.join(self.anno_path_aug, file[:-4] + "_exp.npy"), lbl)


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

    def test_accuracy_dynamic(self, model):
        dhp = DataHelper()
        '''create batches'''
        img_filenames, exp_filenames, spm_up_filenames, spm_md_filenames, spm_bo_filenames = \
            dhp.create_generator_full_path_with_spm(img_path=self.img_path,
                                                    annotation_path=self.anno_path)
        print(len(img_filenames))
        exp_pr_lbl = []
        exp_gt_lbl = []

        dds = DynamicDataset()
        ds = dds.create_dataset(img_filenames=img_filenames,
                                spm_up_filenames=spm_up_filenames,
                                spm_md_filenames=spm_md_filenames,
                                spm_bo_filenames=spm_bo_filenames,
                                anno_names=exp_filenames,
                                is_validation=True,
                                ds=DatasetName.rafdb)
        batch_index = 0
        for global_bunch, upper_bunch, middle_bunch, bottom_bunch, exp_gt_b in ds:
            '''predict on batch'''
            global_bunch = global_bunch[:, -1, :, :]
            upper_bunch = upper_bunch[:, -1, :, :]
            middle_bunch = middle_bunch[:, -1, :, :]
            bottom_bunch = bottom_bunch[:, -1, :, :]

            probab_exp_pr_b, _, _, _, _ = model.predict_on_batch([global_bunch, upper_bunch,
                                                                  middle_bunch, bottom_bunch])
            exp_pr_b = np.array([np.argmax(probab_exp_pr_b[i]) for i in range(len(probab_exp_pr_b))])

            exp_pr_lbl += np.array(exp_pr_b).tolist()
            exp_gt_lbl += np.array(exp_gt_b).tolist()
            batch_index += 1
        exp_pr_lbl = np.int64(np.array(exp_pr_lbl))
        exp_gt_lbl = np.int64(np.array(exp_gt_lbl))

        global_accuracy = accuracy_score(exp_gt_lbl, exp_pr_lbl)
        conf_mat = confusion_matrix(exp_gt_lbl, exp_pr_lbl)
        # conf_mat = tf.math.confusion_matrix(exp_gt_lbl, exp_pr_lbl, num_classes=7)/500.0

        ds = None
        face_img_filenames = None
        eyes_img_filenames = None
        nose_img_filenames = None
        mouth_img_filenames = None
        exp_filenames = None
        global_bunch = None
        upper_bunch = None
        middle_bunch = None
        bottom_bunch = None

        return global_accuracy, conf_mat

    def test_accuracy(self, model):
        dhp = DataHelper()
        batch_size = LearningConfig.batch_size
        # exp_pr_glob = []
        # exp_gt_glob = []
        # acc_per_label = []
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
                                is_validation=True,
                                ds=DatasetName.rafdb)

        batch_index = 0
        for global_bunch, upper_bunch, middle_bunch, bottom_bunch, exp_gt_b in tqdm(ds):
            '''predict on batch'''
            exp_gt_b = exp_gt_b[:, -1]
            global_bunch = global_bunch[:, -1, :, :]
            upper_bunch = upper_bunch[:, -1, :, :]
            middle_bunch = middle_bunch[:, -1, :, :]
            bottom_bunch = bottom_bunch[:, -1, :, :]

            probab_exp_pr_b, _, _, _, _ = model.predict_on_batch([global_bunch, upper_bunch,
                                                                  middle_bunch, bottom_bunch])
            # scores_b = np.array([tf.nn.softmax(probab_exp_pr_b[i]) for i in range(len(probab_exp_pr_b))])
            # exp_pr_b = np.array([np.argmax(scores_b[i]) for i in range(len(probab_exp_pr_b))])
            exp_pr_b = np.array([np.argmax(probab_exp_pr_b[i]) for i in range(len(probab_exp_pr_b))])

            exp_pr_lbl += np.array(exp_pr_b).tolist()
            exp_gt_lbl += np.array(exp_gt_b).tolist()
            batch_index += 1
        exp_pr_lbl = np.float64(np.array(exp_pr_lbl))
        exp_gt_lbl = np.float64(np.array(exp_gt_lbl))

        global_accuracy = accuracy_score(exp_gt_lbl, exp_pr_lbl)
        conf_mat = confusion_matrix(exp_gt_lbl, exp_pr_lbl)
        # conf_mat = tf.math.confusion_matrix(exp_gt_lbl, exp_pr_lbl, num_classes=7)

        ds = None
        face_img_filenames = None
        eyes_img_filenames = None
        nose_img_filenames = None
        mouth_img_filenames = None
        exp_filenames = None

        return global_accuracy, conf_mat

