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


class DataHelper:

    def do_random_augment(self, img_addrs, anno_addrs, lnd_addrs, aug_factor, aug_factor_freq,
                          img_save_path, anno_save_path, class_index):
        for i in tqdm(range(len(img_addrs))):
            if os.path.exists(anno_addrs[i]):

                '''load expression and image'''
                exp = int(np.load(anno_addrs[i]))
                landmark_orig = np.load(lnd_addrs[i])
                img_orig = np.array(Image.open(img_addrs[i]))

                '''set aug_factor'''
                if aug_factor_freq is not None and i % aug_factor_freq == 0 and aug_factor_freq != 1:
                    _aug_factor = aug_factor + 1
                else:
                    _aug_factor = aug_factor

                '''Down sampling if needed:'''
                if _aug_factor < 1:
                    k = int(1 / _aug_factor)
                    if i % k != 0:
                        continue

                '''save original images:'''
                im = Image.fromarray(np.round(img_orig).astype(np.uint8))
                im.save(img_save_path + str(class_index) + '_' + str(i) + '_0' + '.jpg')
                np.save(anno_save_path + str(class_index) + '_' + str(i) + '_0' + '_exp', exp)
                np.save(anno_save_path + str(class_index) + '_' + str(i) + '_0' + '_slnd', landmark_orig)

                if _aug_factor == 1: continue

                _aug_factor = int(_aug_factor) - 1  # we count original as well

                for aug_inx in range(_aug_factor):
                    '''rotate and rescale the image'''
                    img = np.copy(img_orig)
                    landmark = np.copy(landmark_orig)

                    '''flipping image'''
                    if aug_inx % 2 == 0:
                        img, landmark = self._flip_and_relabel(img, landmark)

                    img, landmark = self._rotate_image_and_lnds(img, landmark)
                    '''contras and color modification '''
                    img = self._adjust_gamma(img)
                    # img = self._noisy(img)
                    img = self._blur(img)
                    # img = self._add_occlusion(img)

                    '''save image and landmarks'''
                    im = Image.fromarray(np.round(img * 255.0).astype(np.uint8))
                    im.save(img_save_path + str(class_index) + '_' + str(i) + '_' + str(aug_inx + 1) + '.jpg')
                    np.save(anno_save_path + str(class_index) + '_' + str(i) + '_' + str(
                        aug_inx + 1) + '_exp', exp)
                    np.save(anno_save_path + str(class_index) + '_' + str(i) + '_' + str(
                        aug_inx + 1) + '_slnd',
                            landmark)

                    # self.test_image_print(img_name=str(exp) + '_' + str(i) + 'orig',
                    #                       img=img_orig, landmarks=landmark_orig)
                    # self.test_image_print(img_name=str(exp) + '_' + str(i) + 'flip',
                    #                       img=img_f, landmarks=landmark_f)
                    # self.test_image_print(img_name=str(exp) + '_' + str(i) + 'aug',
                    #                       img=img, landmarks=landmark)

    def create_mean_faces(self, img_path, anno_path):
        labels = [0, 1, 2, 3, 4, 5, 6, 7]
        landmarks = []
        for i, lbl in enumerate(labels):
            # landmarks[i] = []
            landmarks.append(i)
            landmarks[i] = []
        # landmarks = defaultdict(list)

        for i, file in tqdm(enumerate(os.listdir(img_path))):
            if file.endswith(".jpg") or file.endswith(".png"):
                exp = int(np.load(os.path.join(anno_path, file[:-4] + "_exp.npy")))
                if exp > 7: continue
                lnd = np.load(os.path.join(anno_path, file[:-4] + "_slnd.npy"))
                landmarks[exp].append(lnd[34:])
                # img_file_name = os.path.join(img_path, file)
                # self.test_image_print(img_name=str(i), img=Image.open(img_file_name), landmarks=lnd)

        for lbl in labels:
            lnd_array = landmarks[lbl]
            lbl_mean = lnd_array[0]  # np.mean(lnd_array, axis=0)
            self.test_image_print(img_name=str(lbl), img=np.zeros([224, 224, 3]), landmarks=lbl_mean)

    # def predict_fl_and_save(self, model_file_name, img_path, annotation_save_path):
    #     model = tf.keras.models.load_model(model_file_name)
    #     for i, file in tqdm(enumerate(os.listdir(img_path))):
    #         if file.endswith(".png") or file.endswith(".jpg"):
    #             img_file_name = os.path.join(img_path, str(file))
    #             anno_file_name = os.path.join(img_path, 'sfl_' + str(file)[:-3] + "npy")
    #
    #             img = np.expand_dims(np.array(Image.open(img_file_name)) / 255.0, axis=0)
    #             anno_Pre = self.de_normalized(annotation_norm=model.predict(img)[0])
    #             self.test_image_print(img_name='z_' + str(i) + '_pr' + str(i) + '__',
    #                                   img=np.array(Image.open(img_file_name)) / 255.0, landmarks=anno_Pre)

    def crop_resize_aug_img(self, load_img_name, save_img_name, save_anno_name, bbox, landmark, synth_save_anno_name,
                            model, do_aug):
        img = np.array(Image.open(load_img_name))

        landmark = list(map(float, landmark))
        # print('2222landmark')
        # print(landmark)

        annotation_x = []
        annotation_y = []
        for i in range(0, len(landmark), 2):
            annotation_x.append(float(landmark[i]))
            annotation_y.append(float(landmark[i + 1]))

        ''''''
        x, y, width, height = list(map(int, bbox))
        # x_1 = x  #int(x + 0.05 * width)
        # y_1 = y
        # x_2 = x + width # int(min(x + width + 0.15 * width, img.shape[0]))
        # y_2 = y + height

        x_1 = x
        y_1 = y
        x_2 = x + width
        y_2 = y + height

        fix_pad = 5
        _xmin = max(0, int(min(x_1, min(annotation_y) - fix_pad)))
        _ymin = max(0, int(min(y_1, min(annotation_x) - fix_pad)))
        _xmax = int(max(x_2, max(annotation_y) + fix_pad))
        _ymax = int(max(y_2, max(annotation_x) + fix_pad))
        ''''''

        croped_img = img[_xmin:_xmax, _ymin:_ymax]
        annotation_new = []
        for i in range(0, len(landmark), 2):
            annotation_new.append(landmark[i] - _xmin)
            annotation_new.append(landmark[i + 1] - _ymin)

        # print('annotation_new')
        # print(annotation_new)

        '''resize'''
        resized_img, annotation_resized = self.resize_image(img=croped_img, annotation=annotation_new)
        '''synthesize lnd'''
        # anno_Pre = 0  # self.de_normalized(annotation_norm=model.predict(np.expand_dims(resized_img, axis=0))[0])

        '''test print'''
        # self.test_image_print(img_name=save_img_name + '_synth', img=resized_img, landmarks=anno_Pre)
        # self.test_image_print(img_name=save_img_name + 'orig', img=resized_img, landmarks=annotation_resized)

        '''save'''
        im = Image.fromarray((resized_img * 255).astype(np.uint8))

        im.save(save_img_name)
        save(save_anno_name, annotation_resized)
        # save(synth_save_anno_name, anno_Pre)

        return 0

    def resize_image(self, img, annotation):
        if img.shape[0] == 0 or img.shape[1] == 0:
            print('resize_image  ERRORRR')

        resized_img = resize(img, (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                             anti_aliasing=True)
        dims = img.shape
        height = dims[0]
        width = dims[1]
        scale_factor_y = InputDataSize.image_input_size / height
        scale_factor_x = InputDataSize.image_input_size / width

        '''rescale and retrieve landmarks'''
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(landmarks=annotation,
                                                                                scale_factor_x=scale_factor_x,
                                                                                scale_factor_y=scale_factor_y)
        return resized_img, landmark_arr_xy

    def create_landmarks(self, landmarks, scale_factor_x, scale_factor_y):
        # landmarks_splited = _landmarks.split(';')
        landmark_arr_xy = []
        landmark_arr_x = []
        landmark_arr_y = []
        for j in range(0, len(landmarks), 2):
            x = float(landmarks[j]) * scale_factor_x
            y = float(landmarks[j + 1]) * scale_factor_y

            landmark_arr_xy.append(x)
            landmark_arr_xy.append(y)  # [ x1, y1, x2,y2 ]

            landmark_arr_x.append(x)  # [x1, x2]
            landmark_arr_y.append(y)  # [y1, y2]

        return landmark_arr_xy, landmark_arr_x, landmark_arr_y

    def shuffle_data(self, img_filenames, exp_filenames, lnd_filenames,
                     dr_mask_filenames, au_mask_filenames,
                     up_mask_filenames, md_mask_filenames,
                     bo_mask_filenames):
        img_filenames, exp_filenames, lnd_filenames, dr_mask_filenames, au_mask_filenames, \
        up_mask_filenames, md_mask_filenames, bo_mask_filenames = shuffle(img_filenames, exp_filenames, lnd_filenames,
                                                                          dr_mask_filenames, au_mask_filenames,
                                                                          up_mask_filenames, md_mask_filenames,
                                                                          bo_mask_filenames)
        return img_filenames, exp_filenames, lnd_filenames, dr_mask_filenames, au_mask_filenames, \
               up_mask_filenames, md_mask_filenames, bo_mask_filenames

    def create_generators_with_mask_online(self, img_path, annotation_path, num_of_samples, label=None):
        img_filenames, exp_filenames, lnd_filenames = self. \
            _create_image_and_labels_name_online(img_path=img_path,
                                                 annotation_path=annotation_path,
                                                 label=label,
                                                 num_of_samples=num_of_samples)
        # print('shuffle => ')
        '''shuffle'''
        img_filenames, exp_filenames, lnd_filenames = shuffle(img_filenames, exp_filenames, lnd_filenames)
        # print('<== shuffle ')
        return img_filenames, exp_filenames, lnd_filenames

    def create_masked_generator_full_path(self, img_path, annotation_path, num_of_samples, label=None):
        face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames = \
            self._create_masked_image_and_labels_name_full_path(img_path=img_path,
                                                         annotation_path=annotation_path,
                                                         label=label,
                                                         num_of_samples=num_of_samples)
        '''shuffle'''
        face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames = \
            shuffle(face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames)
        return face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames

    def create_masked_generator(self, img_path, annotation_path, num_of_samples, label=None):
        face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames = \
            self._create_image_and_labels_name(img_path=img_path,
                                               annotation_path=annotation_path,
                                               label=label,
                                               num_of_samples=num_of_samples)
        '''shuffle'''
        face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames = \
            shuffle(face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames)
        return face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames, exp_filenames

    def create_generators_with_mask(self, img_path, annotation_path, num_of_samples, label=None):
        # print('read file names =>')
        img_filenames, exp_filenames, lnd_filenames, dr_mask_filenames, au_mask_filenames, up_mask_filenames, \
        md_mask_filenames, bo_mask_filenames = self.__create_image_and_labels_name(img_path=img_path,
                                                                                   annotation_path=annotation_path,
                                                                                   label=label,
                                                                                   num_of_samples=num_of_samples)
        # print('shuffle => ')
        '''shuffle'''
        img_filenames, exp_filenames, lnd_filenames, dr_mask_filenames, au_mask_filenames, up_mask_filenames, \
        md_mask_filenames, bo_mask_filenames = shuffle(img_filenames, exp_filenames, lnd_filenames, dr_mask_filenames,
                                                       au_mask_filenames, up_mask_filenames, md_mask_filenames,
                                                       bo_mask_filenames)
        # print('<== shuffle ')
        return img_filenames, exp_filenames, lnd_filenames, dr_mask_filenames, au_mask_filenames, \
               up_mask_filenames, md_mask_filenames, bo_mask_filenames

    def create_input_bunches(self, img_batch, dr_mask_batch, au_mask_batch, spatial_mask):
        # return img_batch

        # img_batch = np.expand_dims(np.mean(img_batch, axis=3), axis=-1)

        if spatial_mask is None:
            bunch = np.concatenate((img_batch, dr_mask_batch, au_mask_batch), axis=-1)
        else:
            img_batch = img_batch * spatial_mask
            dr_mask_batch = dr_mask_batch * spatial_mask
            au_mask_batch = au_mask_batch * spatial_mask
            bunch = np.concatenate((img_batch, dr_mask_batch, au_mask_batch), axis=-1)
        return bunch

    def get_batch_sample_online(self, batch_index, img_path, annotation_path, img_filenames,
                                exp_filenames, lnd_filenames, batch_size=None):
        if batch_size is None:
            batch_size = LearningConfig.batch_size

        img_path = img_path
        pn_tr_path = annotation_path
        '''create batch data and normalize images'''
        batch_img_names = img_filenames[
                          batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_exp_names = exp_filenames[
                          batch_index * batch_size:(batch_index + 1) * batch_size]

        '''create img and annotations'''
        # exp
        exp_batch = np.int8(np.array([load(pn_tr_path + 'exp_slnd/' + file_name) for file_name in batch_exp_names]))
        # images
        img_batch = np.array([imread(img_path + file_name) for file_name in batch_img_names]) / 255.0
        # '''derivative masks'''
        # dr_mask_batch = np.array([np.expand_dims(
        #     self.create_derivative(img=np.float32(np.array(imread(img_path + file_name)) / 255.0),
        #                            lnd=load(pn_tr_path + 'exp_slnd/' + file_name[:-4] + "_slnd.npy"))
        #     , axis=-1)
        #     for file_name in batch_img_names])
        dr_mask_batch = None
        '''action unit masks'''
        au_mask_batch = np.array([np.expand_dims(
            self.create_AU_mask(img=np.float32(np.array(imread(img_path + file_name)) / 255.0),
                                lnd=load(pn_tr_path + 'exp_slnd/' + file_name[:-4] + "_slnd.npy"))
            , axis=-1)
            for file_name in batch_img_names])

        '''spatial unit masks'''
        up_mask_batch = np.array([np.expand_dims(
            self.create_spatial_mask_single_part(img=np.float32(np.array(imread(img_path + file_name)) / 255.0),
                                                 lnd=load(pn_tr_path + 'exp_slnd/' + file_name[:-4] + "_slnd.npy"),
                                                 part=0), axis=-1) for file_name in batch_img_names])
        md_mask_batch = np.array([np.expand_dims(
            self.create_spatial_mask_single_part(img=np.float32(np.array(imread(img_path + file_name)) / 255.0),
                                                 lnd=load(pn_tr_path + 'exp_slnd/' + file_name[:-4] + "_slnd.npy"),
                                                 part=1), axis=-1) for file_name in batch_img_names])
        bo_mask_batch = np.array([np.expand_dims(
            self.create_spatial_mask_single_part(img=np.float32(np.array(imread(img_path + file_name)) / 255.0),
                                                 lnd=load(pn_tr_path + 'exp_slnd/' + file_name[:-4] + "_slnd.npy"),
                                                 part=2), axis=-1) for file_name in batch_img_names])
        '''global feature bunch'''
        global_bunch = self.create_input_bunches(img_batch=img_batch, dr_mask_batch=dr_mask_batch,
                                                 au_mask_batch=au_mask_batch,
                                                 spatial_mask=None)
        '''Upper feature bunch'''
        upper_bunch = self.create_input_bunches(img_batch=img_batch, dr_mask_batch=dr_mask_batch,
                                                au_mask_batch=au_mask_batch,
                                                spatial_mask=up_mask_batch)
        '''Middle feature bunch'''
        middle_bunch = self.create_input_bunches(img_batch=img_batch, dr_mask_batch=dr_mask_batch,
                                                 au_mask_batch=au_mask_batch,
                                                 spatial_mask=md_mask_batch)
        '''Bottom feature bunch'''
        bottom_bunch = self.create_input_bunches(img_batch=img_batch, dr_mask_batch=dr_mask_batch,
                                                 au_mask_batch=au_mask_batch,
                                                 spatial_mask=bo_mask_batch)
        '''clear memory'''
        dr_mask_batch = None
        au_mask_batch = None
        bo_mask_batch = None
        md_mask_batch = None
        up_mask_batch = None
        bo_3l_mask_batch = None
        md_3l_mask_batch = None
        up_3l_mask_batch = None
        # '''test print'''
        # for i in range(LearningConfig.batch_size): # bs, 224, 224, 5
        #     self.test_image_print(str(batch_index + 1 * (i + 1)) + 'fer_0_img_', bottom_bunch[i, :, :, :3], [])
        #     self.test_image_print(str(batch_index + 1 * (i + 1)) + 'fer_1_dr_', bottom_bunch[i, :, :, 3], [], cmap='gray')
        #     self.test_image_print(str(batch_index + 1 * (i + 1)) + 'fer_2_au_', bottom_bunch[i, :, :, 4], [], cmap='gray')

        return global_bunch, upper_bunch, middle_bunch, bottom_bunch, exp_batch

    def get_batch_sample_masked(self, batch_index, img_path, annotation_path, exp_filenames,
                                face_img_filenames, eyes_img_filenames, nose_img_filenames, mouth_img_filenames,
                                batch_size=None):
        if batch_size is None:
            batch_size = LearningConfig.batch_size

        img_path = img_path
        pn_tr_path = annotation_path
        '''create batch data and normalize images'''
        batch_face_img_filenames = face_img_filenames[
                                   batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_eyes_img_filenames = eyes_img_filenames[
                                   batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_nose_img_filenames = nose_img_filenames[
                                   batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_mouth_img_filenames = mouth_img_filenames[
                                    batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_exp_names = exp_filenames[
                          batch_index * batch_size:(batch_index + 1) * batch_size]
        exp_batch = np.array([load(pn_tr_path + file_name) for file_name in batch_exp_names])
        global_bunch = np.array([load(img_path + file_name)['arr_0'] for file_name in batch_face_img_filenames])
        upper_bunch = np.array([load(img_path + file_name)['arr_0'] for file_name in batch_eyes_img_filenames])
        middle_bunch = np.array([load(img_path + file_name)['arr_0'] for file_name in batch_nose_img_filenames])
        bottom_bunch = np.array([load(img_path + file_name)['arr_0'] for file_name in batch_mouth_img_filenames])

        return global_bunch, upper_bunch, middle_bunch, bottom_bunch, exp_batch

    def get_batch_sample(self, batch_index, img_path, annotation_path, img_filenames, exp_filenames, lnd_filenames,
                         dr_mask_filenames, au_mask_filenames, up_mask_filenames, md_mask_filenames, bo_mask_filenames,
                         batch_size=None):
        if batch_size is None:
            batch_size = LearningConfig.batch_size

        img_path = img_path
        pn_tr_path = annotation_path
        '''create batch data and normalize images'''
        batch_img_names = img_filenames[
                          batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_exp_names = exp_filenames[
                          batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_dr_names = dr_mask_filenames[
                         batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_au_names = au_mask_filenames[
                         batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_up_names = up_mask_filenames[
                         batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_md_names = md_mask_filenames[
                         batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_bo_names = bo_mask_filenames[
                         batch_index * batch_size:(batch_index + 1) * batch_size]
        print('create img and annotations')
        '''create img and annotations'''
        # img_batch = np.array(
        #     [self._do_online_random_aug(imread(img_path + file_name)) for file_name in batch_img_names]) / 255.0

        # exp
        exp_batch = np.array([load(pn_tr_path + 'exp_slnd/' + file_name) for file_name in batch_exp_names])
        # images
        img_batch = np.array([imread(img_path + file_name) for file_name in batch_img_names]) / 255.0
        # derivative masks
        print('derivative masks')
        dr_mask_batch = np.array([np.expand_dims(imread(pn_tr_path + 'dmg/' + file_name), axis=-1)
                                  for file_name in batch_dr_names]) / 255.0
        # action unit masks
        print('action unit masks')
        au_mask_batch = np.array([np.expand_dims(imread(pn_tr_path + 'im/' + file_name), axis=-1)
                                  for file_name in batch_au_names]) / 255.0
        # upper
        print('upper')
        up_mask_batch = np.array([np.expand_dims(imread(pn_tr_path + 'spm/' + file_name), axis=-1)
                                  for file_name in batch_up_names]) / 255.0
        # up_3l_mask_batch = np.array([np.stack([imread(pn_tr_path + 'spm/' + file_name),
        #                                        imread(pn_tr_path + 'spm/' + file_name),
        #                                        imread(pn_tr_path + 'spm/' + file_name)
        #                                        ],
        #                                       axis=-1)
        #                              for file_name in batch_up_names]) / 255.0
        # middle
        print('middle')
        md_mask_batch = np.array([np.expand_dims(imread(pn_tr_path + 'spm/' + file_name), axis=-1)
                                  for file_name in batch_md_names]) / 255.0
        # md_3l_mask_batch = np.array([np.stack([imread(pn_tr_path + 'spm/' + file_name),
        #                                        imread(pn_tr_path + 'spm/' + file_name),
        #                                        imread(pn_tr_path + 'spm/' + file_name)
        #                                        ],
        #                                       axis=-1)
        #                              for file_name in batch_md_names]) / 255.0
        # bottom
        print('bottom')
        bo_mask_batch = np.array([np.expand_dims(imread(pn_tr_path + 'spm/' + file_name), axis=-1)
                                  for file_name in batch_bo_names]) / 255.0
        # bo_3l_mask_batch = np.array([np.stack([imread(pn_tr_path + 'spm/' + file_name),
        #                                        imread(pn_tr_path + 'spm/' + file_name),
        #                                        imread(pn_tr_path + 'spm/' + file_name)
        #                                        ],
        #                                       axis=-1)
        #                              for file_name in batch_bo_names]) / 255.0

        '''global feature bunch'''
        print('global feature bunch')
        global_bunch = self.create_input_bunches(img_batch=img_batch, dr_mask_batch=dr_mask_batch,
                                                 au_mask_batch=au_mask_batch,
                                                 spatial_mask=None)
        '''Upper feature bunch'''
        upper_bunch = self.create_input_bunches(img_batch=img_batch, dr_mask_batch=dr_mask_batch,
                                                au_mask_batch=au_mask_batch,
                                                spatial_mask=up_mask_batch, )
        '''Middle feature bunch'''
        middle_bunch = self.create_input_bunches(img_batch=img_batch, dr_mask_batch=dr_mask_batch,
                                                 au_mask_batch=au_mask_batch,
                                                 spatial_mask=md_mask_batch)
        '''Bottom feature bunch'''
        bottom_bunch = self.create_input_bunches(img_batch=img_batch, dr_mask_batch=dr_mask_batch,
                                                 au_mask_batch=au_mask_batch,
                                                 spatial_mask=bo_mask_batch)
        '''test print'''
        # for i in range(LearningConfig.batch_size): # bs, 224, 224, 5
        #     self.test_image_print(str(batch_index + 1 * (i + 1)) + 'fer_0_img_', global_bunch[i, :, :, :3], [])
        #     self.test_image_print(str(batch_index + 1 * (i + 1)) + 'fer_1_dr_', global_bunch[i, :, :, 3], [])
        #     self.test_image_print(str(batch_index + 1 * (i + 1)) + 'fer_2_au_', global_bunch[i, :, :, 4], [])
        return global_bunch, upper_bunch, middle_bunch, bottom_bunch, exp_batch
        # return img_batch, val_batch, exp_batch, lnd_batch, lnd_avg_batch

    def _do_online_random_aug(self, image):
        try:
            img = self._adjust_gamma(image)
            img = self._blur(img)
            # img = self._noisy(img)
            return img
        except Exception as e:
            print(e)
        return image

    def _blur(self, image, _do=False):
        do_or_not = random.randint(0, 100)
        if _do or do_or_not % 2 == 0:
            try:
                # image = image * 255.0
                image = np.float32(image)
                image = cv2.medianBlur(image, 5)
                # image = image / 255.0
            except Exception as e:
                print(str(e))
                pass
            return image

        return image

    def _add_occlusion(self, image):
        try:
            do_or_not = random.randint(0, 10)
            if do_or_not % 2 == 0:
                for i in range(5):
                    do_or_not = random.randint(0, 10)
                    if do_or_not % 2 == 0:
                        start = (random.randint(0, 170), random.randint(0, 170))
                        extent = (random.randint(10, 50), random.randint(10, 50))
                        rr, cc = rectangle(start, extent=extent, shape=image.shape)
                        color = (np.random.uniform(0, 1), random.randint(0, 1), random.randint(0, 1))
                        set_color(image, (rr, cc), color, alpha=1.0)
                        # image[rr, cc] = 1.0
        except Exception as e:
            print('_add_occlusion:: ' + str(e))
        return image

    def _reverse_gamma(self, image):
        try:
            image = image * 255
            image = np.int8(image)
            gamma = 3.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            image = cv2.LUT(image, table)
            image = image / 255.0
            return image
        except Exception as e:
            print(str(e))
            pass
        return image

    def _adjust_gamma(self, image):
        do_or_not = random.randint(0, 100)
        if do_or_not % 2 == 0 or do_or_not % 3 == 0:
            try:
                image = image * 255
                image = np.int8(image)

                dark_or_light = random.randint(0, 100)
                if dark_or_light % 2 == 0 or dark_or_light % 3 == 0:
                    gamma = np.random.uniform(0.3, 0.8)
                else:
                    gamma = np.random.uniform(1.5, 3.5)
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255
                                  for i in np.arange(0, 256)]).astype("uint8")
                image = cv2.LUT(image, table)
                image = image / 255.0
                return image
            except Exception as e:
                print(str(e))
                pass
        return image

    def _noisy(self, image):
        noise_typ = random.randint(0, 8)
        if noise_typ == 0:
            s_vs_p = 0.1
            amount = 0.1
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        if 1 <= noise_typ <= 2:  # "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        else:
            return image

    def load_and_relable(self, file_name):
        lbl = int(load(file_name))
        if lbl == 6:
            lbl = 3
        return lbl

    def _create_image_and_labels_name_online(self, img_path, annotation_path, label, num_of_samples):
        img_filenames = []
        exp_filenames = []
        lnd_filenames = []

        if num_of_samples is None:
            file_names = os.listdir(img_path)
        else:
            print('reading list -->')
            file_names = os.listdir(img_path)
            print('<-')

        for file in file_names:
            if file.endswith(".jpg") or file.endswith(".png"):
                exp_lbl_file = str(file)[:-4] + "_exp.npy"  # just name
                lnd_lbl_file = str(file)[:-4] + "_slnd.npy"  # just name

                if os.path.exists(annotation_path + 'exp_slnd/' + exp_lbl_file):
                    if label is not None:
                        exp = np.load(annotation_path + 'exp_slnd/' + exp_lbl_file)
                        if label is not None and exp != label:
                            continue

                    img_filenames.append(str(file))
                    exp_filenames.append(exp_lbl_file)
                    lnd_filenames.append(lnd_lbl_file)

        return np.array(img_filenames), np.array(exp_filenames), np.array(lnd_filenames),

    def _create_masked_image_and_labels_name_full_path(self, img_path, annotation_path, label, num_of_samples):
        face_img_filenames = []
        eyes_img_filenames = []
        nose_img_filenames = []
        mouth_img_filenames = []
        exp_filenames = []

        if num_of_samples is None:
            file_names = os.listdir(img_path)
        else:
            print('reading list -->')
            file_names = tqdm(os.listdir(img_path))
            # file_names = os.listdir(img_path)
            print('<-')

        for file in file_names:
            if file.endswith("_face.npz"):
                exp_lbl_file = str(file)[:-9] + "_exp.npy"  # just name

                if os.path.exists(annotation_path + exp_lbl_file):
                    if label is not None:
                        exp = np.load(annotation_path + exp_lbl_file)
                        if label is not None and exp != label:
                            continue

                    face_img_filenames.append(img_path + str(file))
                    eyes_img_filenames.append(img_path + str(str(file)[:-9] + "_eyes.npz"))
                    nose_img_filenames.append(img_path + str(str(file)[:-9] + "_nose.npz"))
                    mouth_img_filenames.append(img_path + str(str(file)[:-9] + "_mouth.npz"))
                    exp_filenames.append(annotation_path + exp_lbl_file)

        return np.array(face_img_filenames), np.array(eyes_img_filenames), \
               np.array(nose_img_filenames), np.array(mouth_img_filenames), np.array(exp_filenames)

    def _create_image_and_labels_name(self, img_path, annotation_path, label, num_of_samples):
        face_img_filenames = []
        eyes_img_filenames = []
        nose_img_filenames = []
        mouth_img_filenames = []
        exp_filenames = []

        if num_of_samples is None:
            file_names = os.listdir(img_path)
        else:
            print('reading list -->')
            # file_names = tqdm(os.listdir(img_path))
            file_names = os.listdir(img_path)
            print('<-')

        for file in file_names:
            if file.endswith("_face.npz"):
                exp_lbl_file = str(file)[:-9] + "_exp.npy"  # just name

                if os.path.exists(annotation_path + exp_lbl_file):
                    if label is not None:
                        exp = np.load(annotation_path + exp_lbl_file)
                        if label is not None and exp != label:
                            continue

                    face_img_filenames.append(str(file))
                    eyes_img_filenames.append(str(str(file)[:-9] + "_eyes.npz"))
                    nose_img_filenames.append(str(str(file)[:-9] + "_nose.npz"))
                    mouth_img_filenames.append(str(str(file)[:-9] + "_mouth.npz"))
                    exp_filenames.append(exp_lbl_file)

        return np.array(face_img_filenames), np.array(eyes_img_filenames), \
               np.array(nose_img_filenames), np.array(mouth_img_filenames), np.array(exp_filenames)

    def __create_image_and_labels_name(self, img_path, annotation_path, label, num_of_samples):
        img_filenames = []
        exp_filenames = []
        lnd_filenames = []
        dr_mask_filenames = []
        au_mask_filenames = []
        up_mask_filenames = []
        md_mask_filenames = []
        bo_mask_filenames = []

        if num_of_samples is None:
            file_names = os.listdir(img_path)
        else:
            print('reading list -->')
            file_names = tqdm(os.listdir(img_path))
            print('<-')
            # file_names1 = [str(i)+'.jpg' for i in range(num_of_samples)]
            # file_names = [str(i)+'.jpg' for i in range(num_of_samples)]

        for file in file_names:
            if file.endswith(".jpg") or file.endswith(".png"):
                exp_lbl_file = str(file)[:-4] + "_exp.npy"  # just name
                lnd_lbl_file = str(file)[:-4] + "_slnd.npy"  # just name
                dr_mask_lbl_file = str(file)[:-4] + "_dmg.jpg"  # just name
                au_mask_lbl_file = str(file)[:-4] + "_im.jpg"  # just name
                up_mask_lbl_file = str(file)[:-4] + "_spm_up.jpg"  # just name
                md_mask_lbl_file = str(file)[:-4] + "_spm_md.jpg"  # just name
                bo_mask_lbl_file = str(file)[:-4] + "_spm_bo.jpg"  # just name

                if os.path.exists(annotation_path + 'exp_slnd/' + exp_lbl_file):
                    if label is not None:
                        exp = np.load(annotation_path + 'exp_slnd/' + exp_lbl_file)
                        if label is not None and exp != label:
                            continue

                    img_filenames.append(str(file))
                    exp_filenames.append(exp_lbl_file)
                    lnd_filenames.append(lnd_lbl_file)
                    dr_mask_filenames.append(dr_mask_lbl_file)
                    au_mask_filenames.append(au_mask_lbl_file)
                    up_mask_filenames.append(up_mask_lbl_file)
                    md_mask_filenames.append(md_mask_lbl_file)
                    bo_mask_filenames.append(bo_mask_lbl_file)

        return np.array(img_filenames), np.array(exp_filenames), \
               np.array(lnd_filenames), np.array(dr_mask_filenames), \
               np.array(au_mask_filenames), np.array(up_mask_filenames), \
               np.array(md_mask_filenames), np.array(bo_mask_filenames),

    def load_and_normalize(self, point_path):
        annotation = load(point_path)

        """for training we dont normalize COFW"""

        '''normalize landmarks based on hyperface method'''
        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((x_center - annotation[p]) / width)
            annotation_norm.append((y_center - annotation[p + 1]) / height)
        return annotation_norm

    def de_normalized(self, annotation_norm):
        """for training we dont normalize"""
        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation = []
        for p in range(0, len(annotation_norm), 2):
            annotation.append(x_center - annotation_norm[p] * width)
            annotation.append(y_center - annotation_norm[p + 1] * height)
        return annotation

    def test_image_print(self, img_name, img, landmarks, bbox_me=None, cmap=None):
        # print(img_name)
        # print(landmarks)
        plt.figure()
        if cmap is not None:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        ''''''
        if bbox_me is not None:
            bb_x = [bbox_me[0], bbox_me[2], bbox_me[4], bbox_me[6]]
            bb_y = [bbox_me[1], bbox_me[3], bbox_me[5], bbox_me[7]]
            plt.scatter(x=bb_x[:], y=bb_y[:], c='red', s=15)

        ''''''
        # print(landmarks)
        landmarks_x = []
        landmarks_y = []
        for i in range(0, len(landmarks), 2):
            landmarks_x.append(landmarks[i])
            landmarks_y.append(landmarks[i + 1])

        for i in range(len(landmarks_x)):
            plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=9, color='red')

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#000000', s=15)
        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fddb3a', s=8)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig('z_' + img_name + '.png', bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()

    def create_synthesized_landmarks_path(self, img_path, anno_path, file, model, test_print):
        landmark_name = anno_path + file[:-4] + "_slnd.npy"
        img = np.float32(Image.open(os.path.join(img_path, file))) / 255.0
        '''synthesize landmark'''
        anno_Pre = self.de_normalized(annotation_norm=model.predict(np.expand_dims(img, axis=0))[0])
        np.save(landmark_name, anno_Pre)

        if test_print:
            self.test_image_print(img_name='z_' + str(file) + '_img', img=img,
                                  landmarks=anno_Pre)

    def create_spatial_mask_single_part(self, img, lnd, part):
        up_mask, mid_mask, bot_mask = self._spatial_masks(landmarks=lnd, img=img)
        if part == 0:
            return up_mask
        elif part == 1:
            return mid_mask
        elif part == 2:
            return bot_mask

    def create_spatial_mask(self, img, lnd):
        up_mask, mid_mask, bot_mask = self._spatial_masks(landmarks=lnd, img=img)
        return up_mask, mid_mask, bot_mask

    def create_spatial_mask_path(self, img_path, anno_path, file, test_print=False):
        lnd = np.load(os.path.join(anno_path, file[:-4] + "_slnd.npy"))
        img_file_name = os.path.join(img_path, file)
        img = np.float32(Image.open(img_file_name)) / 255.0
        '''create mask'''
        up_mask, mid_mask, bot_mask = self._spatial_masks(landmarks=lnd, img=img)

        up_mask = np.stack([up_mask, up_mask, up_mask], axis=2) * img
        mid_mask = np.stack([mid_mask, mid_mask, mid_mask], axis=2) * img
        bot_mask = np.stack([bot_mask, bot_mask, bot_mask], axis=2) * img

        # img_mean = np.mean(img, axis=-1)

        # up_mask_inverted = (1 - up_mask) * img_mean
        # md_mask_inverted = (1 - mid_mask) * img_mean
        # bo_mask_inverted = (1 - bot_mask) * img_mean

        # up_fused_img = 0.1 * img_mean + 0.9 * (up_mask * img_mean)
        # mid_fused_img = 0.1 * img_mean + 0.9 * (mid_mask * img_mean)
        # bot_fused_img = 0.1 * img_mean + 0.9 * (bot_mask * img_mean)

        '''save'''
        # up_mask_name = os.path.join(anno_path, file[:-4] + "_spm_up.npz")
        # md_mask_name = os.path.join(anno_path, file[:-4] + "_spm_md.npz")
        # bo_mask_name = os.path.join(anno_path, file[:-4] + "_spm_bo.npz")
        # np.savez_compressed(up_mask_name, up_mask)
        # np.savez_compressed(md_mask_name, mid_mask)
        # np.savez_compressed(bo_mask_name, bot_mask)

        im = Image.fromarray(np.round(up_mask * 255.0).astype(np.uint8))
        im.save(anno_path + 'spm/' + file[:-4] + "_spm_up.jpg")
        im = Image.fromarray(np.round(mid_mask * 255.0).astype(np.uint8))
        im.save(anno_path + 'spm/' + file[:-4] + "_spm_md.jpg")
        im = Image.fromarray(np.round(bot_mask * 255.0).astype(np.uint8))
        im.save(anno_path + 'spm/' + file[:-4] + "_spm_bo.jpg")

        '''test mask '''
        if test_print:
            self.test_image_print(img_name='z_up_mask_' + str(file) + '_fused_img', img=up_mask,
                                  landmarks=lnd, cmap='gray')
            self.test_image_print(img_name='z_mid_mask_' + str(file) + '_fused_img', img=mid_mask,
                                  landmarks=lnd, cmap='gray')
            self.test_image_print(img_name='z_bot_mask_' + str(file) + '_fused_img', img=bot_mask,
                                  landmarks=lnd, cmap='gray')

    def create_AU_mask(self, img, lnd):
        '''create mask'''
        i_mask = self._inner_mask(landmarks=lnd)
        img_mean = np.mean(img, axis=-1)
        i_mask_inverted = (1 - i_mask) * img_mean
        i_mask = i_mask * img_mean
        i_mask = i_mask + 0.1 * i_mask_inverted
        return i_mask

    def create_AU_mask_path(self, img_path, anno_path, file, test_print=False):
        # exp = int(np.load(os.path.join(anno_path, file[:-4] + "_exp.npy")))
        lnd = np.load(os.path.join(anno_path + 'exp_slnd/', file[:-4] + "_slnd.npy"))
        img_file_name = os.path.join(img_path, file)
        img = np.float32(Image.open(img_file_name)) / 255.0

        '''create mask'''
        i_mask = self._inner_mask(landmarks=lnd)
        img_mean = np.mean(img, axis=-1)
        i_mask_inverted = (1 - i_mask) * img_mean
        i_mask = i_mask * img_mean
        i_mask = i_mask + 0.1 * i_mask_inverted
        #
        '''save'''
        # i_mask_name = os.path.join(anno_path, file[:-4] + "_im.npz")
        # np.savez_compressed(i_mask_name, i_mask)

        im = Image.fromarray(np.round(i_mask * 255.0).astype(np.uint8))
        im.save(anno_path + 'im/' + file[:-4] + "_im.jpg")

        '''test mask '''
        if test_print:
            self.test_image_print(img_name='z_mask_' + str(file) + '_fused_img', img=i_mask,
                                  landmarks=[], cmap='gray')

    def create_derivative(self, img, lnd):
        img_mean = np.mean(np.array(img), axis=-1)

        _, _, mag = self._hog(image=img_mean)
        mag = abs(mag)
        lnd_mask = self._landmark_mask(img_mean, lnd)
        mag = self._blur(mag, _do=True)
        mag = img_mean * mag + (1 + 0.5) * (mag - img_mean * mag)
        mag = lnd_mask * mag  # this mask is hard
        mag = self._normalize_image(mag)
        return mag

    def create_derivative_path(self, img_path, anno_path, file, test_print=False):
        # exp = int(np.load(os.path.join(anno_path, file[:-4] + "_exp.npy")))
        lnd = np.load(os.path.join(anno_path + 'exp_slnd/', file[:-4] + "_slnd.npy"))
        img_file_name = os.path.join(img_path, file)
        img = np.float32(Image.open(img_file_name)) / 255.0

        # img = self._median_filter(img)
        # img_eq = exposure.equalize_hist(img)
        # img_eq = np.mean(np.array(img_eq), axis=-1)

        img_mean = np.mean(np.array(img), axis=-1)

        gx, gy, mag = self._hog(image=img_mean)

        gx = abs(gx)
        gy = abs(gy)
        mag = abs(mag)

        lnd_mask = self._landmark_mask(img_mean, lnd)
        # gx = lnd_mask * gx  # this mask is hard
        # gy = lnd_mask * gy  # this mask is hard

        # gx = exposure.equalize_hist(gx)
        # gy = exposure.equalize_hist(gy)
        # mag = exposure.equalize_hist(mag)
        # mag = exposure.rescale_intensity(mag, in_range=(0.0, 1.0))

        # mag = exposure.equalize_hist(mag)
        # mag[mag <= 0.3] = 0
        mag = self._blur(mag, _do=True)
        mag = img_mean * mag + (1 + 0.5) * (mag - img_mean * mag)
        # mag = 0.1 * img_mean + 0.9 * mag

        mag = lnd_mask * mag  # this mask is hard
        mag = self._normalize_image(mag)

        '''save hog mask'''
        # gx_mask_name = os.path.join(anno_path, file[:-4] + "_dgx.npz")
        # gy_mask_name = os.path.join(anno_path, file[:-4] + "_dgy.npz")
        # mag_mask_name = os.path.join(anno_path, file[:-4] + "_dmg.npz")

        # np.savez_compressed(gx_mask_name, gx)
        # np.savez_compressed(gy_mask_name, gy)
        # np.savez_compressed(mag_mask_name, mag)

        im = Image.fromarray(np.round(mag * 255.0).astype(np.uint8))
        im.save(anno_path + 'dmg/' + file[:-4] + "_dmg.jpg")

        if test_print:
            # self.test_image_print(img_name='z_HOG_' + str(file) + '_orig', img=fused_img, landmarks=lnd)
            # self.test_image_print(img_name=str(i) + '_mag', img=mask_mag, landmarks=[], cmap='gray')

            self.test_image_print(img_name='z_dr_' + str(file) + '_mask_mag', img=mag, landmarks=[],
                                  )
            # self.test_image_print(img_name='z_HOG_' + str(file) + '_mask_gx', img=gx, landmarks=[],
            #                       cmap='gray')
            # self.test_image_print(img_name='z_HOG_' + str(file) + '_mask_gy', img=gy, landmarks=[],
            #                       cmap='gray')

    def _spatial_masks(self, landmarks, img):
        """"""
        '''define mask points'''
        ''''''
        p_1 = (landmarks[2 * 36], landmarks[2 * 36 + 1])
        p_2 = (landmarks[2 * 45], landmarks[2 * 45 + 1])
        '''slop:'''
        m = (p_1[1] - p_2[1]) / (p_1[0] - p_2[0])  # m = D(y)/D(x)
        # b = p_1[1] - m * p_1[0]  # b = y -mx

        d_1 = math.sqrt((landmarks[2 * 27 + 1] - landmarks[2 * 30 + 1]) ** 2
                        + (landmarks[2 * 27] - landmarks[2 * 30]) ** 2)
        d_2 = math.sqrt((landmarks[2 * 33 + 1] - landmarks[2 * 51 + 1]) ** 2
                        + (landmarks[2 * 33] - landmarks[2 * 51]) ** 2)

        y_offset_1 = landmarks[2 * 40 + 1] + 0.2 * d_1 - m * landmarks[2 * 40]
        y_offset_2 = landmarks[2 * 33 + 1] + 0.5 * d_2 - m * landmarks[2 * 33]

        # x_1 = 0
        # x_2 = InputDataSize.image_input_size

        '''upper mask'''
        y_1_up = y_offset_1
        y_2_up = m * InputDataSize.image_input_size + y_offset_1
        up_m_p = np.array([
            (0, 0),
            (0, y_1_up),
            (InputDataSize.image_input_size, y_2_up),
            (InputDataSize.image_input_size, 0)
        ])
        up_mask = self._create_mask_from_points(up_m_p)

        '''mid mask'''
        y_1_mid = y_offset_2
        y_2_mid = m * InputDataSize.image_input_size + y_offset_2
        mid_m_p = np.array([
            (0, y_1_up),
            (0, y_1_mid),
            (InputDataSize.image_input_size, y_2_mid),
            (InputDataSize.image_input_size, y_2_up)
        ])
        mid_mask = self._create_mask_from_points(mid_m_p)

        '''low_mask'''
        y_1_mid = y_offset_2
        y_2_mid = m * InputDataSize.image_input_size + y_offset_2
        mid_m_p = np.array([
            (0, y_1_mid),
            (0, InputDataSize.image_input_size),
            (InputDataSize.image_input_size, InputDataSize.image_input_size),
            (InputDataSize.image_input_size, y_2_mid)
        ])
        low_mask = self._create_mask_from_points(mid_m_p)

        # '''test mask'''
        # img_mean = np.mean(np.array(img), axis=-1)
        # up_fused_img = 0.2 * img_mean + 0.8 * (up_mask * img_mean)
        # mid_fused_img = 0.2 * img_mean + 0.8 * (mid_mask * img_mean)
        # low_fused_img = 0.2 * img_mean + 0.8 * (low_mask * img_mean)
        #
        # plt.figure()
        # plt.imshow(low_fused_img)
        # plt.plot([x_1, x_2], [y_1_up, y_2_up])
        # plt.plot([x_1, x_2], [y_1_mid, y_2_mid])
        # plt.scatter(x=0, y=y_offset_1, c='red', s=25)
        # plt.scatter(x=0, y=y_offset_2, c='green', s=25)
        #
        # landmarks_x = []
        # landmarks_y = []
        # for i in range(0, len(landmarks), 2):
        #     landmarks_x.append(landmarks[i])
        #     landmarks_y.append(landmarks[i + 1])
        # for i in range(len(landmarks_x)):
        #     plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=8, color='blue')
        # plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#000000', s=15)
        # plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fddb3a', s=8)
        #
        # plt.axis('off')
        # plt.tight_layout(pad=0)
        # plt.savefig('z_' + str(landmarks[10]) + '.png')
        # plt.clf()

        return up_mask, mid_mask, low_mask

    def _landmark_mask(self, img, landmarks):
        """remove all parts that are outside of face"""

        face_lnds = []
        # '''define slope based on eyes'''
        # p_1 = (landmarks[2 * 36], landmarks[2 * 36 + 1])
        # p_2 = (landmarks[2 * 45], landmarks[2 * 45 + 1])
        # '''slop:'''
        # m = (p_1[0] - p_2[0]) / (p_1[1] - p_2[1])  # m inverse= D(x)/D(y)
        #
        # b = p_1[1] - m * p_1[0]  # b = y -mx
        # x_1_up = b
        # x_2_up = m * InputDataSize.image_input_size + b

        face_lnds.append((landmarks[2 * 0], 0))
        for i in range(17):
            face_lnds.append((landmarks[2 * i], landmarks[2 * i + 1]))
        face_lnds.append((landmarks[2 * 16], 0))
        lnd_mask = self._create_mask_from_points(np.array(face_lnds))
        return lnd_mask

    def _inner_mask(self, landmarks):
        """"""
        '''define mask points'''
        mask_anchors_location = np.array([(landmarks[2 * 17], landmarks[2 * 17 + 1]),
                                          (landmarks[2 * 19], landmarks[2 * 19 + 1]),
                                          (landmarks[2 * 24], landmarks[2 * 24 + 1]),
                                          (landmarks[2 * 26], landmarks[2 * 26 + 1]),
                                          (0.5 * (landmarks[2 * 46] + landmarks[2 * 14]),
                                           0.5 * (landmarks[2 * 46 + 1] + landmarks[2 * 14 + 1])),
                                          (0.3 * landmarks[2 * 29] + 0.7 * landmarks[2 * 13],
                                           0.5 * (landmarks[2 * 29 + 1] + landmarks[2 * 13 + 1])),
                                          (0.5 * (landmarks[2 * 54] + landmarks[2 * 12]),
                                           0.5 * (landmarks[2 * 54 + 1] + landmarks[2 * 12 + 1])),
                                          (0.5 * (landmarks[2 * 55] + landmarks[2 * 10]),
                                           0.5 * (landmarks[2 * 55 + 1] + landmarks[2 * 10 + 1])),
                                          (0.5 * (landmarks[2 * 57] + landmarks[2 * 8]),
                                           0.5 * (landmarks[2 * 57 + 1] + landmarks[2 * 8 + 1])),
                                          (0.5 * (landmarks[2 * 59] + landmarks[2 * 6]),
                                           0.5 * (landmarks[2 * 59 + 1] + landmarks[2 * 6 + 1])),
                                          (0.5 * (landmarks[2 * 4] + landmarks[2 * 48]),
                                           0.5 * (landmarks[2 * 4 + 1] + landmarks[2 * 48 + 1])),
                                          (0.7 * landmarks[2 * 3] + 0.3 * landmarks[2 * 29],
                                           0.5 * (landmarks[2 * 3 + 1] + landmarks[2 * 29 + 1])),
                                          (0.5 * (landmarks[2 * 2] + landmarks[2 * 41]),
                                           0.5 * (landmarks[2 * 2 + 1] + landmarks[2 * 41 + 1])),
                                          ])
        grid = self._create_mask_from_points(mask_anchors_location)
        return grid

    def _create_mask_from_points(self, mask_anchors_location):
        nx, ny = InputDataSize.image_input_size, InputDataSize.image_input_size

        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x, y)).T

        path = Path(mask_anchors_location)
        grid = path.contains_points(points)
        grid = np.array(grid.reshape((ny, nx)))

        return grid

    def _hog(self, image):
        """"""
        # fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
        #                     cells_per_block=(1, 1), visualize=True, multichannel=True)
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        # return hog_image_rescaled
        '''cv2 version'''
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        return gx, gy, mag

    def _unsharp_filter(self, image):
        gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_image = cv2.addWeighted(image, 1, gaussian_3, -0.5, 0, image)
        return unsharp_image

    def _binary_threshold_filter(self, img):
        return cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)[1]

    def _median_filter(self, img):
        return ndimage.median_filter(img, size=5)

    def _normalize_image(self, img):
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        return img

    def calculate_augmentation_rate(self, sample_count_by_class, base_aug_factor):
        aug_factor_by_class_major = np.ones_like(sample_count_by_class)
        aug_factor_by_class_freq = np.ones_like(sample_count_by_class)

        max_class = np.amax(sample_count_by_class) * base_aug_factor

        for i in range(len(sample_count_by_class)):
            div = max_class / sample_count_by_class[i]
            if div != 1 and sample_count_by_class[i] != 0:
                aug_factor_by_class_major[i] = int(div)
                if div - int(div) > 0:
                    k = np.ceil(1 / (div - int(div)))
                    aug_factor_by_class_freq[i] = int(k)

        return aug_factor_by_class_major, aug_factor_by_class_freq

    def _rotate_image_and_lnds(self, _img, _landmark, num_of_landmarks=68):
        img = np.copy(_img)
        landmark = np.copy(_landmark)

        fix_pad = InputDataSize.image_input_size * 2
        img = np.pad(img, ((fix_pad, fix_pad), (fix_pad, fix_pad), (0, 0)), 'wrap')
        for jj in range(len(landmark)):
            landmark[jj] = landmark[jj] + fix_pad

        scale = (0.75, 1.25)
        translation = (0, 0)
        shear = 0

        rot = np.random.uniform(-1 * 0.45, 0.45)
        sx, sy = scale
        t_matrix = np.array([
            [sx * math.cos(rot), -sy * math.sin(rot + shear), 0],
            [sx * math.sin(rot), sy * math.cos(rot + shear), 0],
            [0, 0, 1]
        ])
        tform = AffineTransform(scale=scale, rotation=rot, translation=translation, shear=np.deg2rad(shear))

        t_img = transform.warp(img, tform.inverse, mode='edge')
        '''affine landmark'''
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(landmark, 1, 1)
        label = np.array(landmark_arr_x + landmark_arr_y).reshape([2, num_of_landmarks])
        margin = np.ones([1, num_of_landmarks])
        label = np.concatenate((label, margin), axis=0)

        t_label = self._reorder(np.delete(np.dot(t_matrix, label), 2, axis=0)
                                .reshape([2 * num_of_landmarks]), num_of_landmarks)

        '''crop data: we add a small margin to the images'''
        c_img, landmark_new = self._crop_image(img=t_img, annotation=t_label)
        '''resize'''
        _img, _landmark = self.resize_image(c_img, landmark_new)

        return _img, _landmark

    def _flip_and_relabel(self, img, landmark, num_of_landmarks=68):
        t_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])

        '''flip landmark'''
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(landmark, 1, 1)
        label = np.array(landmark_arr_x + landmark_arr_y).reshape([2, num_of_landmarks])
        t_label = self._reorder(np.dot(t_matrix, label).reshape([2 * num_of_landmarks]), num_of_landmarks)

        '''we need to shift x'''
        for i in range(0, len(t_label), 2):
            t_label[i] = t_label[i] + img.shape[1]

        '''flip image'''
        img = np.fliplr(img)

        '''need to relabel '''
        t_label = self.relabel_ds(t_label)

        return img, t_label

    def crop_image_bbox(self, img, x_min, y_min, x_max, y_max):
        # rand_padd = 0
        rand_padd = random.randint(1, 5) * img.shape[0] / 100

        xmin = int(max(0, x_min - rand_padd))
        xmax = int(x_max + rand_padd)
        ymin = int(max(0, y_min - rand_padd - 5))
        ymax = int(y_max + rand_padd)
        croped_img = img[ymin:ymax, xmin:xmax]
        return croped_img

    def _crop_image(self, img, annotation):
        # rand_padd = random.randint(1, 5) * img.shape[0]/100
        rand_padd = random.randint(1, 25)

        ann_xy, ann_x, ann_y = self.create_landmarks(annotation, 1, 1)
        xmin = int(max(0, min(ann_x) - rand_padd))
        xmax = int(max(ann_x) + rand_padd)
        ymin = int(max(0, min(ann_y) - rand_padd - 20))
        ymax = int(max(ann_y) + rand_padd)
        annotation_new = []
        for i in range(0, len(annotation), 2):
            annotation_new.append(annotation[i] - xmin)
            annotation_new.append(annotation[i + 1] - ymin)
        croped_img = img[ymin:ymax, xmin:xmax]
        return croped_img, annotation_new

    def _reorder(self, input_arr, num_of_landmarks):
        out_arr = []
        for i in range(num_of_landmarks):
            out_arr.append(input_arr[i])
            k = num_of_landmarks + i
            out_arr.append(input_arr[k])
        return np.array(out_arr)

    def relabel_ds(self, labels):
        new_labels = np.copy(labels)

        index_src = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 31, 32, 36, 37, 38, 39, 40, 41, 48, 49, 50,
                     60, 61, 67, 59, 58]
        index_dst = [16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 35, 34, 45, 44, 43, 42, 47, 46, 54, 53, 52,
                     64, 63, 65, 55, 56]

        for i in range(len(index_src)):
            new_labels[index_src[i] * 2] = labels[index_dst[i] * 2]
            new_labels[index_src[i] * 2 + 1] = labels[index_dst[i] * 2 + 1]

            new_labels[index_dst[i] * 2] = labels[index_src[i] * 2]
            new_labels[index_dst[i] * 2 + 1] = labels[index_src[i] * 2 + 1]
        return new_labels

    def create_generator_full_path(self, img_path, annotation_path, label=None):
        img_filenames, exp_filenames, lnd_filenames = self._create_image_and_labels_name_full_path(img_path=img_path,
                                                                                                   annotation_path=annotation_path,
                                                                                                   label=label)
        return img_filenames, exp_filenames, lnd_filenames

    def create_generator_full_path_with_spm(self, img_path, annotation_path, label=None):
        img_filenames, exp_filenames, spm_up_filenames, spm_md_filenames, spm_bo_filenames =\
            self._create_image_and_labels_name_full_path_with_spm(img_path=img_path,
                                                                  annotation_path=annotation_path,
                                                                  label=label)
        return img_filenames, exp_filenames, spm_up_filenames, spm_md_filenames, spm_bo_filenames

    def _create_image_and_labels_name_full_path_with_spm(self, img_path, annotation_path, label):
        img_filenames = []
        exp_filenames = []
        spm_up_filenames = []
        spm_md_filenames = []
        spm_bo_filenames = []

        print('reading list -->')
        file_names = tqdm(os.listdir(img_path))
        print('<-')

        for file in file_names:
            if file.endswith(".jpg") or file.endswith(".png"):
                exp_lbl_file = str(file)[:-4] + "_exp.npy"  # just name

                spm_up_file = str(file)[:-4] + "_spm_up.jpg"
                spm_md_file = str(file)[:-4] + "_spm_md.jpg"
                spm_bo_file = str(file)[:-4] + "_spm_bo.jpg"

                if os.path.exists(annotation_path + exp_lbl_file):
                    if label is not None:
                        exp = np.load(annotation_path + exp_lbl_file)
                        if label is not None and exp != label:
                            continue

                    img_filenames.append(img_path + str(file))
                    exp_filenames.append(annotation_path + exp_lbl_file)

                    spm_up_filenames.append(annotation_path + 'spm/' + spm_up_file)
                    spm_md_filenames.append(annotation_path + 'spm/' + spm_md_file)
                    spm_bo_filenames.append(annotation_path + 'spm/' + spm_bo_file)

        return np.array(img_filenames), np.array(exp_filenames), \
               np.array(spm_up_filenames), np.array(spm_md_filenames), np.array(spm_bo_filenames)

    def _create_image_and_labels_name_full_path(self, img_path, annotation_path, label):
        img_filenames = []
        exp_filenames = []
        lnd_filenames = []

        print('reading list -->')
        file_names = tqdm(os.listdir(img_path))
        print('<-')

        for file in file_names:
            if file.endswith(".jpg") or file.endswith(".png"):
                exp_lbl_file = str(file)[:-4] + "_exp.npy"  # just name
                lnd_file = str(file)[:-4] + "_slnd.npy"  # just name

                if os.path.exists(annotation_path + exp_lbl_file):
                    if label is not None:
                        exp = np.load(annotation_path + exp_lbl_file)
                        if label is not None and exp != label:
                            continue

                    img_filenames.append(img_path + str(file))
                    exp_filenames.append(annotation_path + exp_lbl_file)
                    lnd_filenames.append(annotation_path + lnd_file)

        return np.array(img_filenames), np.array(exp_filenames), np.array(lnd_filenames)
