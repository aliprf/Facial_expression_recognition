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
import tensorflow as tf
import random
import cv2 as cv


class DataHelper:

    def read_csv(self, ds_name, ds_type, FLD_model_file_name):
        if ds_name == DatasetName.affectnet:
            if ds_type == DatasetType.train:
                csv_path = AffectnetConf.orig_csv_train_path
                load_img_path = AffectnetConf.orig_img_path_prefix
                save_img_path = AffectnetConf.revised_train_img_path
                save_anno_path = AffectnetConf.revised_train_annotation_path
                do_aug = True
            elif ds_type == DatasetType.eval:
                csv_path = AffectnetConf.orig_csv_evaluate_path
                load_img_path = AffectnetConf.orig_img_path_prefix
                save_img_path = AffectnetConf.revised_eval_img_path
                save_anno_path = AffectnetConf.revised_eval_annotation_path
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
                                  arousal_arr=arousal_arr, FLD_model_file_name=FLD_model_file_name, do_aug=do_aug)
        return 0

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
                         expression_lbl_arr, valence_arr, arousal_arr, FLD_model_file_name, do_aug):

        # model = tf.keras.models.load_model(FLD_model_file_name)
        model = None

        for i in tqdm(range(len(img_path_arr))):
            if int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.none or \
                    int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.uncertain or \
                    int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.noface:
                continue
            elif int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.happy or \
                    int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.sad or \
                    int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.neutral or \
                    int(expression_lbl_arr[i]) == ExpressionCodesAffectnet.anger:
                '''crop, resize, augment image'''
                self.crop_resize_aug_img(load_img_name=load_img_path + img_path_arr[i],
                                         save_img_name=save_img_path + str(i) + '.jpg',
                                         bbox=bbox_arr[i], landmark=landmarks_arr[i],
                                         save_anno_name=save_anno_path + str(i) + '_lnd',
                                         synth_save_anno_name=save_anno_path + str(i) + '_slnd',
                                         model=model, do_aug=do_aug)
                '''save annotation: exp_lbl, valence, arousal, landmark '''
                save(save_anno_path + str(i) + '_exp', expression_lbl_arr[i])
                save(save_anno_path + str(i) + '_val', valence_arr[i])
                save(save_anno_path + str(i) + '_aro', arousal_arr[i])

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
        ''''''
        x, y, width, height = list(map(int, bbox))
        x_1 = int(x + 0.05 * width)  # max(0, x-0.1*width)
        y_1 = y  # max(0, y-0.1*height)
        x_2 = int(min(x + width + 0.15 * width, img.shape[0]))  # min(x+width+0.1*width, img.shape[0])
        y_2 = y + height  # int(min(y + height + 0.2 * height, img.shape[1]))
        # print(x_1, x_2, y_1, y_2)
        ''''''
        landmark = list(map(float, landmark))

        # croped_img = img[x:x+width, y:y+height]
        croped_img = img[x_1:x_2, y_1:y_2]
        annotation_new = []
        for i in range(0, len(landmark), 2):
            annotation_new.append(landmark[i] - x)
            annotation_new.append(landmark[i + 1] - y)

        '''resize'''
        resized_img, annotation_resized = self.resize_image(img=croped_img, annotation=annotation_new)
        '''synthesize lnd'''
        anno_Pre = 0  # self.de_normalized(annotation_norm=model.predict(np.expand_dims(resized_img, axis=0))[0])

        '''test print'''
        # self.test_image_print(img_name=save_img_name + '_synth', img=resized_img, landmarks=anno_Pre)
        # self.test_image_print(img_name=save_img_name + 'orig', img=resized_img, landmarks=annotation_resized)

        '''save'''
        im = Image.fromarray((resized_img * 255).astype(np.uint8))

        im.save(save_img_name)
        save(save_anno_name, annotation_resized)
        save(synth_save_anno_name, anno_Pre)

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

    def shuffle_data(self, filenames, labels):
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        return filenames_shuffled, y_labels_shuffled

    def create_test_gen(self, img_path, annotation_path):
        filenames, val_labels = self._create_image_and_labels_name(img_path=img_path, annotation_path=annotation_path)
        return filenames, val_labels

    def create_generators(self, dataset_name, img_path, annotation_path):
        # fn_prefix = './file_names/' + dataset_name + '_'
        # x_trains_path = fn_prefix + 'x_train_fns.npy'
        # x_validations_path = fn_prefix + 'x_val_fns.npy'

        filenames, val_labels = self._create_image_and_labels_name(img_path=img_path, annotation_path=annotation_path)
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, val_labels)
        x_train_filenames, x_val_filenames, y_train, y_val = train_test_split(
            filenames_shuffled, y_labels_shuffled, test_size=LearningConfig.batch_size, random_state=1)
        return x_train_filenames, x_val_filenames, y_train, y_val

    def create_evaluation_batch(self, x_eval_filenames, y_eval_filenames, img_path, annotation_path, mode):
        img_path = img_path
        pn_tr_path = annotation_path
        '''create batch data and normalize images'''
        batch_x = x_eval_filenames[0:LearningConfig.batch_size]
        batch_y = y_eval_filenames[0:LearningConfig.batch_size]
        '''create img and annotations'''
        eval_img_batch = np.array([imread(img_path + file_name) for file_name in batch_x]) / 255.0
        eval_exp_batch = np.array([self.load_and_relable(pn_tr_path + file_name[:-8] + "_exp.npy") for file_name in batch_y])
        # if mode == 0:
        #     eval_val_batch = np.array(
        #         [self.load_and_categorize_valence(pn_tr_path + file_name) for file_name in batch_y])
        # else:
        #     eval_val_batch = np.array([float(load(pn_tr_path + file_name)) for file_name in batch_y])

        # eval_lnd_batch = 0
        # eval_lnd_avg_batch = 0
        return eval_img_batch, eval_exp_batch
        # return eval_img_batch, eval_val_batch, eval_exp_batch, eval_lnd_batch, eval_lnd_avg_batch

    def get_batch_sample(self, batch_index, x_train_filenames, y_train_filenames, img_path, annotation_path, mode):
        img_path = img_path
        pn_tr_path = annotation_path
        '''create batch data and normalize images'''
        batch_x = x_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_y = y_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        '''create img and annotations'''
        img_batch = np.array([self._do_random_aug(imread(img_path + file_name)) for file_name in batch_x]) / 255.0
        exp_batch = np.array([self.load_and_relable(pn_tr_path + file_name[:-8] + "_exp.npy") for file_name in batch_y])
        # if mode==0:
        #     val_batch = np.array([self.load_and_categorize_valence(pn_tr_path + file_name) for file_name in batch_y])
        # else:
        #     val_batch = np.array([int(load(pn_tr_path + file_name)) for file_name in batch_y])

        # lnd_batch = 0
        # lnd_avg_batch = 0

        # '''test: print'''

        # for i in range(LearningConfig.batch_size):
        #     self.test_image_print(str(batch_index + 1 * (i + 1)) + 'fer', img_batch[i], [])

        return img_batch, exp_batch
        # return img_batch, val_batch, exp_batch, lnd_batch, lnd_avg_batch

    def _do_random_aug(self, image):
        try:
            img = self._adjust_gamma(image)
            img = self._blur(img)
            # img = self._noisy(img)
            return img
        except Exception as e:
            print(e)
        return image

    def _blur(self, image):
        do_or_not = random.randint(0, 100)
        if do_or_not % 2 == 0:
            try:
                # image = image * 255.0
                image = np.float32(image)
                image = cv.medianBlur(image, 5)
                # image = image / 255.0
            except Exception as e:
                print(str(e))
                pass
            return image

        return image

    def _adjust_gamma(self, image):
        do_or_not = random.randint(0, 100)
        if do_or_not % 2 == 0 or do_or_not % 3 == 0:
            try:
                # image = image * 255
                image = np.int8(image)

                dark_or_light = random.randint(0, 100)
                if dark_or_light % 2 == 0 or dark_or_light % 3 == 0:
                    gamma = np.random.uniform(0.3, 0.8)
                else:
                    gamma = np.random.uniform(1.5, 3.5)
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255
                                  for i in np.arange(0, 256)]).astype("uint8")
                image = cv.LUT(image, table)
                # image = image / 255.0
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

    def categorize_valence(self, orig_val):
        # print('orig_:' + str(orig_val))
        # if orig_val <= -0.5:
        #     val = 0
        # elif -0.5 < orig_val <= 0.0:
        #     val = 1
        # elif 0.0 < orig_val <= 0.5:
        #     val = 2
        # elif 0.5 < orig_val:
        #     val = 3
        # return val

        if -1.0 <= orig_val <= -0.66:
            val = 0
        elif -0.66 < orig_val <= -0.33:
            val = 1
        elif -0.33 < orig_val <= 0.0:
            val = 2
        elif 0.0 < orig_val <= 0.33:
            val = 3
        elif 0.33 < orig_val <= 0.66:
            val = 4
        elif 0.66 < orig_val <= 1.0:
            val = 5
        else:
            print('INCORRECT VALENCE')
            raise 0
        return val

    def load_and_relabel_exp(self, valence_path):
        orig_val = int(load(valence_path))
        if orig_val == 6:
            orig_val = 3
        return orig_val

    def load_and_categorize_valence(self, valence_path):
        orig_val = float(load(valence_path))
        # if -1.0 <= orig_val <= -0.5:
        #     val = 0
        # elif -0.5 < orig_val <= 0.0:
        #     val = 1
        # elif 0.0 < orig_val <= 0.5:
        #     val = 2
        # elif 0.5 < orig_val <= 1:
        #     val = 3
        # return val

        if -1.0 <= orig_val <= -0.66:
            val = 0
        elif -0.66 < orig_val <= -0.33:
            val = 1
        elif -0.33 < orig_val <= 0.0:
            val = 2
        elif 0.0 < orig_val <= 0.33:
            val = 3
        elif 0.33 < orig_val <= 0.66:
            val = 4
        elif 0.66 < orig_val <= 1.0:
            val = 5
        else:
            print('INCORRECT VALENCE')
            return 0
        return val

    def _create_image_and_labels_name(self, img_path, annotation_path):
        img_filenames = []
        lbls_filenames = []

        for file in os.listdir(img_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                val_lbl_file = str(file)[:-4] + "_exp.npy"  # just name
                if os.path.exists(annotation_path + val_lbl_file):
                    img_filenames.append(str(file))
                    lbls_filenames.append(val_lbl_file)

        return np.array(img_filenames), np.array(lbls_filenames)

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

    def test_image_print(self, img_name, img, landmarks, bbox_me=None):
        # print(img_name)
        # print(landmarks)
        plt.figure()
        plt.imshow(img)
        ''''''
        if bbox_me is not None:
            bb_x = [bbox_me[0], bbox_me[2], bbox_me[4], bbox_me[6]]
            bb_y = [bbox_me[1], bbox_me[3], bbox_me[5], bbox_me[7]]
            plt.scatter(x=bb_x[:], y=bb_y[:], c='red', s=15)

        ''''''
        landmarks_x = []
        landmarks_y = []
        for i in range(0, len(landmarks), 2):
            landmarks_x.append(landmarks[i])
            landmarks_y.append(landmarks[i + 1])

        for i in range(len(landmarks_x)):
            plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=9, color='red')

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#000000', s=15)
        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fddb3a', s=8)
        plt.savefig(img_name + '.png')
        # plt.show()
        plt.clf()
