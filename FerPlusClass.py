from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, ExpressionCodesAffectnet
from config import LearningConfig, InputDataSize, DatasetName, FerPlusConf, DatasetType

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

class FerPlus:

    def __init__(self, ds_type):
        """we set the parameters needed during the whole class:
        """
        self.ds_type = ds_type
        if ds_type == DatasetType.train:
            self.img_path = FerPlusConf.no_aug_train_img_path
            self.anno_path = FerPlusConf.no_aug_train_annotation_path
            self.img_path_aug = FerPlusConf.aug_train_img_path
            self.anno_path_aug = FerPlusConf.aug_train_annotation_path
            self.masked_img_path = FerPlusConf.aug_train_masked_img_path

        elif ds_type == DatasetType.test:
            self.img_path = FerPlusConf.test_img_path
            self.anno_path = FerPlusConf.test_annotation_path
            self.img_path_aug = FerPlusConf.test_img_path
            self.anno_path_aug = FerPlusConf.test_annotation_path
            self.masked_img_path = FerPlusConf.test_masked_img_path

