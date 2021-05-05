import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from config import DatasetName, ExpressionCodesRafdb, ExpressionCodesAffectnet
from keras import backend as K


class CustomLosses:
    # def __init__(self):
    #     pass

    def regressor_loss(self, y_gt, y_pr):
        loss = tf.reduce_mean(tf.square(y_gt - y_pr))
        return loss

    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # loss = tf.keras.losses.sparse_categorical_crossentropy(y_gt, y_pr, from_logits=False, axis=-1)

    def triplet_loss(self, y_gt, y_pr):
        """"""
        '''tfa.losses.TripletSemiHardLoss()'''
        triplet_loss_obj = tfa.losses.TripletSemiHardLoss()
        tr_loss = triplet_loss_obj(y_true=y_gt, y_pred=y_pr)
        return tr_loss

    def cross_entropy_loss(self, y_gt, y_pr, num_classes, ds_name):
        loss_weights = tf.ones_like(y_gt)
        y_gt_oh = tf.one_hot(y_gt, depth=num_classes)
        if ds_name == DatasetName.affectnet:
            # neutral happy sad surprise fear disgust anger
            weight_map = [2, 1, 3, 5, 7, 10, 3]
        elif ds_name == DatasetName.rafdb:
            # Surprise Fear Disgust Happiness Sadness Anger Neutral
            # [1290.  281.  717. 4772. 1982.  705. 2524.]
            weight_map = [3, 6, 4, 1, 2, 4, 2]

        # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
        #                                                       reduction=tf.keras.losses.Reduction.NONE)
        # loss_cross_entropy = loss_object(y_gt_oh, y_pr)

        y_pred = y_pr
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # calc
        loss = y_gt_oh * tf.math.log(y_pred) * weight_map
        loss = tf.reduce_mean(-tf.reduce_sum(loss))

        accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pr, y_gt_oh)) * 100.0

        # y_gt = tf.one_hot(y_gt, depth=num_classes)
        # loss_object = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        # loss_cross_entropy = loss_object(y_gt, y_pr)

        # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # loss_cross_entropy = loss_object(y_gt, y_pr)

        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss_cross_entropy = loss_object(y_gt, y_pr)

        # '''tfa.losses.TripletSemiHardLoss()'''
        # triplet_loss_obj = tfa.losses.TripletSemiHardLoss()
        # tr_loss_face = triplet_loss_obj(y_true=y_gt, y_pred=embedding_layer_face)
        # tr_loss_eyes = triplet_loss_obj(y_true=y_gt, y_pred=embedding_layer_eyes)
        # tr_loss_nose = triplet_loss_obj(y_true=y_gt, y_pred=embedding_layer_nose)
        # tr_loss_mouth = triplet_loss_obj(y_true=y_gt, y_pred=embedding_layer_mouth)

        return loss, accuracy
        # return 10 * loss_cross_entropy
