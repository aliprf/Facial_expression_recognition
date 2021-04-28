import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

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

    def cross_entropy_loss(self, y_gt, y_pr):
        y_gt = tf.one_hot(y_gt, depth=8)

        loss_object = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        loss_cross_entropy = loss_object(y_gt, y_pr)

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

        return 10 * loss_cross_entropy
