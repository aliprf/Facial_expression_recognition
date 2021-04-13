import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

class CustomLosses:
    # def __init__(self):
    #     pass

    def regressor_loss(self, y_gt, y_pr):
        loss = tf.reduce_mean(tf.square(y_gt - y_pr))
        return loss

    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # loss = tf.keras.losses.sparse_categorical_crossentropy(y_gt, y_pr, from_logits=False, axis=-1)
    def cross_entropy_loss(self, y_gt, y_pr):
        pre_categorical = y_pr[0]
        embedding_layer_face = y_pr[1]
        embedding_layer_eyes = y_pr[2]
        embedding_layer_nose = y_pr[3]
        embedding_layer_mouth = y_pr[4]

        '''categorical loss'''
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_cross_entropy = loss_object(y_gt, pre_categorical)

        '''tfa.losses.TripletSemiHardLoss()'''
        triplet_loss_obj = tfa.losses.TripletSemiHardLoss()
        tr_loss_face = triplet_loss_obj(y_true=y_gt, y_pred=embedding_layer_face)
        tr_loss_eyes = triplet_loss_obj(y_true=y_gt, y_pred=embedding_layer_eyes)
        tr_loss_nose = triplet_loss_obj(y_true=y_gt, y_pred=embedding_layer_nose)
        tr_loss_mouth = triplet_loss_obj(y_true=y_gt, y_pred=embedding_layer_mouth)

        return loss_cross_entropy
