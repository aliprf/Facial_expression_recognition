
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

    def correlation_loss(self, exp_gt, exp_v, face_fv, eye_fv, nose_fv, mouth_fv):
        """
        :param exp_v: bs * 7
        :param face_fv: bs * 256
        :param eye_fv: bs * 256
        :param nose_fv: bs * 256
        :param mouth_fv: bs * 256
        :return:
        """
        exp_gt = tf.expand_dims(tf.one_hot(exp_gt, depth=7), -1)

        exp_v = tf.expand_dims(exp_v, -1)  # bs * 7 * 1
        face_fv = tf.expand_dims(face_fv, -1)  # bs * 256 * 1
        eye_fv = tf.expand_dims(eye_fv, -1)  # bs * 256 * 1
        nose_fv = tf.expand_dims(nose_fv, -1)  # bs * 256 * 1
        mouth_fv = tf.expand_dims(mouth_fv, -1)  # bs * 256 * 1
        '''we calculate dot of feature vectors and the final Probability '''
        face_exp_mat = tf.einsum('sab,sbc->sac', exp_v, tf.einsum('sab->sba', face_fv))  # 3 * 7 * 256
        eye_exp_mat = tf.einsum('sab,sbc->sac', exp_v, tf.einsum('sab->sba', eye_fv))  # 3 * 7 * 256
        nose_exp_mat = tf.einsum('sab,sbc->sac', exp_v, tf.einsum('sab->sba', nose_fv))  # 3 * 7 * 256
        mouth_exp_mat = tf.einsum('sab,sbc->sac', exp_v, tf.einsum('sab->sba', mouth_fv))   # 3 * 7 * 256
        '''Cov matrix'''
        face_exp_cor = np.array([np.corrcoef(face_exp_mat[i, :, :]) for i in range(face_exp_mat.shape[0])]) # 3 * 7 * 7
        eye_exp_cor = np.array([np.corrcoef(eye_exp_mat[i, :, :]) for i in range(eye_exp_mat.shape[0])])
        nose_exp_cor = np.array([np.corrcoef(nose_exp_mat[i, :, :]) for i in range(nose_exp_mat.shape[0])])
        mouth_exp_cor = np.array([np.corrcoef(mouth_exp_mat[i, :, :]) for i in range(mouth_exp_mat.shape[0])])
        '''creating the coeff'''


        pass

    def triplet_loss(self, y_gt, y_pr):
        """"""
        '''tfa.losses.TripletSemiHardLoss()'''
        triplet_loss_obj = tfa.losses.TripletSemiHardLoss()
        tr_loss = triplet_loss_obj(y_true=y_gt, y_pred=y_pr)
        return tr_loss

    def cross_entropy_loss_with_dynamic_loss(self, y_gt, y_pr, num_classes, conf_mat, ds_name):
        if ds_name == DatasetName.affectnet:
            # neutral happy sad surprise fear disgust anger
            fixed_weight_map = [
                [2, K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon()],
                [K.epsilon(), 1, K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon()],
                [K.epsilon(), K.epsilon(), 3, K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon()],
                [K.epsilon(), K.epsilon(), K.epsilon(), 5, K.epsilon(), K.epsilon(), K.epsilon()],
                [K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon(), 7, K.epsilon(), K.epsilon()],
                [K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon(), 10, K.epsilon()],
                [K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon(), K.epsilon(), 4]]
        elif ds_name == DatasetName.rafdb:
            # Surprise Fear Disgust Happiness Sadness Anger Neutral
            # [1290.  281.  717. 4772. 1982.  705. 2524.]
            fixed_weight_map = [3, 6, 4, 1, 2, 4, 2]

        conf_mat = conf_mat * (1 - np.identity(7, dtype=np.float))
        global_weight_map = fixed_weight_map + conf_mat

        weight_map = np.array([global_weight_map[y_gt[i], :] for i in range(len(y_gt))])

        '''calculate loss'''
        y_gt_oh = tf.one_hot(y_gt, depth=num_classes)
        y_pred = y_pr
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        categorical_loss = -(y_gt_oh * tf.math.log(y_pred) * weight_map)
        loss_reg = (1 - y_gt_oh) * tf.abs(y_gt_oh - y_pred) * weight_map

        loss = 10 * tf.reduce_mean(categorical_loss + loss_reg)
        accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pr, y_gt_oh)) * 100.0

        return 5.0 * loss, accuracy

    # def cross_entropy_loss_with_dynamic_loss(self, y_gt, y_pr, num_classes, conf_mat):
    #     y_pred_sparse = np.array([np.argmax(y_pr[i]) for i in range(len(y_pr))])
    #     # weight_map = np.array([conf_mat[y_gt[i], y_pred_sparse[i]] for i in range(len(y_pred_sparse))])
    #     weight_map = np.array([conf_mat[y_gt[i], :] for i in range(len(y_pred_sparse))])
    #
    #     y_gt_oh = tf.one_hot(y_gt, depth=num_classes)
    #     y_pred = y_pr
    #     y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    #     # clip
    #     y_pred = K.clip(y_pred, K.epsilon(), 1)
    #     # calc
    #     loss = -10.0*tf.reduce_mean(y_gt_oh * tf.math.log(y_pred) * weight_map)
    #
    #     # loss = y_gt_oh * tf.math.log(y_pred) * weight_map
    #     # loss = tf.reduce_mean(-tf.reduce_sum(loss))
    #
    #     accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pr, y_gt_oh))*100.0
    #
    #     return 5.0 * loss, accuracy

    def cross_entropy_loss(self, y_gt, y_pr, num_classes, ds_name):
        y_gt_oh = tf.one_hot(y_gt, depth=num_classes)
        if ds_name == DatasetName.affectnet:
            # neutral happy sad surprise fear disgust anger
            # weight_map = [2, 1, 3, 5, 7, 10, 3]
            weight_map = [5, 1, 3, 8, 7, 10, 9]
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
        loss = -10.0 * tf.reduce_mean(y_gt_oh * tf.math.log(y_pred) * weight_map)

        accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pr, y_gt_oh))

        '''focal lost'''
        # y_gt = tf.one_hot(y_gt, depth=num_classes)
        # loss_object = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        # loss_cross_entropy = loss_object(y_gt, y_pr)

        '''CategoricalCrossentropy'''
        # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # loss_cross_entropy = loss_object(y_gt, y_pr)
        '''sparse CategoricalCrossentropy'''
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss_cross_entropy = loss_object(y_gt, y_pr)

        return 5*loss, accuracy
        # return 10 * loss_cross_entropy
