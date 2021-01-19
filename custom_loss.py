import tensorflow as tf

class CustomLosses:
    # def __init__(self):
    #     pass

    def regressor_loss(self, y_gt, y_pr):
        loss = tf.reduce_mean(tf.square(y_gt - y_pr))
        return loss

    def cross_entropy_loss(self, y_gt, y_pr):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = loss_object(y_gt, y_pr)
        return loss
