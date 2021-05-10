import tensorflow as tf

class TFAugmentation:

    def flip(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x)
        return x

    def color(self, x: tf.Tensor) -> tf.Tensor:
        # print(x.shape)
        p = 0.5
        if tf.random.uniform([]) < p:
            x = tf.image.random_hue(x, 0.3)
        if tf.random.uniform([]) < p:
            x = tf.image.random_saturation(x, 0.1, 1.2,)
        if tf.random.uniform([]) < p:
           x = tf.image.random_brightness(x, 0.4,)
        if tf.random.uniform([]) < p:
            x = tf.image.random_contrast(x, 0.1, 1.3)
        return x

    def random_invert_img(self, x: tf.Tensor, p=0.3) -> tf.Tensor:

        if tf.random.uniform([]) < p:
            x = (1 - x)
        else:
            x
        return x