import tensorflow as tf

class TFAugmentation:

    def flip(self, x: tf.Tensor) -> tf.Tensor:
        p = 0.5
        if tf.random.uniform([]) < p:
            x = tf.image.random_flip_left_right(x)
        return x

    def color(self, x: tf.Tensor) -> tf.Tensor:
        # print(x.shape)
        p = 0.5
        if tf.random.uniform([]) < p:
            if 0 <= tf.random.uniform([]) < 0.25:
                x = tf.image.random_hue(x, 0.2)
            elif 0.25 <= tf.random.uniform([]) <= 0.5:
                x = tf.image.random_saturation(x, 0.1, 1.2,)
            elif 0.5 <= tf.random.uniform([]) <= 0.75:
                x = tf.image.random_brightness(x, 0.5,)
            else:
                x = tf.image.random_contrast(x, 0.1, 0.9)
        return x

    def random_invert_img(self, x: tf.Tensor, p=0.2) -> tf.Tensor:
        if tf.random.uniform([]) < p:
            x = 0.8*tf.abs(0.5 - x) + 0.2 * x
        return x

    # def random_rotate(self, x: tf.Tensor, p=0.5) -> tf.Tensor:
    #     if tf.random.uniform([]) < p:
    #         x = tf.keras.preprocessing.image.random_rotation(x, rg=25,)
    #     return x

    def random_zoom(self, x: tf.Tensor, p=0.7) -> tf.Tensor:
        if tf.random.uniform([]) < p:
            x = tf.keras.preprocessing.image.random_zoom(x, zoom_range=[0.8, 1.1])
        return x

    def random_quality(self, x: tf.Tensor, p=0.5) -> tf.Tensor:
        if tf.random.uniform([]) < p:
            x = tf.image.random_jpeg_quality(x, 50, 100)
        return x

