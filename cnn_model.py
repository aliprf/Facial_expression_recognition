from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig
# from hg_Class import HourglassNet

import tensorflow as tf
# from tensorflow import keras
# from skimage.transform import resize
from keras.models import Model

from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet, resnet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, \
    Dropout, ReLU, Concatenate, Input, GlobalMaxPool2D, LeakyReLU, Softmax, ELU
import efficientnet.tfkeras as efn


class CNNModel:
    def get_model(self, arch, num_of_classes):
        if arch == 'mobileNetV2':
            model = self._create_MobileNet_with_embedding(num_of_classes,
                                                          input_shape=[InputDataSize.image_input_size,
                                                                       InputDataSize.image_input_size, 3])

        elif arch == 'efn-b3':
            model = self._create_efnb3_with_embedding(num_of_classes,
                                                      input_shape=[InputDataSize.image_input_size,
                                                                   InputDataSize.image_input_size, 3])
        elif arch == 'efn-b0':
            model = self._create_efnb0_with_embedding(num_of_classes,
                                                      input_shape=[InputDataSize.image_input_size,
                                                                   InputDataSize.image_input_size, 3])

        elif arch == 'mobileNetV2_single':
            model = self._create_MobileNet_with_embedding_single(num_of_classes,
                                                                 input_shape=[InputDataSize.image_input_size,
                                                                              InputDataSize.image_input_size, 3])
        elif arch == 'efficientNet':
            model = self._create_efficientNet()
        return model

    def _create_MobileNet_with_embedding_single(self, num_of_classes, input_shape):
        mobilenet_model_face = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_face.layers.pop()
        global_average_pooling2d = mobilenet_model_face.get_layer('global_average_pooling2d').output  # 1280
        x_l_face = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            global_average_pooling2d)  # No activation on final dense layer
        embedding_layer_face = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_face)  # L2 normalize embeddings
        '''FC layer for out'''
        x_l = Dense(LearningConfig.embedding_size * 2)(global_average_pooling2d)
        x_l = BatchNormalization()(x_l)
        x_l = ReLU()(x_l)

        x_l = Dense(LearningConfig.embedding_size)(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = ReLU()(x_l)

        x_l = Dense(LearningConfig.embedding_size // 2)(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = ReLU()(x_l)
        '''Dropout'''
        x_l = Dropout(rate=0.2)(x_l)
        '''out'''
        out_categorical = Dense(num_of_classes,
                                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                activation='softmax',
                                # activation='linear',
                                name='out')(x_l)

        inp = [mobilenet_model_face.input]
        revised_model = Model(inp, [out_categorical,
                                    embedding_layer_face])
        revised_model.summary()
        '''save json'''
        model_json = revised_model.to_json()

        with open("./model_archs/mn_v2_cat_emb_sinle.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def _create_efnb0_with_embedding(self, num_of_classes, input_shape):
        model_face = efn.EfficientNetB0(
            input_shape=input_shape,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        model_face.layers.pop()

        '''eyes'''
        model_eyes = efn.EfficientNetB0(
            input_shape=input_shape,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        model_eyes.layers.pop()

        '''nose'''
        model_nose = efn.EfficientNetB0(
            input_shape=input_shape,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        model_nose.layers.pop()

        '''mouth'''
        model_mouth = efn.EfficientNetB0(
            input_shape=input_shape,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        model_mouth.layers.pop()

        for layer in model_face.layers:
            layer._name = 'face_' + layer.name
        for layer in model_eyes.layers:
            layer._name = 'eyes_' + layer.name
        for layer in model_nose.layers:
            layer._name = 'nose_' + layer.name
        for layer in model_mouth.layers:
            layer._name = 'mouth_' + layer.name

        # mobilenet_model_mouth.summary()
        ''''''
        # o_relu_l_face = model_face.get_layer('face_top_activation').output  # 1280
        # o_relu_l_eyes = model_eyes.get_layer('eyes_top_activation').output  # 1280
        # o_relu_l_nose = model_nose.get_layer('nose_top_activation').output  # 1280
        # o_relu_l_mouth = model_mouth.get_layer('mouth_top_activation').output  # 1280

        g_x_l_face = model_face.get_layer('face_avg_pool').output  # 1280
        g_x_l_eyes = model_eyes.get_layer('eyes_avg_pool').output  # 1280
        g_x_l_nose = model_nose.get_layer('nose_avg_pool').output  # 1280
        g_x_l_mouth = model_mouth.get_layer('mouth_avg_pool').output  # 1280

        '''embedding'''
        # g_x_l_face = GlobalAveragePooling2D()(o_relu_l_face)
        # x_l_face = Dense(LearningConfig.embedding_size * 2,
        #                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_face)
        # x_l_face = BatchNormalization()(x_l_face)
        # x_l_face = Dropout(rate=0.2)(x_l_face)
        # x_l_face = ReLU()(x_l_face)

        x_l_face = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            g_x_l_face)  # No activation on final dense layer
        embedding_layer_face = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_face)  # L2 normalize embeddings

        '''eyes'''
        # g_x_l_eyes = GlobalAveragePooling2D()(o_relu_l_eyes)
        # x_l_eyes = Dense(LearningConfig.embedding_size * 2,
        #                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_eyes)
        # x_l_eyes = BatchNormalization()(x_l_eyes)
        # x_l_eyes = Dropout(rate=0.2)(x_l_eyes)
        # x_l_eyes = ReLU()(x_l_eyes)
        # x_l_eyes = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
        #     x_l_eyes)  # No activation on final dense layer
        embedding_layer_eyes = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            g_x_l_eyes)  # L2 normalize embeddings

        '''nose'''
        # g_x_l_nose = GlobalAveragePooling2D()(o_relu_l_nose)
        # x_l_nose = Dense(LearningConfig.embedding_size * 2,
        #                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_nose)
        # x_l_nose = BatchNormalization()(x_l_nose)
        # x_l_nose = Dropout(rate=0.2)(x_l_nose)
        # x_l_nose = ReLU()(x_l_nose)
        x_l_nose = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            g_x_l_nose)  # No activation on final dense layer
        embedding_layer_nose = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_nose)  # L2 normalize embeddings
        '''mouth'''
        # g_x_l_mouth = GlobalAveragePooling2D()(o_relu_l_mouth)
        # x_l_mouth = Dense(LearningConfig.embedding_size * 2,
        #                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_mouth)
        # x_l_mouth = BatchNormalization()(x_l_mouth)
        # x_l_mouth = Dropout(rate=0.3)(x_l_mouth)
        # x_l_mouth = ReLU()(x_l_mouth)
        # x_l_mouth = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
        #     x_l_mouth)  # No activation on final dense layer
        embedding_layer_mouth = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            g_x_l_mouth)  # L2 normalize embeddings

        '''concat'''
        fused_global_avg_pool = tf.keras.layers.Concatenate(axis=1)([g_x_l_face,
                                                                     g_x_l_eyes,
                                                                     g_x_l_nose,
                                                                     g_x_l_mouth])

        # fused_global_avg_pool = tf.keras.layers.Add()([g_x_l_face,
        #                                                g_x_l_eyes,
        #                                                g_x_l_nose,
        #                                                g_x_l_mouth])

        # concat_embeddings = tf.keras.layers.Concatenate(axis=1)([o_relu_l_face,
        #                                                          o_relu_l_eyes,
        #                                                          o_relu_l_nose,
        #                                                          o_relu_l_mouth])

        # fused_global_avg_pool = GlobalAveragePooling2D()(concat_embeddings)

        '''FC layer for out'''
        # x_l = Dense(LearningConfig.embedding_size)(fused_global_avg_pool)
        # x_l = BatchNormalization()(x_l)
        # x_l = Dropout(rate=0.2)(x_l)
        # x_l = ReLU()(x_l)
        #
        # x_l = Dense(LearningConfig.embedding_size // 2)(x_l)
        # x_l = BatchNormalization()(x_l)
        # x_l = Dropout(rate=0.2)(x_l)
        # x_l = ReLU()(x_l)
        #
        # x_l = Dense(LearningConfig.embedding_size // 4)(x_l)
        # x_l = BatchNormalization()(x_l)
        # x_l = Dropout(rate=0.2)(x_l)
        # x_l = ReLU()(x_l)
        #
        # x_l = Dense(LearningConfig.embedding_size // 8)(x_l)
        # x_l = BatchNormalization()(x_l)
        # x_l = Dropout(rate=0.2)(x_l)
        # x_l = ReLU()(x_l)
        #
        # x_l = Dense(LearningConfig.embedding_size // 16)(x_l)
        # x_l = BatchNormalization()(x_l)
        # x_l = Dropout(rate=0.2)(x_l)
        # x_l = ReLU()(x_l)

        '''out'''
        x_l = Dropout(rate=0.2)(fused_global_avg_pool)
        out_categorical = Dense(num_of_classes,
                                activation='softmax',
                                name='out')(x_l)

        inp = [model_face.input, model_eyes.input,
               model_nose.input, model_mouth.input]
        revised_model = Model(inp, [out_categorical,
                                    embedding_layer_face,
                                    embedding_layer_eyes,
                                    embedding_layer_nose,
                                    embedding_layer_mouth])
        revised_model.summary()
        '''save json'''
        model_json = revised_model.to_json()

        with open("./model_archs/efnB0_cat_emb.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def _create_MobileNet_with_embedding(self, num_of_classes, input_shape):
        mobilenet_model_face = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_face.layers.pop()

        '''eyes'''
        mobilenet_model_eyes = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_eyes.layers.pop()

        '''nose'''
        mobilenet_model_nose = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_nose.layers.pop()

        '''mouth'''
        mobilenet_model_mouth = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_mouth.layers.pop()

        for layer in mobilenet_model_face.layers:
            layer._name = 'face_' + layer.name
        for layer in mobilenet_model_eyes.layers:
            layer._name = 'eyes_' + layer.name
        for layer in mobilenet_model_nose.layers:
            layer._name = 'nose_' + layer.name
        for layer in mobilenet_model_mouth.layers:
            layer._name = 'mouth_' + layer.name

        # mobilenet_model_mouth.summary()
        ''''''
        face_out_relu = mobilenet_model_face.get_layer('face_out_relu').output  # 1280
        face_global_avg_pool = GlobalAveragePooling2D()(face_out_relu)
        embedding_layer_face = tf.keras.layers.Dense(LearningConfig.embedding_size,
                                                     activation='relu')(face_global_avg_pool)

        eyes_out_relu = mobilenet_model_eyes.get_layer('eyes_out_relu').output  # 1280
        eyes_global_avg_pool = GlobalAveragePooling2D()(eyes_out_relu)
        embedding_layer_eyes = tf.keras.layers.Dense(LearningConfig.embedding_size,
                                                     activation='relu')(eyes_global_avg_pool)

        nose_out_relu = mobilenet_model_nose.get_layer('nose_out_relu').output  # 1280
        nose_global_avg_pool = GlobalAveragePooling2D()(nose_out_relu)
        embedding_layer_nose = tf.keras.layers.Dense(LearningConfig.embedding_size,
                                                     activation='relu')(nose_global_avg_pool)

        mouth_out_relu = mobilenet_model_mouth.get_layer('mouth_out_relu').output  # 1280
        mouth_global_avg_pool = GlobalAveragePooling2D()(mouth_out_relu)
        embedding_layer_mouth = tf.keras.layers.Dense(LearningConfig.embedding_size,
                                                      activation='relu')(mouth_global_avg_pool)

        '''concat'''
        concat_globs = tf.keras.layers.Concatenate(axis=1)([face_out_relu,
                                                            eyes_out_relu,
                                                            nose_out_relu,
                                                            mouth_out_relu])
        global_avg_pool = GlobalAveragePooling2D()(concat_globs)

        '''out'''
        x = Dropout(0.4)(global_avg_pool)

        out_categorical = Dense(num_of_classes,
                                activation='softmax',
                                name='out')(x)

        inp = [mobilenet_model_face.input, mobilenet_model_eyes.input,
               mobilenet_model_nose.input, mobilenet_model_mouth.input]
        revised_model = Model(inp, [out_categorical,
                                    embedding_layer_face,
                                    embedding_layer_eyes,
                                    embedding_layer_nose,
                                    embedding_layer_mouth])
        revised_model.summary()
        '''save json'''
        model_json = revised_model.to_json()

        with open("./model_archs/mn_v2_cat_emb.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def pre_create_MobileNet_with_embedding(self, num_of_classes, input_shape):
        mobilenet_model_face = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_face.layers.pop()

        '''eyes'''
        mobilenet_model_eyes = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_eyes.layers.pop()

        '''nose'''
        mobilenet_model_nose = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_nose.layers.pop()

        '''mouth'''
        mobilenet_model_mouth = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_mouth.layers.pop()

        for layer in mobilenet_model_face.layers:
            layer._name = 'face_' + layer.name
        for layer in mobilenet_model_eyes.layers:
            layer._name = 'eyes_' + layer.name
        for layer in mobilenet_model_nose.layers:
            layer._name = 'nose_' + layer.name
        for layer in mobilenet_model_mouth.layers:
            layer._name = 'mouth_' + layer.name

        # mobilenet_model_mouth.summary()
        ''''''
        o_relu_l_face = mobilenet_model_face.get_layer('face_out_relu').output  # 1280
        o_relu_l_eyes = mobilenet_model_eyes.get_layer('eyes_out_relu').output  # 1280
        o_relu_l_nose = mobilenet_model_nose.get_layer('nose_out_relu').output  # 1280
        o_relu_l_mouth = mobilenet_model_mouth.get_layer('mouth_out_relu').output  # 1280

        '''embedding'''
        g_x_l_face = GlobalAveragePooling2D()(o_relu_l_face)
        x_l_face = Dense(LearningConfig.embedding_size * 2,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_face)
        x_l_face = BatchNormalization()(x_l_face)
        x_l_face = Dropout(rate=0.3)(x_l_face)
        x_l_face = ELU()(x_l_face)

        x_l_face = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_face)  # No activation on final dense layer
        embedding_layer_face = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_face)  # L2 normalize embeddings
        # eyes
        g_x_l_eyes = GlobalAveragePooling2D()(o_relu_l_eyes)
        x_l_eyes = Dense(LearningConfig.embedding_size * 2,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_eyes)
        x_l_eyes = BatchNormalization()(x_l_eyes)
        x_l_eyes = Dropout(rate=0.3)(x_l_eyes)
        x_l_eyes = ELU()(x_l_eyes)
        x_l_eyes = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_eyes)  # No activation on final dense layer
        embedding_layer_eyes = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_eyes)  # L2 normalize embeddings
        # nose
        g_x_l_nose = GlobalAveragePooling2D()(o_relu_l_nose)
        x_l_nose = Dense(LearningConfig.embedding_size * 2,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_nose)
        x_l_nose = BatchNormalization()(x_l_nose)
        x_l_nose = Dropout(rate=0.3)(x_l_nose)
        x_l_nose = ELU()(x_l_nose)
        x_l_nose = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_nose)  # No activation on final dense layer
        embedding_layer_nose = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_nose)  # L2 normalize embeddings
        # mouth
        g_x_l_mouth = GlobalAveragePooling2D()(o_relu_l_mouth)
        x_l_mouth = Dense(LearningConfig.embedding_size * 2,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_mouth)
        x_l_mouth = BatchNormalization()(x_l_mouth)
        x_l_mouth = Dropout(rate=0.3)(x_l_mouth)
        x_l_mouth = ELU()(x_l_mouth)
        x_l_mouth = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_mouth)  # No activation on final dense layer
        embedding_layer_mouth = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_mouth)  # L2 normalize embeddings

        '''concat'''
        # concat_embeddings = tf.keras.layers.Concatenate(axis=1)([embedding_layer_face,
        #                                                          embedding_layer_eyes,
        #                                                          embedding_layer_nose,
        #                                                          embedding_layer_mouth])

        concat_embeddings = tf.keras.layers.Concatenate(axis=1)([o_relu_l_face,
                                                                 o_relu_l_eyes,
                                                                 o_relu_l_nose,
                                                                 o_relu_l_mouth])

        fused_global_avg_pool = GlobalAveragePooling2D()(concat_embeddings)
        '''FC layer for out'''
        x_l = Dense(LearningConfig.embedding_size,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(fused_global_avg_pool)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ELU()(x_l)

        x_l = Dense(LearningConfig.embedding_size // 2,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ELU()(x_l)

        x_l = Dense(LearningConfig.embedding_size // 4,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ELU()(x_l)

        x_l = Dense(LearningConfig.embedding_size // 8,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ELU()(x_l)

        x_l = Dense(LearningConfig.embedding_size // 16,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ELU()(x_l)

        '''out'''
        out_categorical = Dense(num_of_classes,
                                activation='softmax',
                                name='out')(x_l)

        inp = [mobilenet_model_face.input, mobilenet_model_eyes.input,
               mobilenet_model_nose.input, mobilenet_model_mouth.input]
        revised_model = Model(inp, [out_categorical,
                                    embedding_layer_face,
                                    embedding_layer_eyes,
                                    embedding_layer_nose,
                                    embedding_layer_mouth])
        revised_model.summary()
        '''save json'''
        model_json = revised_model.to_json()

        with open("./model_archs/mn_v2_cat_emb.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def _1_create_MobileNet_with_embedding(self, num_of_classes, input_shape):
        # mnv3 = mobilenet_v3.

        mobilenet_model_face = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_face.layers.pop()

        '''eyes'''
        mobilenet_model_eyes = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_eyes.layers.pop()

        '''nose'''
        mobilenet_model_nose = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_nose.layers.pop()

        '''mouth'''
        mobilenet_model_mouth = mobilenet_v2.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model_mouth.layers.pop()

        for layer in mobilenet_model_face.layers:
            layer._name = 'face_' + layer.name
        for layer in mobilenet_model_eyes.layers:
            layer._name = 'eyes_' + layer.name
        for layer in mobilenet_model_nose.layers:
            layer._name = 'nose_' + layer.name
        for layer in mobilenet_model_mouth.layers:
            layer._name = 'mouth_' + layer.name

        # mobilenet_model_mouth.summary()
        ''''''
        g_x_l_face = mobilenet_model_face.get_layer('face_global_average_pooling2d').output  # 1280
        g_x_l_eyes = mobilenet_model_eyes.get_layer('eyes_global_average_pooling2d_1').output  # 1280
        g_x_l_nose = mobilenet_model_nose.get_layer('nose_global_average_pooling2d_2').output  # 1280
        g_x_l_mouth = mobilenet_model_mouth.get_layer('mouth_global_average_pooling2d_3').output  # 1280

        '''embedding'''
        x_l_face = Dense(LearningConfig.embedding_size * 2,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_face)
        x_l_face = BatchNormalization()(x_l_face)
        x_l_face = Dropout(rate=0.3)(x_l_face)
        x_l_face = ReLU()(x_l_face)

        x_l_face = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_face)  # No activation on final dense layer
        embedding_layer_face = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_face)  # L2 normalize embeddings
        # eyes
        x_l_eyes = Dense(LearningConfig.embedding_size * 2,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_eyes)
        x_l_eyes = BatchNormalization()(x_l_eyes)
        x_l_eyes = Dropout(rate=0.3)(x_l_eyes)
        x_l_eyes = ReLU()(x_l_eyes)
        x_l_eyes = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_eyes)  # No activation on final dense layer
        embedding_layer_eyes = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_eyes)  # L2 normalize embeddings
        # nose
        x_l_nose = Dense(LearningConfig.embedding_size * 2,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_nose)
        x_l_nose = BatchNormalization()(x_l_nose)
        x_l_nose = Dropout(rate=0.3)(x_l_nose)
        x_l_nose = ReLU()(x_l_nose)
        x_l_nose = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_nose)  # No activation on final dense layer
        embedding_layer_nose = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_nose)  # L2 normalize embeddings
        # mouth
        x_l_mouth = Dense(LearningConfig.embedding_size * 2,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))(g_x_l_mouth)
        x_l_mouth = BatchNormalization()(x_l_mouth)
        x_l_mouth = Dropout(rate=0.3)(x_l_mouth)
        x_l_mouth = ReLU()(x_l_mouth)
        x_l_mouth = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_mouth)  # No activation on final dense layer
        embedding_layer_mouth = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_mouth)  # L2 normalize embeddings

        '''concat'''
        # concat_embeddings = tf.keras.layers.Concatenate(axis=1)([embedding_layer_face,
        #                                                          embedding_layer_eyes,
        #                                                          embedding_layer_nose,
        #                                                          embedding_layer_mouth])

        concat_embeddings = tf.keras.layers.Concatenate(axis=1)([g_x_l_face,
                                                                 g_x_l_eyes,
                                                                 g_x_l_nose,
                                                                 g_x_l_mouth])

        '''FC layer for out'''
        x_l = Dense(LearningConfig.embedding_size * 2,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001))(concat_embeddings)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ReLU()(x_l)

        x_l = Dense(LearningConfig.embedding_size,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ReLU()(x_l)

        x_l = Dense(LearningConfig.embedding_size // 2,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ReLU()(x_l)

        x_l = Dense(LearningConfig.embedding_size // 4,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ReLU()(x_l)

        x_l = Dense(LearningConfig.embedding_size // 8,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = Dropout(rate=0.3)(x_l)
        x_l = ReLU()(x_l)

        '''out'''
        out_categorical = Dense(num_of_classes,
                                # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                activation='softmax',
                                # activation='linear',
                                name='out')(x_l)

        inp = [mobilenet_model_face.input, mobilenet_model_eyes.input,
               mobilenet_model_nose.input, mobilenet_model_mouth.input]
        revised_model = Model(inp, [out_categorical,
                                    embedding_layer_face,
                                    embedding_layer_eyes,
                                    embedding_layer_nose,
                                    embedding_layer_mouth])
        revised_model.summary()
        '''save json'''
        model_json = revised_model.to_json()

        with open("./model_archs/mn_v2_cat_emb.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def _create_efficientNet(self):
        initializer = tf.keras.initializers.glorot_uniform()
        eff_net = efn.EfficientNetB3(include_top=True,
                                     weights=None,
                                     input_tensor=None,
                                     input_shape=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],
                                     pooling=None,
                                     classes=LearningConfig.expression_output_len_shrunk)  # or weights='noisy-student'
        '''save json'''
        # model_json = eff_net.to_json()
        #
        # with open("./model_archs/eff_net_b0.json", "w") as json_file:
        #     json_file.write(model_json)
        '''revise model'''
        eff_net.layers.pop()
        x = eff_net.get_layer('top_dropout').output
        # o_velocity = Dense(LearningConfig.velocity_output_len, name='O_Velocity')(x)
        o_expression = Dense(LearningConfig.expression_output_len_shrunk, activation='linear', name='O_Expression_sh')(
            x)
        # o_velocity = Dense(1, name='O_Velocity_reg')(x)

        # x = GlobalAveragePooling2D()(x)
        # x = Dropout(0.2)(x)
        # x = Dense(512, use_bias=False, kernel_initializer=initializer)(x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        # x = Dropout(0.2)(x)
        # x = Dense(256, use_bias=False, kernel_initializer=initializer)(x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        # '''create outputs'''
        # o_velocity = Dense(1, name='O_Velocity_reg')(x)
        # o_velocity = Dense(LearningConfig.velocity_output_len, name='O_Velocity_categorical')(x)
        # o_category = Dense(output_len, name='O_Category')(x)

        inp = eff_net.input
        revised_model = Model(inp, [o_expression])
        # revised_model = Model(inp, [o_velocity, o_expression])
        # revised_model = Model(inp, [o_velocity, o_category])
        revised_model.summary()
        '''save json'''
        model_json = revised_model.to_json()

        with open("./model_archs/ef_b3.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model
