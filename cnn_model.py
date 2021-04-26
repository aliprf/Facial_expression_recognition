from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig
# from hg_Class import HourglassNet

import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from keras.models import Model

from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, \
    Dropout, ReLU, Concatenate, Input, GlobalMaxPool2D, LeakyReLU, Softmax
import efficientnet.tfkeras as efn


class CNNModel:
    def get_model(self, arch, num_of_classes):
        if arch == 'mobileNetV2':
            model = self._create_MobileNet_with_embedding(num_of_classes,
                                                          input_shape=[InputDataSize.image_input_size,
                                                                       InputDataSize.image_input_size, 5])
        elif arch == 'mobileNetV2_single':
            model = self._create_MobileNet_with_embedding_single(num_of_classes,
                                                          input_shape=[InputDataSize.image_input_size,
                                                                       InputDataSize.image_input_size, 5])
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
        x_l_face = mobilenet_model_face.get_layer('global_average_pooling2d').output  # 1280
        x_l_face = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_face)  # No activation on final dense layer
        embedding_layer_face = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_face)  # L2 normalize embeddings
        '''FC layer for out'''
        # x_l = Dense(LearningConfig.embedding_size * 2)(x_l_face)
        # x_l = BatchNormalization()(x_l)
        # x_l = ReLU()(x_l)
        #
        # x_l = Dense(LearningConfig.embedding_size)(x_l)
        # x_l = BatchNormalization()(x_l)
        # x_l = ReLU()(x_l)
        #
        # x_l = Dense(LearningConfig.embedding_size // 2)(x_l)
        # x_l = BatchNormalization()(x_l)
        # x_l = ReLU()(x_l)
        '''Dropout'''
        x_l = Dropout(rate=0.2)(x_l_face)
        # x_l = Dropout(rate=0.2)(x_l)
        '''out'''
        out_categorical = Dense(num_of_classes,
                                # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                activation='softmax',
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

    def _create_MobileNet_with_embedding(self, num_of_classes, input_shape):
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
        x_l_face = mobilenet_model_face.get_layer('face_global_average_pooling2d').output  # 1280
        x_l_eyes = mobilenet_model_eyes.get_layer('eyes_global_average_pooling2d_1').output  # 1280
        x_l_nose = mobilenet_model_nose.get_layer('nose_global_average_pooling2d_2').output  # 1280
        x_l_mouth = mobilenet_model_mouth.get_layer('mouth_global_average_pooling2d_3').output  # 1280

        '''embedding'''
        x_l_face = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_face)  # No activation on final dense layer
        embedding_layer_face = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_face)  # L2 normalize embeddings

        x_l_eyes = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_eyes)  # No activation on final dense layer
        embedding_layer_eyes = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_eyes)  # L2 normalize embeddings

        x_l_nose = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_nose)  # No activation on final dense layer
        embedding_layer_nose = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_nose)  # L2 normalize embeddings

        x_l_mouth = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(
            x_l_mouth)  # No activation on final dense layer
        embedding_layer_mouth = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
            x_l_mouth)  # L2 normalize embeddings

        '''concat'''
        concat_embeddings = tf.keras.layers.Concatenate(axis=1)([embedding_layer_face,
                                                                 embedding_layer_eyes,
                                                                 embedding_layer_nose,
                                                                 embedding_layer_mouth])
        '''FC layer for out'''
        x_l = Dense(LearningConfig.embedding_size * 2)(concat_embeddings)
        x_l = BatchNormalization()(x_l)
        x_l = ReLU()(x_l)

        x_l = Dense(LearningConfig.embedding_size)(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = ReLU()(x_l)

        x_l = Dense(LearningConfig.embedding_size//2)(x_l)
        x_l = BatchNormalization()(x_l)
        x_l = ReLU()(x_l)
        '''Dropout'''
        x_l = Dropout(rate=0.5)(x_l)
        '''out'''
        out_categorical = Dense(num_of_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001),
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
        o_expression = Dense(LearningConfig.expression_output_len_shrunk,activation='linear', name='O_Expression_sh')(x)
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
