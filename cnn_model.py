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
    def get_model(self, arch):
        if arch == 'mobileNetV2':
            model = self._create_MobileNet()
        elif arch == 'efficientNet':
            model = self._create_efficientNet()
        return model

    def _create_MobileNet(self):
        initializer = tf.keras.initializers.glorot_uniform()

        mobilenet_model = mobilenet_v2.MobileNetV2(
            input_shape=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None)
        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d').output  # 1280
        x = Dropout(0.2)(x)
        x = Dense(512, use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        '''create outputs'''
        o_velocity = Dense(LearningConfig.velocity_output_len, name='O_Velocity')(x)
        # o_category = Dense(output_len, name='O_Category')(x)

        inp = mobilenet_model.input
        revised_model = Model(inp, [o_velocity])
        # revised_model = Model(inp, [o_velocity, o_category])
        revised_model.summary()
        '''save json'''
        model_json = revised_model.to_json()

        with open("./model_archs/mn_v2.json", "w") as json_file:
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
        o_expression = Dense(LearningConfig.expression_output_len_shrunk, name='O_Expression_sh')(x)
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

        with open("./model_archs/ef_b0.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model
