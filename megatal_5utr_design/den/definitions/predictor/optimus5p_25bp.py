import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D
from keras.layers import GRU, BatchNormalization, LocallyConnected2D, Permute
from keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
import keras.losses

import tensorflow as tf

import isolearn.keras as iso

import numpy as np


#APARENT Saved Model definition

def load_saved_predictor(model_path) :

    # Model parameters
    inp_len = 25
    conv_layers = 2
    conv_kernel_size = 5
    conv_filters_first = 128
    conv_dropout=0.45
    dense_units=18
    dense_dropout=0.0
    reg_lambda = 0.0

    saved_model = tf.keras.models.load_model(model_path)

    def _initialize_predictor_weights(predictor_model, saved_model=saved_model):
        # Initialize predictor weights and mark them as non-trainable

        for conv_layer_idx in range(conv_layers):
            conv_layer_0_name = 'conv_{}_0'.format(conv_layer_idx + 1)
            conv_layer_1_name = 'conv_{}_1'.format(conv_layer_idx + 1)

            predictor_model.get_layer('vgg16_25bp_rand_' + conv_layer_0_name).set_weights(
                saved_model.get_layer(conv_layer_0_name).get_weights())
            predictor_model.get_layer('vgg16_25bp_rand_' + conv_layer_1_name).set_weights(
                saved_model.get_layer(conv_layer_1_name).get_weights())

            predictor_model.get_layer('vgg16_25bp_rand_' + conv_layer_0_name).trainable = False
            predictor_model.get_layer('vgg16_25bp_rand_' + conv_layer_1_name).trainable = False

        predictor_model.get_layer('vgg16_25bp_rand_dense_1').set_weights(
            saved_model.get_layer('dense_1').get_weights())
        predictor_model.get_layer('vgg16_25bp_rand_dense_1').trainable = False

        predictor_model.get_layer('vgg16_25bp_rand_dense_2').set_weights(
            saved_model.get_layer('dense_2').get_weights())
        predictor_model.get_layer('vgg16_25bp_rand_dense_2').trainable = False

    def _load_predictor_func(sequence_input, sequence_class) :

        predictor_output = Reshape((inp_len, 4))(sequence_input)

        for conv_layer_idx in range(conv_layers):
            predictor_output = Conv1D(
                activation="relu",
                padding='same',
                filters=conv_filters_first*(2**conv_layer_idx),
                kernel_size=conv_kernel_size,
                name='vgg16_25bp_rand_conv_{}_0'.format(conv_layer_idx + 1),
            )(predictor_output)
            predictor_output = Conv1D(
                activation="relu",
                padding='same',
                filters=conv_filters_first*(2**conv_layer_idx),
                kernel_size=conv_kernel_size,
                name='vgg16_25bp_rand_conv_{}_1'.format(conv_layer_idx + 1),
            )(predictor_output)
            predictor_output = MaxPooling1D(
                pool_size=2,
                strides=2,
                padding='same',
            )(predictor_output)
            predictor_output = Dropout(conv_dropout)(predictor_output)

        predictor_output = Flatten()(predictor_output)

        predictor_output = Dense(
            dense_units,
            name='vgg16_25bp_rand_dense_1',
            activation='relu',
        )(predictor_output)
        predictor_output = Dropout(dense_dropout)(predictor_output)

        predictor_output = Dense(
            1,
            name='vgg16_25bp_rand_dense_2',
            activation='linear',
        )(predictor_output)

        predictor_inputs = []
        predictor_outputs = [predictor_output]

        return predictor_inputs, predictor_outputs, _initialize_predictor_weights

    return _load_predictor_func
