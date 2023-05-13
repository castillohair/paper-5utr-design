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

    saved_model = tf.keras.models.load_model(model_path)

    def _initialize_predictor_weights(predictor_model, saved_model=saved_model):
        # Initialize predictor weights and mark them as non-trainable

        predictor_model.get_layer('optimus5p_50bp_conv_1').set_weights(
            saved_model.get_layer('conv_1').get_weights())
        predictor_model.get_layer('optimus5p_50bp_conv_1').trainable = False

        predictor_model.get_layer('optimus5p_50bp_conv_2').set_weights(
            saved_model.get_layer('conv_2').get_weights())
        predictor_model.get_layer('optimus5p_50bp_conv_2').trainable = False

        predictor_model.get_layer('optimus5p_50bp_conv_3').set_weights(
            saved_model.get_layer('conv_3').get_weights())
        predictor_model.get_layer('optimus5p_50bp_conv_3').trainable = False

        predictor_model.get_layer('optimus5p_50bp_dense_1').set_weights(
            saved_model.get_layer('dense_1').get_weights())
        predictor_model.get_layer('optimus5p_50bp_dense_1').trainable = False

        predictor_model.get_layer('optimus5p_50bp_dense_2').set_weights(
            saved_model.get_layer('dense_2').get_weights())
        predictor_model.get_layer('optimus5p_50bp_dense_2').trainable = False

    def _load_predictor_func(sequence_input, sequence_class) :
        # Optimus 5' parameters
        nb_epoch=3
        border_mode='same'
        inp_len=50
        nodes=40
        nbr_filters=120
        filter_len=8
        dropout1=0
        dropout2=0
        dropout3=0.2

        sequence_input = Reshape((inp_len, 4))(sequence_input)
        
        # Shared model definition
        predictor_output = Conv1D(
            activation="relu",
            input_shape=(inp_len, 4),
            padding=border_mode,
            filters=nbr_filters,
            kernel_size=filter_len,
            name='optimus5p_50bp_conv_1')(sequence_input)
        predictor_output = Conv1D(
            activation="relu",
            input_shape=(inp_len, 1),
            padding=border_mode,
            filters=nbr_filters,
            kernel_size=filter_len,
            name='optimus5p_50bp_conv_2')(predictor_output)
        predictor_output = Dropout(dropout1)(predictor_output)
        predictor_output = Conv1D(
            activation="relu",
            input_shape=(inp_len, 1),
            padding=border_mode,
            filters=nbr_filters,
            kernel_size=filter_len,
            name='optimus5p_50bp_conv_3')(predictor_output)
        predictor_output = Dropout(dropout2)(predictor_output)
        predictor_output = Flatten()(predictor_output)

        predictor_output = Dense(nodes, name='optimus5p_50bp_dense_1')(predictor_output)
        predictor_output = Activation('relu')(predictor_output)
        predictor_output = Dropout(dropout3)(predictor_output)
        
        predictor_output = Dense(1, name='optimus5p_50bp_dense_2')(predictor_output)
        predictor_output = Activation('linear')(predictor_output)

        predictor_inputs = []
        predictor_outputs = [predictor_output]

        return predictor_inputs, predictor_outputs, _initialize_predictor_weights

    return _load_predictor_func
