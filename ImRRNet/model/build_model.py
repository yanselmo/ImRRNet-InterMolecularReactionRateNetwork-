#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers
from tensorflow.keras.models import Model

from gensim.models import word2vec
from mol2vec import features

from .custom_layers import MultiHeadInterMolecularAttention

def ImRRNet_model(embedding_matrix,
                n_blocks = 1,
                num_heads = 6, 
                dropout_rate = 0.1,
                maxlen = 117,
                dim = 300,
                dff = 1024,
                kernel_regularizer = 'l2',
                kernel_initializer = 'randomnormal'):
    """
    build multihead interattetion model
    include embedding layer, multihead interatttention layer, final dense layer.
    *args
    embedding_matrix(str or numpy ndarray): if str, path where mol2vec embedding matrix is saved as numpy *.npy format
                                            if numpy ndarray, mol2vec embbedding matrix including zero padding
    n_blocks(int): the number of intermolecular attention blocks
    num_heads(int): the number of attention head in attention blocks
    dropout_rate(float): dropout rate
    maxlen(int): maximum length of sentence-like input
    dff(int or list of int): if int, only one final dense layer whose dimension is dim is used
                             if list of int, multiple final dense layers are used. dimension of each 
                             dense layer is equal to element in the list
    kernel_regularizer(str or tensorflow regularizer object)
    kernel_initializer(str or tensorflow initializer object)
    """
    # currently, embedding matrix is not included in the model
    if type(embedding_matrix) == str: embedding_matrix = np.load(embedding_matrix)
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(position, d_model):
        ''' 
        function to make inputs positional encoded
        position: equal meaning with timestep t in sequential data
        d_model: equal meaning with the number of calss 
        '''
        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)

      # use sin function to even index
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

      # use cos function to odd index
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    if kernel_initializer == 'randomnormal':
        randomnormal = initializers.RandomNormal()
    else:
        #TODO: for completness, change the name of variable "randomnormal" to "kernel_initializer"
        randomnormal = kernel_initializer
    if kernel_regularizer == 'l2':
        l2 = regularizers.l2(l = 1e-4)
    else:
        l2 = kernel_regularizer
    tf.keras.backend.clear_session()    
    ## building model ##
    n_in = layers.Input(maxlen)
    e_in = layers.Input(maxlen)

    index = len(embedding_matrix)-1
    def create_padding_mask(x):
        mask = tf.cast(tf.math.equal(x, index), tf.float32)
        # batch_size, 1, 1, maxlen
        return mask[:, tf.newaxis, tf.newaxis, :]
    n_mask = layers.Lambda(create_padding_mask, output_shape = (1,1,None))(n_in)
    e_mask = layers.Lambda(create_padding_mask, output_shape = (1,1,None))(e_in)

    embedding_n = layers.Embedding(input_dim = embedding_matrix.shape[0],
                                output_dim = embedding_matrix.shape[1],
                                weights = [embedding_matrix],
                                trainable = False)
    embedding_e = layers.Embedding(input_dim = embedding_matrix.shape[0],
                                output_dim = embedding_matrix.shape[1],
                                weights = [embedding_matrix],
                                trainable = False)
    n_out = embedding_n(n_in)
    e_out= embedding_e(e_in)
    
    def pos_enc(inputs):
        positional_arr = tf.squeeze(positional_encoding(maxlen,dim))
        inputs += positional_arr
        return inputs

    n_out = layers.Lambda(pos_enc)(n_out)
    e_out = layers.Lambda(pos_enc)(e_out)
    if n_blocks == 1:
        n_out_, e_out_ = MultiHeadInterMolecularAttention(dim, num_heads,
                                                         kernel_initializer = randomnormal,
                                                         kernel_regularizer = l2,
                                                         dropout_rate = dropout_rate)([n_out, e_out, n_mask, e_mask])
        n_out = layers.Add()([n_out, n_out_])
        e_out = layers.Add()([e_out, e_out_])

        n_out_ = layers.Dense(dim, activation = 'relu',
                             kernel_initializer = randomnormal,
                             kernel_regularizer = l2)(n_out)
        n_out_ = layers.Dropout(dropout_rate)(n_out_)
        e_out_ = layers.Dense(dim, activation = 'relu',
                             kernel_initializer = randomnormal,
                             kernel_regularizer = l2)(e_out)
        e_out_ = layers.Dropout(dropout_rate)(e_out_)
        n_out = layers.Add()([n_out, n_out_])
        e_out = layers.Add()([e_out, e_out_])
    elif n_blocks == 0:
        pass
    else:
        for i in range(n_blocks):
            n_out_, e_out_ = MultiHeadInterMolecularAttention(dim, num_heads,
                                                             kernel_initializer = randomnormal,
                                                             kernel_regularizer = l2,
                                                             dropout_rate = dropout_rate)([n_out, e_out, n_mask, e_mask])
            n_out = layers.Add()([n_out, n_out_])
            e_out = layers.Add()([e_out, e_out_])

            n_out_ = layers.Dense(dim, activation = 'relu',
                                 kernel_initializer = randomnormal,
                                 kernel_regularizer = l2)(n_out)
            n_out_ = layers.Dropout(dropout_rate)(n_out_)
            e_out_ = layers.Dense(dim, activation = 'relu',
                                 kernel_initializer = randomnormal,
                                 kernel_regularizer = l2)(e_out)
            e_out_ = layers.Dropout(dropout_rate)(e_out_)
            n_out = layers.Add()([n_out, n_out_])
            e_out = layers.Add()([e_out, e_out_])

    n_out = layers.GlobalMaxPool1D()(n_out)
    e_out = layers.GlobalMaxPool1D()(e_out)

    conc = layers.Concatenate()([n_out, e_out])
    if type(dff) == list:
        for i, d in enumerate(dff):
            if i == 0:
                out = layers.Dense(d, activation = 'relu',
                                   kernel_initializer = randomnormal,
                                   kernel_regularizer = l2)(conc)
                out = layers.Dropout(dropout_rate)(out)
            else:
                out = layers.Dense(d, activation = 'relu',
                                   kernel_initializer = randomnormal,
                                   kernel_regularizer = l2)(out)
                out = layers.Dropout(dropout_rate)(out)
    else:
        out = layers.Dense(dff, activation = 'relu', 
                           kernel_initializer = randomnormal,
                          kernel_regularizer = l2)(conc)
        out = layers.Dropout(dropout_rate)(out)
    out = layers.Dense(1, activation = None,
                    kernel_initializer = randomnormal,
                    kernel_regularizer = l2)(out)

    model = Model([n_in, e_in], out)    
    return model



def rnn_model(embedding_matrix,
                rnntype = 'lstm',
                dropout_rate = 0.1,
                maxlen = 117,
                dim = 300,
                dff = 1024,
                kernel_regularizer = 'l2',
                kernel_initializer = 'randomnormal'):
    if type(embedding_matrix) == str: embedding_matrix = np.load(embedding_matrix)
    if kernel_initializer == 'randomnormal':
        randomnormal = initializers.RandomNormal()
    else:
        #TODO: change "randomnormal" to "kernel_initializer"
        randomnormal = kernel_initializer
    if kernel_regularizer == 'l2':
        l2 = regularizers.l2(l = 1e-4)
    else:
        l2 = kernel_regularizer

    if rnntype == 'lstm':
        n_recurrent = layers.LSTM(dim,
                kernel_initializer = randomnormal,
                kernel_regularizer = l2,
                dropout = dropout_rate,
                return_sequences = True)
        e_recurrent = layers.LSTM(dim,
                kernel_initializer = randomnormal,
                kernel_regularizer = l2,
                dropout = dropout_rate,
                return_sequences = True)

    elif rnntype == 'gru':
        n_recurrent = layers.GRU(dim,
                kernel_initializer = randomnormal,
                kernel_regularizer = l2,
                dropout = dropout_rate,
                return_sequences = True)
        e_recurrent = layers.GRU(dim,
                kernel_initializer = randomnormal,
                kernel_regularizer = l2,
                dropout = dropout_rate,
                return_sequences = True)
    else:
        raise ValueError('rnntype should be lstm or gru')
    tf.keras.backend.clear_session()    
    ## building model ##
    n_in = layers.Input(maxlen)
    e_in = layers.Input(maxlen)

    index = len(embedding_matrix)-1
    def create_padding_mask(x):
        mask = tf.cast(tf.math.equal(x, index), tf.float32)
        # batch_size, 1, 1, maxlen
        return mask[:, tf.newaxis, tf.newaxis, :]
    n_mask = layers.Lambda(create_padding_mask, output_shape = (1,1,None))(n_in)
    e_mask = layers.Lambda(create_padding_mask, output_shape = (1,1,None))(e_in)

    embedding_n = layers.Embedding(input_dim = embedding_matrix.shape[0],
                                output_dim = embedding_matrix.shape[1],
                                weights = [embedding_matrix],
                                trainable = False)
    embedding_e = layers.Embedding(input_dim = embedding_matrix.shape[0],
                                output_dim = embedding_matrix.shape[1],
                                weights = [embedding_matrix],
                                trainable = False)
    n_out = embedding_n(n_in)
    e_out= embedding_e(e_in)
    
    n_out_ = n_recurrent(n_out)
    e_out_ = e_recurrent(e_out)

    n_out = layers.Add()([n_out, n_out_])
    e_out = layers.Add()([e_out, e_out_])

    out = layers.Concatenate()([n_out, e_out])
    conc = layers.GlobalMaxPool1D()(out)

    if type(dff) == list:
        for i, d in enumerate(dff):
            if i == 0:
                out = layers.Dense(d, activation = 'relu',
                                   kernel_initializer = randomnormal,
                                   kernel_regularizer = l2)(conc)
                out = layers.Dropout(dropout_rate)(out)
            else:
                out = layers.Dense(d, activation = 'relu',
                                   kernel_initializer = randomnormal,
                                   kernel_regularizer = l2)(out)
                out = layers.Dropout(dropout_rate)(out)
    else:
        out = layers.Dense(dff, activation = 'relu', 
                           kernel_initializer = randomnormal,
                          kernel_regularizer = l2)(conc)
        out = layers.Dropout(dropout_rate)(out)
    out = layers.Dense(1, activation = None,
                    kernel_initializer = randomnormal,
                    kernel_regularizer = l2)(out)

    model = Model([n_in, e_in], out)   
    return model


def Delfos_model(embedding_matrix,
                maxlen = 117,
                dim = 300,
                 dff = 400,
                kernel_regularizer = 'l2',
                kernel_initializer = 'randomnormal'):
    if type(embedding_matrix) == str: embedding_matrix = np.load(embedding_matrix)
    if kernel_initializer == 'randomnormal':
        kernel_initializer = initializers.RandomNormal()
    else:
        #TODO: change "rnadomnormal"
        kernel_initializer = kernel_initializer
    if kernel_regularizer == 'l2':
        l2 = regularizers.l2(l = 1e-4)
    else:
        l2 = kernel_regularizer
    dropout_rate = 0.0
    n_in = layers.Input(maxlen)
    e_in = layers.Input(maxlen)
    
    index = len(embedding_matrix)-1
    def create_padding_mask(x):
        mask = tf.cast(tf.math.equal(x, index), tf.float32)
        # batch_size, 1, 1, maxlen
        return mask[:, tf.newaxis, tf.newaxis, :]
    n_mask = layers.Lambda(create_padding_mask, output_shape = (1,1,None))(n_in)
    e_mask = layers.Lambda(create_padding_mask, output_shape = (1,1,None))(e_in)

    embedding_n = layers.Embedding(input_dim = embedding_matrix.shape[0],
                                output_dim = embedding_matrix.shape[1],
                                weights = [embedding_matrix],
                                trainable = False)(n_in)
    embedding_e = layers.Embedding(input_dim = embedding_matrix.shape[0],
                                output_dim = embedding_matrix.shape[1],
                                weights = [embedding_matrix],
                                trainable = False)(e_in)

    N_seq_out= layers.Bidirectional(layers.LSTM(dim, 
                                                return_sequences = True, 
                                                return_state = False,
                                               kernel_regularizer = kernel_regularizer,
                                               kernel_initializer = kernel_initializer))(embedding_n)
    E_seq_out= layers.Bidirectional(layers.LSTM(dim, 
                                                return_sequences = True,
                                               return_state = False,
                                               kernel_regularizer = kernel_regularizer,
                                               kernel_initializer = kernel_initializer))(embedding_e)

    dot_out = tf.matmul(N_seq_out, E_seq_out, transpose_b = True)
    attention_matrix_P= layers.Softmax(axis = -1)(dot_out)
    attention_matrix_Q= layers.Softmax(axis = -1)(tf.transpose(dot_out, (0,2,1)))

    P = tf.matmul(attention_matrix_P, E_seq_out)
    Q = tf.matmul(attention_matrix_Q, N_seq_out)

    H_P = layers.Concatenate(axis = -1)([N_seq_out, P])
    G_Q = layers.Concatenate(axis = -1)([E_seq_out, Q])

    u = layers.GlobalMaxPool1D(data_format = 'channels_last')(H_P)
    v = layers.GlobalMaxPool1D(data_format = 'channels_last')(G_Q)

    conc = layers.Concatenate()([u,v])
    
    if type(dff) == list:
        for i, d in enumerate(dff):
            if i == 0:
                out = layers.Dense(d, activation = 'relu',
                                    kernel_regularizer = kernel_regularizer,
                                    kernel_initializer = kernel_initializer)(conc)
                out = layers.Dropout(dropout_rate)(out)
            else:
                out = layers.Dense(d, activation = 'relu',
                                    kernel_regularizer = kernel_regularizer,
                                    kernel_initializer = kernel_initializer)(out)
                out = layers.Dropout(dropout_rate)(out)
    else:
        out = layers.Dense(dff, activation = 'relu', 
                           kernel_initializer = kernel_initializer,
                          kernel_regularizer = kernel_regularizer)(conc)
        out = layers.Dropout(dropout_rate)(out)
    out = layers.Dense(1, activation = None,
                    kernel_initializer = kernel_initializer,
                    kernel_regularizer = kernel_regularizer)(out)

    model = Model(inputs = [n_in, e_in], outputs = out)
    return model
