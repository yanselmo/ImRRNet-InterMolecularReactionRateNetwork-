import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers


class MultiHeadInterMolecularAttention(layers.Layer):
    """Inter multihead attention layer. An Attention layer based transformer and Delfos.

    args
       d_model: dimension of model. dtype = int
       num_heads: number of heads. d_model/numhead sholu be int. dtype = int

    kwargs
       kernel_initializer = None : keras initializer object initializing kernel weights
       kernel_regularizer = None : keras regularizer object regularizing kernel weights to prevent overfitting
    
    args of call
        inputs: list. len of inputs can be 2 or 4.
            if len == 2, inputs include two inputs
            if len == 4, there are additional two inputs: mask of two inputs
    """
    def __init__(self, 
                 d_model, 
                 num_heads,
                 kernel_initializer = None,
                 kernel_regularizer = None, 
                 dropout_rate = 0, 
                 **kwargs):
        super(MultiHeadInterMolecularAttention, self).__init__(**kwargs)
        self.num_heads = num_heads 
        self.d_model = d_model
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        if dropout_rate == 0:
            self.use_dropout = False
            self.dropout_rate = dropout_rate
        else:
            self.use_dropout = True
            self.dropout_rate = dropout_rate
        assert d_model % self.num_heads == 0 
        
        self.depth = d_model // self.num_heads
        
        self.a_q_dense = layers.Dense(d_model, 
                                           name = 'q_a',
                                          kernel_initializer = self.kernel_initializer,
                                          kernel_regularizer = self.kernel_regularizer) #query weights for a
        self.a_k_dense = layers.Dense(d_model, 
                                           name = 'k_a',
                                          kernel_initializer = self.kernel_initializer,
                                          kernel_regularizer = self.kernel_regularizer) #key weights for a
        self.a_v_dense = layers.Dense(d_model, 
                                           name = 'v_a',
                                          kernel_initializer = self.kernel_initializer,
                                          kernel_regularizer = self.kernel_regularizer) #value weights for a

        self.b_q_dense = layers.Dense(d_model, 
                                            name = 'q_b',
                                           kernel_initializer = self.kernel_initializer,
                                           kernel_regularizer = self.kernel_regularizer) #query weights for b
        self.b_k_dense = layers.Dense(d_model, 
                                            name = 'k_b',
                                           kernel_initializer = self.kernel_initializer,
                                           kernel_regularizer = self.kernel_regularizer) #key weights for a
        self.b_v_dense = layers.Dense(d_model, 
                                            name = 'v_b',
                                           kernel_initializer = self.kernel_initializer,
                                           kernel_regularizer = self.kernel_regularizer) #value weights for a
        
        self.a_dense = layers.Dense(d_model, 
                                         name = 'a_attention',
                                        kernel_initializer = self.kernel_initializer,
                                        kernel_regularizer = self.kernel_regularizer)
        self.b_dense = layers.Dense(d_model, 
                                          name = 'b_attention',
                                         kernel_initializer = self.kernel_initializer,
                                         kernel_regularizer = self.kernel_regularizer)
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape = (batch_size, -1, self.num_heads, self.depth))

        #batch, heads, timesteps, depth
        return tf.transpose(inputs, perm=[0,2,1,3])

    def scaled_dot_attention(self, q, k, v, mask_other, mask_self):
        matmul_qk = tf.matmul(q, k, transpose_b = True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask_other != None and mask_self != None:
            logits += (mask_other * -1e9)
        attention_weights = tf.nn.softmax(logits, axis = -1)
        if mask_other != None and mask_self != None:
            attention_weights = attention_weights * (1 - tf.transpose(mask_self, perm=[0,1,3,2])) + (1e-9 * tf.transpose(mask_self, perm = [0,1,3,2]))

        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, inputs):
        """ inputs should be list.
        len of inptuts can be 2 or 4
        if 2, inputs include input a and input b.
        if 4, inputs include additional 2 inputs, mask a and mask b
        """
        assert len(inputs)==2 or len(inputs)==4

        a = inputs[0]
        b = inputs[1]
        if len(inputs)==4:
            mask_a = inputs[2]
            mask_b = inputs[3]
        else:
            mask_a = None
            mask_b = None

        batch_size = tf.shape(a)[0]
        if self.use_dropout:
            a_q = self.a_q_dense(a)
            a_q = layers.Dropout(self.dropout_rate)(a_q)
            a_k = self.a_k_dense(a)
            a_k = layers.Dropout(self.dropout_rate)(a_k)
            a_v = self.a_v_dense(a)
            a_v = layers.Dropout(self.dropout_rate)(a_v)
            b_q = self.b_q_dense(b)
            b_q = layers.Dropout(self.dropout_rate)(b_q)
            b_k = self.b_k_dense(b)
            b_k = layers.Dropout(self.dropout_rate)(b_k)
            b_v = self.b_v_dense(b)
            b_v = layers.Dropout(self.dropout_rate)(b_v)
        else:
            a_q = self.a_q_dense(a)
            a_k = self.a_k_dense(a)
            a_v = self.a_v_dense(a)

            b_q = self.b_q_dense(b)
            b_k = self.b_k_dense(b)
            b_v = self.b_v_dense(b)

        # implement multihead
        a_q = self.split_heads(a_q, batch_size)
        a_k = self.split_heads(a_k, batch_size)
        a_v = self.split_heads(a_v, batch_size)
        b_q = self.split_heads(b_q, batch_size)
        b_k = self.split_heads(b_k, batch_size)
        b_v = self.split_heads(b_v, batch_size)

        #a_attention, self.a_attention_weights = self.scaled_dot_attention(a_q, b_k, b_v, mask_a)
        a_attention, self.a_attention_weights = self.scaled_dot_attention(a_q, b_k, b_v, mask_b, mask_a)
        a_attention = tf.transpose(a_attention, perm = [0,2,1,3])
        self.a_attention_by_heads = a_attention
        # batch, heads, timesteps, depth -> batch, timesteps, heads, depth
        a_attention = tf.reshape(a_attention, (batch_size, -1, self.d_model))
        self.a_attention_total = a_attention

        #b_attention, self.b_attention_weights = self.scaled_dot_attention(b_q, a_k, a_v, mask_b)
        b_attention, self.b_attention_weights = self.scaled_dot_attention(b_q, a_k, a_v, mask_a, mask_b)
        b_attention = tf.transpose(b_attention, perm = [0,2,1,3])
        self.b_attention_by_heads = b_attention
        b_attention = tf.reshape(b_attention, (batch_size, -1, self.d_model))
        self.b_attention_total = b_attention
        
        if self.use_dropout:
            a_out = self.a_dense(a_attention)
            a_out = layers.Dropout(self.dropout_rate)(a_out)
            b_out = self.b_dense(b_attention)
            b_out = layers.Dropout(self.dropout_rate)(b_out)
        else:
            a_out = self.a_dense(a_attention)
            b_out = self.b_dense(b_attention)

        return [a_out, b_out]

    def get_config(self):
        base_config = super(MultiHeadInterMolecularAttention, self).get_config()
        base_config['num_heads'] = self.num_heads
        base_config['d_model'] = self.d_model
        base_config['depth'] = self.depth

        return base_config