import tensorflow as tf
from math import sqrt
import numpy as np
from tf_helpers import learning_rate_multiplier

FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 32


def model(
    use_context,
    lang_dim,
    context_dim,
    embedding_dim,
    sequence_length,
    kind,
    test_model=False
):
    print_tensor = tf.constant(False)
    print_tensor2 = tf.constant(False)
    
    with tf.variable_scope("setup"):
        # input
        if use_context:
            x_context = tf.placeholder(tf.float32, [None, context_dim], name="x_context")

        x_text = tf.placeholder(tf.float32, [None, lang_dim, embedding_dim], name="x_text")
        y = tf.placeholder(tf.float32, None, name="y")

        # dropout
        dropout = tf.placeholder_with_default(.0, shape=(), name="dropout_rate")
    
        # L2 regularization
        l2_loss = tf.constant(0.0)

    with tf.variable_scope("text"):
        if kind == 'LSTM-CNN':
            lstm_cell = tf.keras.layers.LSTMCell(embedding_dim, name="LSTM-cell")

            lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell, x_text, dtype=tf.float32)
            lstm_out_expanded = tf.expand_dims(lstm_out, -1, name='expand-dimensions')

            # CONVOLUTION LAYER + MAXPOOLING LAYER (per filter)
            pooled_outputs = []
            for filter_size in FILTER_SIZES:
                # CONVOLUTION LAYER
                filter_shape = [filter_size, embedding_dim, 1, NUM_FILTERS]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]), name="b")
                conv = tf.nn.conv2d(lstm_out_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # NON-LINEARITY
                h = tf.nn.sigmoid(tf.nn.bias_add(conv, b), name="sigmoid")
                # MAXPOOLING
                pooled = tf.nn.max_pool(
                        h,
                        ksize=[
                            1,
                            sequence_length - filter_size + 1, 1, 1
                        ],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool"
                    )
                pooled_outputs.append(pooled)

            num_filters_total = NUM_FILTERS * len(FILTER_SIZES)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        elif kind == 'CNN':
            embedded_chars_expanded = tf.expand_dims(x_text, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(FILTER_SIZES):
                # Convolution Layer
                filter_shape = [filter_size, embedding_dim, 1, NUM_FILTERS]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool"
                )
                pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = NUM_FILTERS * len(FILTER_SIZES)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            h_pool_flat_dropout = tf.nn.dropout(h_pool_flat, rate=dropout)
        else:
            raise ValueError

    if use_context:
        with tf.variable_scope("context"):
            hydro_dense_nodes = 20
            hydro_dense = tf.layers.dense(
                x_context,
                hydro_dense_nodes,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
                kernel_initializer=tf.initializers.random_uniform(0, 2 * sqrt(6 / context_dim)),
                activation=tf.nn.relu,
            )
            hydro_output_node = tf.layers.dense(
                hydro_dense,
                1,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
                kernel_initializer=tf.initializers.random_uniform(0, 2 * sqrt(6 / hydro_dense_nodes)),
                activation=tf.nn.relu,
            )

    with tf.variable_scope("combination"):
        if use_context:
            combination_input = tf.concat([h_pool_flat_dropout, hydro_output_node], axis=1)
        else:
            combination_input = h_pool_flat_dropout

        another_dense = tf.layers.dense(
            combination_input,
            20,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),  # xavier initializer works well with sigmoid/tanh
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.),
            activation=tf.nn.tanh,
        )

        another_dense = tf.layers.dense(
            another_dense,
            20,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),  # xavier initializer works well with sigmoid/tanh
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.),
            activation=tf.nn.tanh,
        )

        # another_dense = tf.nn.dropout(another_dense, rate=0.1)
        logits = tf.squeeze(tf.layers.dense(
            another_dense,
            1, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(),  # xavier initializer works well with sigmoid/softmax
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.),
            activation=tf.nn.sigmoid   
        ), axis=1, name="logits")  # tf.squeeze to remove dimension

        predictions = tf.cast(tf.round(logits, name="predictions"), tf.int32)
        actual = tf.cast(tf.round(y), tf.int32)
        
    l2_loss += tf.losses.get_regularization_loss()

    if use_context:
        x = (x_text, x_context)
    else:
        x = x_text
    return (
        x,
        y,
        actual,
        logits,
        predictions,
        dropout,
        l2_loss,
        print_tensor,
        print_tensor2
    )