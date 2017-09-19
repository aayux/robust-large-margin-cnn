import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn

class TextCNN(object):
    """
    A CNN for text classification.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
       
        # Embedding layer
        self.word_embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        self.embedding_init = self.word_embedding.assign(self.embedding_placeholder)
        
        with tf.device('/cpu:0'), tf.name_scope("embedding"):            
            self.embedded_chars = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        print('Embedding: {}'.format(self.embedded_chars_expanded.get_shape()))
        
        # Create a convolution + maxpool layer for each filter size
        layer_outouts = []
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv%s-maxpool-1" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                with variable_scope(("conv%s-maxpool-1" % filter_size),  reuse=None):
                    W = self.init_weight(filter_shape)
                b = self.init_bias([num_filters])
                conv = self.convolution(self.embedded_chars_expanded, W)
                print('Conv1-{}: {}'.format(filter_size, conv.get_shape()))
                
                # Apply nonlinearity
                h = self.non_linearity(conv, bias)
                
                # Maxpooling over the outputs
                ksize = [1, sequence_length // 2 - filter_size, 1, 1]
                pooled = self.maxpool(h, ksize)
                print('Maxpool1-{}: {}'.format(filter_size, pooled.get_shape()))
                pooled_outputs.append(pooled)

        layer_outputs = pooled_outputs

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        print('Concatenated: {}'.format(self.h_pool.get_shape()))       
        
        # Second convolution + maxpool layer
        with tf.name_scope("conv-maxpool-2"):
            filter_shape = [4, 1, self.h_pool.get_shape()[3].value, num_filters // 2]
            with variable_scope("conv-maxpool-2",  reuse=None):
                W = self.init_weight(filter_shape)
            b = self.init_bias([num_filters // 2])
            conv = self.convolution(self.h_pool, W)
            print('Conv2: {}'.format(conv.get_shape()))
            
            h = self.non_linearity(conv, bias)
            
            ksize = [1, 128 // 8 - 1, 1, 1]
            self.pooled_2 = self.maxpool(h, ksize)

            layer_outputs.append(self.pooled_2)
            print('Maxpool2: {}'.format(self.pooled_2.get_shape()))
        
        # Third convolution + maxpool layer            
        with tf.name_scope("conv-maxpool-3"):
            filter_shape = [6, 1, self.pooled_2.get_shape()[3].value, num_filters // 2]
            with variable_scope("conv-maxpool-3",  reuse=None):
                W = self.init_weight(filter_shape)
            b = self.init_bias([num_filters // 2])
            conv = self.convolution(self.pooled_2, W)
            print('Conv3: {}'.format(conv.get_shape()))
            
            h = self.non_linearity(conv, bias)
            
            ksize=[1, conv_3.get_shape()[1].value, 1, 1]
            self.pooled_3 = self.maxpool(h, ksize)
            
            layer_outputs.append(self.pooled_2)
            print('Maxpool3: {}'.format(self.pooled_3.get_shape()))
        
        # Flatten into a long feature vector
        self.h_pool_flat = tf.reshape(self.pooled_3, [-1, num_filters // 2])
        print(self.h_pool_flat.get_shape())
        
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = self.add_dropout(self.h_pool_flat, self.dropout_keep_prob)
        
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            with variable_scope("output",  reuse=None):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters // 2, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        
        # Jacobian Regularizer            
        with tf.name_scope("jacobian_reg"):
            if jac_reg > 0.0:
                layer_names = ["conv4-maxpool-1", "conv5-maxpool-1", "conv-maxpool-2", "conv-maxpool-3", "output"]
                for idx, scope in enumerate(layer_names):
                    with tf.variable_scope(scope, reuse=True):
                        W = tf.get_variable("W")
                        # jacobian matrix of network output w.r.t. the outputs of layer L
                        # dimension: (batch_size, number of filters, height, width)
                        # reshaped to: (batch_size, height, width, number of filters) - > (batch_size*height*width, number of filters)
                        
                        # NOTE: This is not correct. Dimensions need to be fixed but general arithmetic is correct.
                        g_x = tf.gradients(tf.add(tf.multiply(input_y, self.predictions), layer_outputs[idx]).dimshuffle((0, 2, 3, 1)).reshape(
                                (-1, W.shape[0])))
                        
                        # covariance matrix of jacobian vectors
                        reg = tf.matmul(tf.transpose(g_x), g_x)

                        # parameter update
                        # NOTE: This is not correct. Dimensions need to be fixed but general arithmetic is correct.
                        W -= 1e-3 * jac_reg * tf.tensordot(reg, W, axes=[[1], [0]])

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    # TO DO:
    # Add Fully Connected layer
    # Fix Jacobian Regularizer


    def init_weight(self, shape):
        return tf.get_variable("W", shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))

    def init_bias(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape), name="b")

    def non_linearity(self, conv, bias):
        return tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")

    def add_dropout(self, drop_input, keep_prob):
        return tf.nn.dropout(drop_input, keep_prob)

    def convolution(self, conv_input, weights):
        conv = tf.nn.conv2d(
            conv_input,
            weights,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="convolution")
        return conv

    def maxpool(self, pool_input, ksize):
            pooled = tf.nn.max_pool(
                        pool_input,
                        ksize=ksize,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
            return pooled

    def fully_conected(self, fc_in, in_shape, out_shape):
        W =  tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = self.init_bias(out_shape)
        return tf.nn.xw_plus_b(fc_in, W, b, name="fully_connected")
