# based on github.com/scharmchi/char-level-cnn-tf

import tensorflow as tf

class CharCNN(object):
    """
    A CNN for text classification.
    based on the Character-level Convolutional Networks for Text Classification.
    """
    def __init__(self, sequence_length, quantization_size, num_classes, filter_sizes, num_filters, 
        learning_rate, l2_reg_lambda=0.0, jac_reg=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, quantization_size, sequence_length, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        layer_outputs  = []
        # Layer 1
        with tf.name_scope("conv-maxpool-1"):
            filter_shape = [quantization_size, filter_sizes[0], 1, num_filters]
            with tf.variable_scope("conv-maxpool-1",  reuse=None):
                W = tf.get_variable(shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='VALID',
                name="pool1")
            layer_outputs.append(pooled)

        # Layer 2
        with tf.name_scope("conv-maxpool-2"):
            filter_shape = [1, filter_sizes[1], num_filters, num_filters]
            with tf.variable_scope("conv-maxpool-2",  reuse=None):
                W = tf.get_variable(shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='VALID',
                name="pool2")
            layer_outputs.append(pooled)

        # Layer 3
        with tf.name_scope("conv-3"):
            filter_shape = [1, filter_sizes[2], num_filters, num_filters]
            with tf.variable_scope("conv-3",  reuse=None):
                W = tf.get_variable(shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            layer_outputs.append(h)

        # Layer 4
        with tf.name_scope("conv-4"):
            filter_shape = [1, filter_sizes[3], num_filters, num_filters]
            with tf.variable_scope("conv-4",  reuse=None):
                W = tf.get_variable(shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv4")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            layer_outputs.append(h)

        # Layer 5
        with tf.name_scope("conv-5"):
            filter_shape = [1, filter_sizes[4], num_filters, num_filters]
            with tf.variable_scope("conv-5",  reuse=None):
                W = tf.get_variable(shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv5")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            layer_outputs.append(h)

        # Layer 6
        with tf.name_scope("conv-maxpool-6"):
            filter_shape = [1, filter_sizes[5], num_filters, num_filters]
            with tf.variable_scope("conv-maxpool-6",  reuse=None):
                W = tf.get_variable(shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv6")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='VALID',
                name="pool6")
            layer_outputs.append(pooled)

        # Layer 7
        feature_vec_length = 34 * num_filters
        h_pool_flat = tf.reshape(pooled, [-1, feature_vec_length])

        # Add dropout
        with tf.name_scope("dropout-1"):
            drop1 = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # First fully connected layer
        with tf.name_scope("fc-1"):
            with tf.variable_scope("fc-1",  reuse=None):
                W = tf.get_variable(shape=[feature_vec_length, 1024], initializer=tf.truncated_normal_initializer(stddev=0.05), name="W")
            # W = tf.get_variable("W", shape=[num_features_total, 1024],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(drop1, W, b), name="fc-1-out")
            layer_outputs.append(fc_1_output)

        # Layer 8
        # Add dropout
        with tf.name_scope("dropout-2"):
            drop2 = tf.nn.dropout(fc_1_output, self.dropout_keep_prob)

        # Second fully connected layer
        with tf.name_scope("fc-2"):
            with tf.variable_scope("fc-2",  reuse=None):
                W = tf.get_variable(shape=[1024, 1024], initializer=tf.truncated_normal_initializer(stddev=0.05), name="W")
            # W = tf.get_variable("W", shape=[1024, 1024],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(drop2, W, b), name="fc-2-out")
            layer_outputs.append(fc_2_output)

        # Layer 9
        # Output layer
        with tf.name_scope("output"):
            with tf.variable_scope("output",  reuse=None):
                W = tf.get_variable(shape=[1024, num_classes], initializer=tf.truncated_normal_initializer(stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            softmax_pred = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(softmax_pred) + l2_reg_lambda * l2_loss

        # Jacobian Regularizer 
        with tf.name_scope("jacobian-regularizer"):
            if jac_reg > 0.0:
                layer_names = ["conv-maxpool-1", "conv-maxpool-2",
                               "conv-3", "conv-4", "conv-5",
                               "conv-maxpool-6", "fc-1", "fc-2"]
                for (idx, scope) in enumerate(layer_names):
                    with tf.variable_scope(scope, reuse=True):
                        W = tf.get_variable("W")

                        # jacobian matrix of network output w.r.t. the outputs of layer L
                        # dimension: (batch_size, width, height, number of filters)

                        # g_x = tf.gradients(tf.reduce_sum(tf.multiply(self.input_y, self.scores)), layer_outputs[idx])
                        g_x = tf.gradients(tf.multiply(self.input_y,
                                self.scores), layer_outputs[idx])

                        # reshape (batch_size, height, width, number of filters) to (batch_size * width, number of filters)
                        if scope not in ["fc-1", "fc-2"]:
                            g_x = tf.reshape(g_x, shape=[-1, tf.shape(W)[3]])
                        else:
                            g_x = tf.squeeze(g_x)

                        # covariance matrix of jacobian vectors
                        reg = tf.matmul(tf.transpose(g_x), g_x)

                        # parameter update
                        W -= learning_rate * jac_reg * tf.tensordot(reg, W, axes=[[1], [0]])

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
