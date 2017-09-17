
'''This code currently contains the model and the basic training module
'''
import os
import time
import datetime

import tensorflow as tf
import numpy as np
import data_utils as utils

from tensorflow.contrib import learn
#from text_cnn import TextCNN
from data_utils import IMDBDataset


# In[4]:

sequence_length = 128
num_classes = 2
vocab_size = 75099
embedding_dim = 300

print ("Loading Dataset ...")
dataset = IMDBDataset('/home/aayush/robust-large-margin-cnn-develop/data/aclImdb/train', '/home/aayush/robust-large-margin-cnn-develop/data/vocab.pckl')
X, Y = dataset.load()
print ("Dataset loaded. Preparing data and loading embeddings ...")

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(Y)))

X_train = X[shuffle_indices]
Y_train = Y[shuffle_indices]

embedding_path = '/home/aayush/robust-large-margin-cnn-develop/data/embeddings.npy'
embedding = utils.load_embeddings(embedding_path, vocab_size, embedding_dim)
print ("Embeddings loaded. Initialising model hyperparameters ...")


# In[3]:

# embedding.shape


# In[ ]:

def init_weight(self, shape):
    return tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

def init_bias(self, shape):
    return tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

def convolution(self, inp, kernelShape, biasShape):
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="weight")
    b = tf.Variable(tf.constant(0.1, shape=[num_filters // 2]), name="bias")
    conv = tf.nn.conv2d(
        self.h_pool,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="convolution")
    return conv

def non_linearity():
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

def maxpool(self, inp, kernelShape):
        pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length // 2 - filter_size, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print('Maxpool1-{}: {}'.format(filter_size, pooled.get_shape()))
                pooled_outputs.append(pooled)
        return pooled

def fully_connected(self, inp, inpShape, outShape, activation=False):
    weights = self.init_weight([inpShape, outShape])
    bias = self.init_bias(outShape)
    out = tf.matmul(inp, weights) + bias
    if activation:
        return tf.nn.relu(out)
    return out


# In[117]:

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
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv1-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print('Conv1-{}: {}'.format(filter_size, conv.get_shape()))
                
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length // 2 - filter_size, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print('Maxpool1-{}: {}'.format(filter_size, pooled.get_shape()))
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        print('Concatenated: {}'.format(self.h_pool.get_shape()))
        #self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
       
        
        with tf.name_scope("conv2-maxpool"):
            # Convolution Layer
            filter_shape = [4, 1, self.h_pool.get_shape()[3].value, num_filters // 2]
            W_2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_2")
            b_2 = tf.Variable(tf.constant(0.1, shape=[num_filters // 2]), name="b_2")
            conv_2 = tf.nn.conv2d(
                self.h_pool,
                W_2,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_2")
            print('Conv2: {}'.format(conv_2.get_shape()))
            
            # Apply nonlinearity
            h_2 = tf.nn.relu(tf.nn.bias_add(conv_2, b_2), name="relu_2")            
            
            # Maxpooling over the outputs
            self.pooled_2 = tf.nn.max_pool(
                h_2,
                ksize=[1, 128 // 8 - 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool_2")
            print('Maxpool2: {}'.format(self.pooled_2.get_shape()))
            
        with tf.name_scope("conv3-maxpool"):
            # Convolution Layer
            filter_shape = [6, 1, self.pooled_2.get_shape()[3].value, num_filters // 2]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_3")
            b_3 = tf.Variable(tf.constant(0.1, shape=[num_filters // 2]), name="b_3")
            conv_3 = tf.nn.conv2d(
                self.pooled_2,
                W_3,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_3")
            print('Conv3: {}'.format(conv_3.get_shape()))
            
            # Apply nonlinearity
            h_3 = tf.nn.relu(tf.nn.bias_add(conv_3, b_3), name="relu_3")            
            
            # Maxpooling over the outputs
            self.pooled_3 = tf.nn.max_pool(
                h_3,
                ksize=[1, conv_3.get_shape()[1].value, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool_3")
            print('Maxpool3: {}'.format(self.pooled_3.get_shape()))
        
        # Flatten into a long feature vector
        self.h_pool_flat = tf.reshape(self.pooled_3, [-1, num_filters // 2])
        print(self.h_pool_flat.get_shape())
        
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters // 2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# ##To do
# - Rewrite each layer as a function of class
# - Introduce network depth

# In[118]:

# Model Hyperparameters
filter_sizes = [4, 5]
num_filters = 64
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

# Training parameters
batch_size = 50
num_epochs = 10
checkpoint_every = 100
num_checkpoints = 6


# In[119]:

print("Starting training ...")

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=sequence_length,
            num_classes=num_classes,
            vocab_size=vocab_size,
            embedding_size=embedding_dim,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Checkpoints and logs will be written into {}\n".format(out_dir))

        # Creating heckpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        sess.run(cnn.embedding_init, feed_dict={cnn.embedding_placeholder: embedding})
        
        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        
        batches = utils.batch_iter(
        list(zip(X_train[:1000], Y_train[:1000])), batch_size, num_epochs)
        
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

