import os
import time
import datetime

import numpy as np
import pickle as pckl
import tensorflow as tf
import data_utils as utils

from char_cnn import CharCNN
from data_utils import YelpDataset
from tensorflow.contrib import learn

# Load dataset
print ("Loading Dataset ...")

pcklfile = "./data/dump.pckl"

if not os.path.isfile(pcklfile):
    print ("No data dump found. Pickling dataset ...")
    dataset = YelpDataset('./data/review.json')
    X, Y = dataset.load()
    pckl.dump([X, Y], open(pcklfile, "wb"))
else:
    X, Y = pckl.load(open(pcklfile, "rb"))

print ("Dataset loaded. Preparing data ...")

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(Y)))

x_shuff = X[shuffle_indices]
y_shuff = Y[shuffle_indices]

# Percentage of the training data to use for validation
val_sample = .2

# Split train/test set
idx = -1 * int(val_sample * float(len(Y)))
x_train, x_val = x_shuff[:idx], x_shuff[idx:]
y_train, y_val = y_shuff[:idx], y_shuff[idx:]
print("Train/Val split: {:d}/{:d}".format(len(y_train), len(y_val)))

# Input parameters
sequence_length = 1014
quantization_size = 70
num_classes = 2

# Model parameters
filter_sizes = (7, 7, 3, 3, 3, 3)
n_layers = 8
num_filters = 256
l2_reg_lambda = 0.0
jac_reg = 0.0

# Training parameters
batch_size = 128
num_epochs = 50
starter_learning_rate = 1e-3
checkpoint_every = 1000
validate_every = 5000
num_checkpoints = 3

print("Starting training ...")

print("Dimensions:")

print("Starting training ...")

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CharCNN(
            sequence_length=sequence_length,
            quantization_size=quantization_size,
            num_classes=num_classes,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)
        
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   10000, 0.5, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)        
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
       
        # Jacobian Regularizer
        idx = 0
        w_update = [0 for _ in range(n_layers)]
        for _, w in grads_and_vars:
            if ("W" in w.name) and ("output" not in w.name):
                # jacobian matrix of network output w.r.t. the outputs of layer L
                g = tf.gradients(tf.multiply(cnn.input_y, cnn.scores), cnn.out_accumulator[idx])
                
                # reshape (batch_size, height, width, depth) to (batch_size * height * width, depth)
                dim = w.get_shape()[-1].value
                g = tf.reshape(g, shape=[-1, dim])
                
                # covariance matrix of jacobian vectors
                gg = tf.matmul(tf.transpose(g), g)
                
                # update step
                w_update[idx] = tf.assign_sub(w, 
                    learning_rate * jac_reg_alpha * tf.tensordot(var, gg, axes=[[-1], [1]]))
                idx += 1
      
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Val summaries
        val_summary_op = tf.summary.merge([loss_summary, acc_summary])
        val_summary_dir = os.path.join(out_dir, "summaries", "val")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        #ckpt = tf.train.get_checkpoint_state(os.path.dirname('./runs/<>'))
        #if ckpt and ckpt.model_checkpoint_path:
        #    saver.restore(sess, ckpt.model_checkpoint_path)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
               
        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
            }
            _, step, summaries, loss, accuracy, weight_update = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.W_update],
                feed_dict)
            if jace_reg > 0.:
                for idx in range(n_layers):
                    sess.run(w_update[idx], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def validation_step(x_batch, y_batch, writer=None):
            val_size = len(x_batch)
            # batch size is chosen arbitrarily
            batch_size = 500
            n_batches = val_size / batch_size
            
            for idx in range(n_batches):
                x_batch_val, y_batch_val = utils.one_hot_x(x_batch, 
                                            y_batch, idx * batch_size, 
                                            (idx +1) * batch_size)
                feed_dict = {
                    cnn.input_x: x_batch_val,
                    cnn.input_y: y_batch_val,
                    cnn.dropout_keep_prob: 1.0
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, val_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                
                time_str = datetime.datetime.now().isoformat()
                print("{}: val_batch {}, loss {:g}, acc {:g}".format(time_str, idx + 1, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
        
        batches = utils.batch_iter(x_train, y_train, batch_size, num_epochs)
        
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            
            if current_step % validate_every == 0:
                print("\nValidation: ")
                validation_step(x_val, y_val, writer=val_summary_writer)

            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
