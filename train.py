import os
import time
import datetime

import tensorflow as tf
import numpy as np
import data_utils as utils

from tensorflow.contrib import learn
from text_cnn import TextCNN
from data_utils import IMDBDataset


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

# Model Hyperparameters
filter_sizes = [4, 5]
num_filters = 64
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
jac_reg = 0.1

# Training parameters
batch_size = 50
num_epochs = 10
checkpoint_every = 100
num_checkpoints = 6

print("Starting training ...")

print("Dimensions:")

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
            l2_reg_lambda=l2_reg_lambda,
            jac_reg=jac_reg)
        
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)             
                
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
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

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
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
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
        
        batches = utils.batch_iter(
        list(zip(X_train, Y_train)), batch_size, num_epochs)
        
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))