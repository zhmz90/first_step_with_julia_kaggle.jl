from __future__ import division
from __future__ import print_function

import os
from os.path import join
import sys
import time
import random

import numpy as np
from six.moves import xrange
import tensorflow as tf

NUM_LABELS = 4
VALIDATION_SIZE = 1000
SEED = 201607
BATCH_SIZE = 64
NUM_EPOCHS = 1000
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100
NUM_LABELS = 4
NUM_CHANNELS = 1

tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Use half floats instead of full floats if True.")
tf.app.flags.DEFINE_boolean("debug", False, "Run in debug mode")
FLAGS = tf.app.flags.FLAGS

def data_type():
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32

def countlines(file_name):
    with open(file_name) as file:
        for i,l in enumerate(file):
            pass
    return i+1

def split_data(data, ratios=[0.6,0.8]): 
    nrow = len(data)
    if ratios == [0.6,0.8]:
        train = data[0:1473232] # # potential bug due to fix val
        val   = data[1473232:1973232]
        test  = data[1973232:]
    else:
        tr_end_index = int(nrow*ratios[0])
        val_end_index = int(nrow*ratios[1])
        train = data[0:tr_end_index]
        val   = data[tr_end_index:val_end_index]
        test  = data[val_end_index:]
    return train,val,test

def split_context_target(data):
    return map(lambda x:x.strip().split("\t"), data)

def get_context_data(context_file=join(WORK_DIRECTORY, "mutation_meta_context.tsv")):
    raw  = []
    with open(context_file) as file:
        header = file.readline()
        for i,line in enumerate(file):
            raw.append(line)
    random.shuffle(raw)
    train_raw,val_raw,test_raw = split_data(raw)
    train = split_context_target(train_raw)
    val   = split_context_target(val_raw)
    test  = split_context_target(test_raw)
    return train,val,test

def column_index_of_encoder(s):
    if s == 'A' or s == 'a':
        index = 0
    elif s == 'T' or s == 't':
        index = 1
    elif s == 'C' or s == 'c':
        index = 2
    elif s == 'G' or s == 'g':
        index = 3
    elif s == 'N' or s == 'n':
        index = random.choice([0,1,2,3])
    else:
        raise Exception("unknown nucletides %s", s)
    return index

def onehotencode_nt(s):
    l = len(s)
    assert l == 1, "Unexpected character of %s" % s
    try:
        col_index = column_index_of_encoder(s)
    except:
        print("Unexpected target %s" % s)
        raise Exception("Unexpected target %s", s)
    return col_index

def onehotencode_nts(seq):
    """
    ATCG --> [[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]]
    """
    l = len(seq)
    assert l > 1, "the sequence %s cann't be less than 1" % seq
    data = np.zeros((4,l))
    for i,s in enumerate(seq):
        try:
            col_index = column_index_of_encoder(s)
        except:
            print("Unexpected the %d nucletid %s of context %s" % (i, s, seq))
            raise Exception("Unexpected the %d nucletid %s of context %s", i, s, seq)
        data[col_index, i] = 1
    return data

count = 0
def numericalize_context_target(example):
    if FLAGS.debug:
        global count
        count += 1
    x = onehotencode_nts(example[1]+example[0])
    y = onehotencode_nt(example[2])
    return (x,y)

def numericalize_text(data):
    data = map(numericalize_context_target, data) # [(x1,y1),(x2,y2) ...] x1: np.zeros((4,l))
    # should be 3-dim 
    # [num_example, context_len, 4]
    num_example = len(data)
    context_len = data[0][0].shape[1]
    print(context_len)
    X = np.empty((num_example,4,context_len))
    Y = np.empty(num_example)
    for i in xrange(num_example):
        X[i] = data[i][0]
        Y[i] = data[i][1]
    return X,Y


def error_rate(predictions, labels):
    return 100.0 - (
        100.0 * 
        np.sum(np.argmax(predictions,1) == labels) /
        predictions.shape[0])

def main(argv=None):
    # Get the context data 
    preprocess_start = time.time()
    print("Geting text data from context file") # better output context_len
    train_text_data, val_text_data, test_text_data = get_context_data() # shuffled
    print("Numericalizing text data")
    if FLAGS.debug:
        train_text_data = train_text_data[0:100000]
        val_text_data   = val_text_data[100000:150000]
        test_text_data  = test_text_data[2000:3000]
    train_data,train_labels = numericalize_text(train_text_data)
    val_data,val_labels     = numericalize_text(val_text_data)
    test_data,test_labels   = numericalize_text(test_text_data)
    preprocess_end  = time.time()

    context_len = len(train_data[0][0])
    print("Context len is %d" % context_len)
    train_size = len(train_labels)
    print("There are %d training examples" % train_size)
    print("There are %d validation examples" % len(val_labels))
    print("There are %d test examples" % len(test_labels))
    print("Processing time cost: %fs" % ((preprocess_end - preprocess_start)/60))

    # reshape data
    train_data = np.reshape(train_data, (train_data.shape +(1,)) )
    val_data = np.reshape(val_data, (val_data.shape + (1,)) )
    test_data = np.reshape(test_data, (test_data.shape+(1,)) )

    # build model
    print("Building CNN")
    train_data_node = tf.placeholder(data_type(), shape=(BATCH_SIZE, 4, context_len, NUM_CHANNELS)) # 4*201*1
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))#(BATCH_SIZE, 4))
    eval_data = tf.placeholder(data_type(), shape=(BATCH_SIZE, 4, context_len, NUM_CHANNELS))
    
    conv1_weights = tf.Variable( # 4x5 filter wiht depth 32
        tf.truncated_normal([4, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=data_type()))
    conv1_biases = tf.Variable(tf.zeros([32]), dtype=data_type())
    conv2_weights= tf.Variable(
        tf.truncated_normal([2, 5, 32, 64], stddev=0.1, seed=SEED, dtype=data_type())) # 
    conv2_biases  = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
    fc1_weights  = tf.Variable(
            tf.truncated_normal([51 * 64, 512], stddev=0.1, seed=SEED, dtype=data_type()))
    fc1_biases   = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
    fc2_weights  = tf.Variable(
        tf.truncated_normal([512, 4], stddev=0.1, seed=SEED, dtype=data_type()))
    fc2_biases   = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))

    def model(data, train=False):
        conv = tf.nn.conv2d(data,conv1_weights,strides=[1,1,1,1],padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        conv = tf.nn.conv2d(pool,conv2_weights,strides=[1,1,1,1],padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        
        pool_shape = pool.get_shape().as_list()
        print("pool_shape is %s" % pool_shape)
        reshape = tf.reshape(pool,[pool_shape[0],pool_shape[1]*pool_shape[2]*pool_shape[3]])
        hidden  = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    logits = model(train_data_node, True)
    if FLAGS.debug:
        print(logits)
        print(train_labels_node)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, train_labels_node))
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    loss += 5e-4 * regularizers
    
    batch = tf.Variable(0, dtype=data_type())
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate. 
        batch * BATCH_SIZE,  # Current index into the dataset.      
        train_size,          # Decay step.
        0.95,                # Decay rate.     
        staircase=True)
    
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step = batch)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = batch)

    train_prediction = tf.nn.softmax(logits)
    eval_prediction  = tf.nn.softmax(model(eval_data))
    
    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end,:] = sess.run(eval_prediction, feed_dict={eval_data:data[begin:end,...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction, feed_dict={eval_data:data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:,:] = batch_predictions[begin-size:,:]
        return predictions
        
    if FLAGS.debug:
        num_epochs = 10000
    else:
        num_epochs = NUM_EPOCHS

    start_time = time.time()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print("Initialized tensorflow graph")
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE), ...]
            
            feed_dict = {train_data_node: batch_data, train_labels_node:batch_labels}
            _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction],
                                              feed_dict=feed_dict)
            if step % EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time
                start_time   = time.time()
                print("Step %d (epoch %.2f), %.1f ms" %
                      (step, float(step)*BATCH_SIZE/train_size, 1000*elapsed_time/EVAL_FREQUENCY))
                print("Minibatch loss: %.3f, learning rate: %.6f" % (l,lr))
                print("Minibatch error: %.1f%%" % error_rate(predictions, batch_labels))
                print("Validation error: %.1f%%" % error_rate(eval_in_batches(val_data, sess),
                                                              val_labels))
                sys.stdout.flush()
            #test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
            #print("Test error: %.1f%%" % test_error)
        
if __name__ == "__main__":
    tf.app.run()
