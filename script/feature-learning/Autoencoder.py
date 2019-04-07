# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:33:02 2019

@author: Damon
"""

import tensorflow as tf
import numpy as np
# Parameter
learning_rate = 0.001
training_epochs = 50
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters  &  hidden layer settings
input_dim = 2048  # input features 
hidden1_dim = 1024 # 1st layer num features
hidden2_dim= 512 # 2nd layer num features
hidden3_dim = 256  # 3rd layer num features


def loadData():
    train_data_path = './data_process/data_train.npy'
    train_label_path = './data_process/label_train.txt'
    test_data_path = './data_process/data_test.npy'
    test_label_path = './data_process/label_test.txt'    

    print('Begin loading data...')
    train_X = np.load(train_data_path)
    print('traing_data load done.')
    train_y = np.genfromtxt(fname=train_label_path,
							dtype=np.int, max_rows=37322-14929, skip_header=0)
    print('train_label load done.')
    test_X = np.load(test_data_path)
    print('test_data load done.')
    test_y = np.genfromtxt(fname=test_label_path,
							dtype=np.int, max_rows=14929, skip_header=0)
    print('test_label load done.')
    print('load successfully->\n')

    return train_X, train_y, test_X, test_y


train_X, train_y, test_X, test_y = loadData()

gene_data = np.concatenate((train_X, test_X), axis=0 )

(gene_data_size, feature_size) = gene_data.shape
batch_step = 0
print("data_shape:  ", gene_data.shape)
def next_batch(batch_size):
    global gene_data, gene_data_size, feature_size, batch_step
    start = batch_step*batch_size
    if(start+batch_size < gene_data_size):
        batch_step += 1
        end = batch_step*batch_size
    else:
        end = gene_data_size
        batch_step = 0

    data_batch = np.zeros(shape=[end-start, feature_size], dtype=np.float32)
    for i in range(start, end):
        data_batch[i-start] = gene_data[i]

    return data_batch


def encoder(X):
    weights={
    'encoder_h1' : tf.Variable(tf.random_normal(shape=[input_dim, hidden1_dim])),
    'encoder_h2' : tf.Variable(tf.random_normal(shape=[hidden1_dim, hidden2_dim])),
    'encoder_h3' : tf.Variable(tf.random_normal(shape=[hidden2_dim, hidden3_dim])),
    }

    bias={
    'encoder_b1' : tf.Variable(tf.zeros(shape=[hidden1_dim])),
    'encoder_b2' : tf.Variable(tf.zeros(shape=[hidden2_dim])),
    'encoder_b3' : tf.Variable(tf.zeros(shape=[hidden3_dim])),
    }
    layer1 = tf.nn.relu( tf.matmul(X, weights['encoder_h1']) + bias['encoder_b1'])
    layer2 = tf.nn.relu( tf.matmul(layer1, weights['encoder_h2']) + bias['encoder_b2'])
    layer3 = tf.nn.relu( tf.matmul(layer2, weights['encoder_h3']) + bias['encoder_b3'])

    return layer3

def decoder(X):
    weights={
    'decoder_h1' : tf.Variable(tf.random_normal(shape=[hidden3_dim, hidden2_dim])),
    'decoder_h2' : tf.Variable(tf.random_normal(shape=[hidden2_dim, hidden1_dim])),
    'decoder_h3' : tf.Variable(tf.random_normal(shape=[hidden1_dim, input_dim]))
    }

    bias={
    'decoder_b1' : tf.Variable(tf.zeros(shape=[hidden2_dim])),
    'decoder_b2' : tf.Variable(tf.zeros(shape=[hidden1_dim])),
    'decoder_b3' : tf.Variable(tf.zeros(shape=[input_dim]))
    }
    layer1 = tf.nn.relu( tf.matmul(X, weights['decoder_h1']) + bias['decoder_b1'])
    layer2 = tf.nn.relu( tf.matmul(layer1, weights['decoder_h2']) + bias['decoder_b2'])
    layer3 = tf.nn.relu( tf.matmul(layer2, weights['decoder_h3']) + bias['decoder_b3'])

    return layer3

def loss(logits, labels):
    cost = tf.pow((logits-labels), 2)
    cost = tf.reduce_mean(cost)

    return cost

def trainOp(loss, learning_rate=1e-4):
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, "global_step")
    train_step = optimizer.minimize(loss, global_step=global_step)

    return train_step

def run_training(save_path, test_save_path):

    with tf.Graph().as_default():
        X = tf.placeholder(dtype=tf.float32, shape=(None, input_dim))

        encode_result = encoder(X)
        decode_result = decoder(encode_result)

        loss_value = loss(decode_result, X)
        train_step = trainOp(loss_value, learning_rate)

        init = tf.global_variables_initializer()

        #saver = tf.train.Saver()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter("log", sess.graph)

            sess.run(init)

            train_nums = training_epochs* (gene_data_size // batch_size +1)
            for step_ in range(train_nums):
                batch_x = next_batch(batch_size)
                _, loss_val = sess.run([train_step, loss_value], feed_dict={X:batch_x})

                if(step_%100==0):
                    print("step :", step_, "   loss_value:", loss_value)

            # encode, with trained parameters
            encode_res_train = sess.run(encode_result, feed_dict={X:train_X})
            encode_res_test  = sess.run(encode_result, feed_dict={X:test_X})
            print("data_shape:  ", encode_res_train.shape, encode_res_test.shape)
            np.save(save_path, encode_res_train)
            np.save(test_save_path, encode_res_test)

if __name__ == '__main__':
    #AutoEncoder_data_save_path = r'..\data\AutoEncoder_gene_data_small.npy'
    AutoEncoder_data_save_path = './data_autoencoder/autoencoder_data_{}.npy'.format(hidden3_dim)
    test_save_path = './data_autoencoder/autoencoder_data_{}_test.npy'.format(hidden3_dim)
    run_training(AutoEncoder_data_save_path,test_save_path )


