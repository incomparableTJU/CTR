#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:55:50 2018

@author: weiqing
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class autoencoder(object):
    learning_rate = 1e-2
    def __init__(self,input_size=273,hidden_layers = [256,128,64,32],
                 model_path= 'model/autoencoder/autoencoder.tfmodel'):
        self.load_hidden_sizes(input_size,hidden_layers)
        self.model_path = model_path
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder("float", [None, input_size])
            self.encoder_op = self.coder(self.X,self.encoder_sizes,tf.nn.sigmoid)
            self.decoder_op = self.coder(self.encoder_op,self.decoder_sizes,tf.nn.sigmoid)
            self.y_pred = self.decoder_op
            self.y_true = self.X
            self.loss = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
    def load_hidden_sizes(self,input_size,hidden_layers):
        last_size = input_size
        self.encoder_sizes = []
        self.decoder_sizes = []
        for hidden_size in hidden_layers:
            self.encoder_sizes += [[last_size,hidden_size]]
            last_size = hidden_size
        for hidden_size in hidden_layers[-2::-1]+[input_size]:
            self.decoder_sizes += [[last_size,hidden_size]]
            last_size = hidden_size
            
    @staticmethod
    def coder(input_layer,layer_sizes,act_func):
        layer_i = input_layer
        for hidden_size in layer_sizes:
            w_i = tf.Variable(tf.random_normal(hidden_size))
            b_i = tf.Variable(tf.random_normal([hidden_size[-1]]))
            layer_i = act_func(tf.add(tf.matmul(layer_i, w_i),
                                           b_i))
        return layer_i
    
    def train(self,X,num_steps):
        assert X.max().max() == 1
        assert X.min().min().min() == 0
        batch_size = 1000
        display_step = 10
        for i in range(1, num_steps+1):
            batch_x = X.sample(batch_size)
            l,_ = self.sess.run([self.loss,self.optimizer], feed_dict={self.X: batch_x})
#            if i % display_step == 0 or i == 1:
#                print('Step %i: Minibatch Loss: %f' % (i, l))
    def get_feature(self,X):
        return self.sess.run(self.encoder_op,feed_dict={self.X: X})
    def load_model(self):
        tf.reset_default_graph()
        self.saver.restore(self.sess,self.model_path)
    def save_model(self):
        return self.saver.save(self.sess,self.model_path)


class autoencoder_add_classify(autoencoder):
    def __init__(self,input_size=269,hidden_layers = [256,128,64,32],
                 model_path= 'model/autoencoder_add_classify/autoencoder_add_classify.tfmodel',
                 alpha = 0.1):
        super(autoencoder_add_classify,self).__init__(input_size,hidden_layers,model_path)
        with self.graph.as_default():
            self.label_in = tf.placeholder("float", [None])
            self.label = tf.reshape(self.label_in,[-1,1])
            hidden_sizes = [[hidden_layers[-1],16],[16,8],[8,1]]
            self.lr_layer_out = self.coder(self.encoder_op,hidden_sizes,tf.nn.sigmoid)
            self.loss_lr = tf.reduce_mean(- self.label * tf.log(self.lr_layer_out) - (1 - self.label) * tf.log(1 - self.lr_layer_out))
            self.loss_all = alpha*self.loss_lr+(1-alpha)*self.loss
            #print (tf.trainable_variables())
            #print ([x for x in tf.trainable_variables()][-6:])
            self.optimizer_lr = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_lr,var_list = [x for x in tf.trainable_variables()][-6:])
            self.optimizer_all = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_all)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
    def predict(self,X):
        return self.sess.run(self.lr_layer_out,feed_dict={self.X: X})
    
    def get_loss(self,X,Y):
        return self.sess.run(self.loss_lr,feed_dict={self.X: X,self.label_in:Y})
    
    def train_lr(self,X,Y,num_steps):
        assert X.max().max() == 1
        assert X.min().min().min() == 0
        display_step = 1
        for i in range(1, num_steps+1):
            loss,lr_loss,_ = \
                self.sess.run([self.loss,self.loss_lr,self.optimizer_lr], feed_dict={self.X: X,self.label_in:Y})
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch LR Loss: %f' % (i, lr_loss))
                
    def train_all(self,X,Y,num_steps):
        assert X.max().max() == 1
        assert X.min().min().min() == 0
        my_X = X.copy()
        my_X['Y'] = Y
        batch_size = 1000
        display_step = 10
        for i in range(1, num_steps+1):
            batch = my_X.sample(batch_size)
            batch_x = batch.iloc[:,:-1]
            batch_y = batch.iloc[:,-1]
            loss,lr_loss,_ = \
                self.sess.run([self.loss,self.loss_lr,self.optimizer_all], feed_dict={self.X: batch_x,self.label_in:batch_y})
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, loss))
                print('Step %i: Minibatch LR Loss: %f' % (i, lr_loss))
