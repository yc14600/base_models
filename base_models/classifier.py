from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import scipy as sp
import numpy as np
import pandas as pd
import tensorflow as tf
import six
import sys
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from edward.models import Normal,TransformedDistribution,Gamma

from utils.train_util import *
from utils.model_util import *
from base_models.gans import GAN


class Classifier(object):
    def __init__(self,x_dim,y_dim,net_shape,batch_size,sess=None,epochs=100,conv=False,ac_fn=tf.nn.relu,batch_norm=False,training=None,\
                    reg=None,lambda_reg=0.0001,learning_rate=0.001,op_type='adam',decay=None,clip=None,scope='classifier',\
                    print_e=20,*args,**kargs):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.net_shape = net_shape
        print('net shape',net_shape)
        self.epochs = epochs
        self.scope = scope
        self.print_e = print_e
        self.conv = conv
        if conv:
            batch_norm = True
        
        if batch_norm and training is None:
            self.training = tf.placeholder(dtype=tf.bool,shape=[])
        else:
            self.training = training

        if sess is None:
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        else:
            self.sess = sess

        with tf.variable_scope(scope):
            if conv:
                self.x_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size]+x_dim,name='x_ph')
            else:
                self.x_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size,x_dim],name='x_ph')
            self.y_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size,y_dim],name='y_ph')
            self.W, self.B, self.H = GAN.define_d_net(self.x_ph,net_shape,reuse=False,conv=conv,ac_fn=ac_fn,\
                                                        batch_norm=batch_norm,training=self.training,reg=reg)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.H[-1], labels=self.y_ph))

        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        self.loss += reg_loss*lambda_reg

        self.train, self.var_lsit,self.opt = GAN.config_train(self.loss,scope,learning_rate=learning_rate,op_type=op_type,\
                                                        decay=decay,clip=clip)

    def fit(self,X,Y):
            
        with self.sess.as_default():
       
            tf.global_variables_initializer().run()
            
            ii = 0
            num_iters = int(np.ceil(X.shape[0]/self.batch_size))
            if self.training is None:
                feed_dict = {}
            else:
                feed_dict = {self.training:True}

            for e in range(self.epochs):
                for t in range(num_iters):
                    x_batch,y_batch,ii=get_next_batch(X,self.batch_size,ii,labels=Y) 
                    feed_dict.update({self.x_ph:x_batch,self.y_ph:y_batch})

                    __, loss = self.sess.run([self.train,self.loss],feed_dict=feed_dict)

                if (e+1)%self.print_e==0 or (e==0):
                    print(e+1,'loss',loss)

    
    def save_params(self):
        # save previous params
        self.prev_W,self.prev_B = self.sess.run([self.W,self.B]) 
        self.prev_H = GAN.restore_d_net(self.x_ph,self.prev_W,self.prev_B)


    def extract_feature(self,x,prev=False):
        ii = 0
        iters = int(np.ceil(x.shape[0]/self.batch_size))
        if self.conv:
            rlt = np.zeros([x.shape[0],self.net_shape[-1][-2]])
        else:
            rlt = np.zeros([x.shape[0],self.net_shape[-1]])
        if self.training is None:
            feed_dict = {}
        else:
            feed_dict = {self.training:False}
        for i in range(iters):
            start = ii
            x_batch,_,ii = get_next_batch(x,self.batch_size,ii)
            end = ii if ii < x.shape[0] and ii > start else x.shape[0]
            feed_dict.update({self.x_ph:x_batch})
            if prev:
                rlt[start:end] = self.sess.run(self.prev_H[-2],feed_dict)[:end-start]
            else:
                rlt[start:end] = self.sess.run(self.H[-2],feed_dict)[:end-start]
        return rlt


