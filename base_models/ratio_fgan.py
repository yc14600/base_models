from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import scipy as sp
import six
import tensorflow as tf
from abc import ABC, abstractmethod
from utils.train_util import config_optimizer,plot
import matplotlib.pyplot as plt
from utils.model_util import *
from base_models.gans import fGAN,GAN


class Ratio_fGAN(fGAN):

    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.001,op_type='adam',clip=None,divergence='KL',lambda_constr=0.,*args,**kargs):
        self.lambda_constr = lambda_constr

        super(Ratio_fGAN,self).__init__(x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,\
                batch_norm,learning_rate,op_type,clip,divergence)

        return

    
    @staticmethod
    def get_idf_gf(divergence):
        if divergence == 'KL':
            def idf_gf(x):
                return tf.exp(x - 1.)

        elif divergence == 'rv_KL':
            def idf_gf(x):
                return tf.exp(x) 

        elif divergence == 'Pearson':
            def idf_gf(x):
                return 0.5 * x + 1.

        elif divergence == 'Hellinger':
            def idf_gf(x):
                return tf.exp(2.*x)

        elif divergence in ['Jensen_Shannon','GAN']:
            def idf_gf(x):
                return tf.exp(x)

        else:
            raise NotImplementedError('Divergence NOT supported.')

        return idf_gf

    @staticmethod
    def get_f_idf_gf(divergence):
        if divergence == 'KL':
            def f_idf_gf(v):
                return (v - 1.)*tf.exp(v - 1.)

        elif divergence == 'rv_KL':
            def f_idf_gf(v):
                return -v

        elif divergence == 'Pearson':
            def f_idf_gf(v):
                return 0.25 * tf.square(v)

        elif divergence == 'Hellinger':
            def f_idf_gf(v):
                return tf.square(tf.exp(v) - 1.)

        elif divergence == 'Jensen_Shannon':
            def f_idf_gf(v):
                return v * tf.exp(v) - (tf.exp(v)+1.)*tf.log(0.5*(tf.exp(v)+1.))

        elif divergence == 'GAN':
            def f_idf_gf(v):
                return v * tf.exp(v) - (tf.exp(v)+1.)*tf.log(tf.exp(v)+1.)

        else:
            raise NotImplementedError('Divergence NOT supported.')

        return f_idf_gf

    def set_loss(self):
        act_fn,conj_f = self.get_act_conj_fn(self.divergence)
        idf_gf = self.get_idf_gf(self.divergence)
        d_loss = -(tf.reduce_mean(act_fn(self.d_H[-1])) + tf.reduce_mean(-conj_f(act_fn(self.d_fake_H[-1]))))
        r_constr =  tf.square(tf.reduce_mean(idf_gf(self.d_fake_H[-1])) - 1.) + tf.square(tf.reduce_mean(1./idf_gf(self.d_H[-1])) - 1.)

        g_loss = tf.reduce_mean(-conj_f(act_fn(self.d_fake_H[-1])))

        return g_loss, d_loss + self.lambda_constr * r_constr

    def ratio(self,x):

        idf_gf = self.get_idf_gf(self.divergence)
        r = idf_gf(self.d_H[-1])

        feed_dict = {} if self.is_training is None else {self.is_training:False}
        feed_dict.update({self.x_ph:x})

        return self.sess.run(r, feed_dict=feed_dict)

    def print_log(self,e,feed_dict,*args,**kargs):
        super(Ratio_fGAN,self).print_log(e,feed_dict)
        fake_x = self.sess.run(self.g_H[-1],feed_dict)
        r_q= self.ratio(fake_x)
        r_p = self.ratio(feed_dict[self.x_ph])
        print('ratio constraints, r_q',np.mean(r_q),'r_p',np.mean(1./r_p))
        return 



