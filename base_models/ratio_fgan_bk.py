from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import scipy as sp
import six
import tensorflow as tf
from abc import ABC, abstractmethod
from utils.train_util import config_optimizer,get_var_list,plot
import matplotlib.pyplot as plt
from utils.model_util import *
from base_models.gan import GAN
from dre.estimators import LogLinear_Estimator,KL_Loglinear_Estimator


class Ratio_fGAN(GAN):

    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                        batch_norm=False,learning_rate=0.001,op_type='adam',divergence='rv_KL',\
                        lambda_reg=1.,bayes=False,constr=False,*args,**kargs):

        self.divergence = divergence
        if divergence != 'rv_KL':
            raise NotImplementedError('only support rv_KL divergence currently.')
        super(Ratio_fGAN,self).__init__(x_ph,g_net_shape,d_net_shape,batch_size,conv=conv,sess=sess,\
                                        ac_fn=ac_fn,batch_norm=batch_norm,learning_rate=learning_rate,\
                                        op_type=op_type,lambda_reg=lambda_reg,bayes=bayes,constr=constr,*args,**kargs)


    
    def define_model(self,x_ph,g_net_shape,d_net_shape,batch_size,conv,ac_fn,batch_norm,\
                        learning_rate,op_type,lambda_reg=1.,bayes=False,constr=False,*args,**kargs):
        
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[],name='batch_norm') if batch_norm else None
        self.x_ph = x_ph # true data
        k = g_net_shape[0][0] if conv else g_net_shape[0]       
        self.e_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,k],name='noise')     

        self.g_W,self.g_B,self.g_H = self.define_g_net(self.e_ph,g_net_shape,reuse=False,conv=conv,\
                                                        ac_fn=ac_fn,batch_norm=batch_norm,training=self.is_training)

        self.ratio_estimator = self.define_estimator(d_net_shape,x_ph,self.g_H[-1],conv=conv,\
                                                        batch_norm=batch_norm,ac_fn=ac_fn,scope='ratio',\
                                                        lambda_reg=lambda_reg,bayes=bayes,constr=constr) 

        
        
        self.g_loss,self.d_loss = self.set_loss()
        self.g_train,self.g_var_list = self.config_train(self.g_loss,scope='generator',learning_rate=learning_rate*10)
        self.d_train,self.d_var_list = self.config_train(self.d_loss,scope='ratio',learning_rate=learning_rate)
        return

    
    @abstractmethod
    def set_loss(self,*args,**kargs):
        pass

    @abstractmethod
    def define_estimator(self,net_shape,nu_ph,de_ph,coef=None,conv=False,batch_norm=False,\
                            ac_fn=tf.nn.relu,reg=None,scope='ratio',batch_train=False,\
                            lambda_reg=1.,bayes=False,constr=False,*args,**kargs):
        pass

    # overwrite the function defined in GAN, output desity ratio
    def discriminator(self,x,x_de=None,*args,**kargs):
        feed_dict = {} if self.is_training is None else {self.is_training:False}

        if x_de is None:
            feed_dict.update({self.ratio_estimator.nu_ph: x})
        else:
            feed_dict.update({self.ratio_estimator.nu_ph: x, self.ratio_estimator.de_ph:x_de})
        
        return self.sess.run(self.ratio_estimator.ratio(x,x_de),feed_dict=feed_dict)


    def training(self,X,Y,d_obj,g_obj,batch_size,epoch,disc_iter=1,vis=False,result_path='vis_results/',warm_start=False):
        with self.sess.as_default():
            if not warm_start:
                tf.global_variables_initializer().run()
          
            feed_dict = {} if self.is_training is None else {self.is_training:True,self.ratio_estimator.is_training:True}
            
            num_iters = int(np.ceil(X.shape[0]/batch_size))
            for e in range(epoch):
                ii = 0
                for i in range(num_iters):
                    x_batch,y_batch,ii = get_next_batch(X,batch_size,ii,labels=Y)
                    e_batch = np.random.uniform(-1.,1.,size=(batch_size,self.g_W[0].shape[0].value)).astype(np.float32)
                    feed_dict.update({self.x_ph:x_batch,self.e_ph:e_batch})
                    
                    _,d_loss = self.sess.run([d_obj,self.d_loss],feed_dict=feed_dict)
                    #d_loss = self.sess.run(self.d_loss,feed_dict=feed_dict)                    
                    #print('d loss',d_loss)

                    if (i+1) % disc_iter == 0:
                        _,g_loss = self.sess.run([g_obj,self.g_loss],feed_dict=feed_dict)
                        #g_loss = self.sess.run(self.g_loss,feed_dict=feed_dict)                        
                        #print('g loss',g_loss)

                print('epoch',e,'d loss',d_loss,'g loss',g_loss)
                if vis and (e+1)%1==0:
                    e_samples = np.random.uniform(-1.,1.,size=(batch_size,self.e_ph.shape[1].value)).astype(np.float32)
                    x_samples = self.generator(e_samples)
                    ng = int(np.sqrt(batch_size))
                    fig = plot(x_samples[:ng*ng],shape=[ng,ng])
                    fig.savefig(result_path+'e'+str(e+1)+'.pdf')
                    plt.close()



class LogLinear_Ratio_fGAN(Ratio_fGAN):

    def define_estimator(self,net_shape,nu_ph,de_ph,coef=None,conv=False,batch_norm=False,\
                            ac_fn=tf.nn.relu,reg=None,scope='ratio',batch_train=False,\
                            lambda_reg=1.,bayes=False,constr=False,*args,**kargs):
        if self.divergence == 'rv_KL':
            estimator = fGAN_Ratio_Estimator(net_shape,nu_ph,de_ph,coef=coef,conv=conv,batch_norm=batch_norm,\
                                                ac_fn=ac_fn,reg=reg,scope=scope,batch_train=batch_train,\
                                                lambda_reg=lambda_reg,bayes=bayes,constr=constr)
        else:
            raise NotImplementedError('only support rv_KL currently.')

        return estimator

    def set_loss(self,*args,**kargs):

        d_loss = self.ratio_estimator.loss
        
        if self.divergence == 'rv_KL':
            print('Optimize generatior by reverse KL divergence')
            g_loss = tf.log(tf.reduce_mean(tf.exp(self.ratio_estimator.de_r))) - tf.reduce_mean(self.ratio_estimator.de_r)

        else:
            raise NotImplementedError

        return g_loss, d_loss


    def training(self,X,Y,batch_size,epoch,disc_iter=1,vis=False,result_path='vis_results/',warm_start=False):
        d_obj = self.d_train#[self.d_train,self.clip_D]
        g_obj = self.g_train
        super(LogLinear_Ratio_fGAN,self).training(X,Y,d_obj,g_obj,batch_size=batch_size,epoch=epoch,disc_iter=disc_iter,\
                                    vis=vis,result_path=result_path,warm_start=warm_start)

        return

class fGAN_Ratio_Estimator(LogLinear_Estimator):

    def set_loss(self,divergence='rv_KL'):
        if divergence == 'rv_KL':
            loss = tf.reduce_mean(tf.reduce_mean(tf.exp(self.de_r)) / tf.exp(self.nu_r)) - \
                    tf.reduce_mean(1. + tf.log(tf.reduce_mean(tf.exp(self.de_r))) - \
                                        tf.reduce_mean(self.de_r) )
        return loss

