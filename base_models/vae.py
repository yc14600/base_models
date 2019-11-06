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

from edward.models import Normal,TransformedDistribution,Gamma

from utils.train_util import *
from utils.model_util import *
from hsvi import Hierarchy_SVI
from base_models.gans import GAN

from tensorflow.contrib import slim


ds = tf.contrib.distributions






class VAE(object):

    def __init__(self,x_dim,z_dim,batch_size,e_net_shape,d_net_shape,sess=None,train_size=10000,noise_std=0.1,\
                    epochs=50, print_e=20, learning_rate=0.001,conv=False,scope='vae',reg=None,lamb_reg=0.001,\
                    prior_std=1.,bayes=False,ac_fn=tf.nn.relu,output_ac=tf.nn.sigmoid,*args,**kargs):

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.epochs = epochs
        self.print_e = print_e
        self.conv = conv
        self.scope = scope
        self.reg = reg
        self.lamb_reg = lamb_reg
        self.prior_std = prior_std
        self.bayes = bayes
        self.ac_fn = ac_fn
        self.output_ac = output_ac
        if sess is None:
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        else:
            self.sess = sess

        self.define_model(x_dim,z_dim,batch_size,e_net_shape,d_net_shape,noise_std,learning_rate)

        self.config_inference(self.opt,train_size)


    def define_model(self,x_dim,z_dim,batch_size,e_net_shape,d_net_shape,noise_std,learning_rate,*args,**kargs):

        x_dim = x_dim if self.conv else [x_dim]
        self.x_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size]+x_dim,name='x_ph')

        self.pz = Normal(loc=tf.zeros([z_dim]),scale=tf.ones([z_dim])*self.prior_std,sample_shape=[batch_size])
        #self.noise_p_prior = Gamma(1.,.01)

        if self.bayes and not self.conv:
            e_net_shape = x_dim + e_net_shape + [z_dim]
            d_net_shape = [z_dim] + d_net_shape + x_dim

        with tf.variable_scope(self.scope):
        
            self.z_mu, self.z_sigma = self.encoder(self.x_ph,e_net_shape,scope='encoder')
            self.qz = Normal(loc=self.z_mu,scale=self.z_sigma)

            self.rec_x = self.decoder(self.qz,d_net_shape,scope='decoder')

            #self.q_noise_prc = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([1])), \
            #                scale=tf.ones([1])),bijector=ds.bijectors.Exp())

            self.qx = Normal(loc=self.rec_x,scale=noise_std)    
            self.opt = config_optimizer(learning_rate,'vae_step','adam')
    

    def config_inference(self,opt,train_size):
        latent_vars={self.scope:{self.pz:self.qz}}
        prior = Normal(loc=0.,scale=1e-2)
        if self.bayes:
            for qw in self.eW+self.eB+self.dW+self.dB:
                latent_vars[self.scope].update({prior:qw})
            if self.conv:
                for qw in self.conv_eW+self.conv_dW:
                    latent_vars[self.scope].update({prior:qw})

        self.inference = Hierarchy_SVI(latent_vars=latent_vars,data={self.scope:{self.qx:self.x_ph}})

        if self.reg is not None:
            reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses()) * self.lamb_reg
            self.inference.initialize(vi_types={self.scope:'KLqp_analytic'},optimizer={self.scope:opt},train_size=train_size,extra_loss={self.scope:reg_loss})
        else:
            self.inference.initialize(vi_types={self.scope:'KLqp_analytic'},optimizer={self.scope:opt},train_size=train_size)

    
    def encoder(self,x,net_shape,scope='encoder',reuse=False):
        if self.bayes:
            if self.conv:
                conv_net_shape = net_shape[0]
                net_shape = net_shape[1]
                self.conv_eW,self.conv_parm_var,self.conv_eh = build_bayes_conv_net(x,self.batch_size,conv_net_shape)
                x = self.conv_eh 

            self.eW,self.eB,self.eH,_,_,_,self.parm_var = build_nets(net_shape,x,bayes=True,ac_fn=self.ac_fn,output_ac=self.output_ac,bayes_output=False)
            
            with tf.variable_scope(scope,reuse=reuse):
                    self.sigma_w,self.sigma_b,z_sigma,_,sigma_var = build_dense_layer(self.eH[-2],len(net_shape),net_shape[-2],self.z_dim,\
                                                                                    ac_fn=tf.nn.softplus,reg=self.reg,bayes=True)
                    self.parm_var.update(sigma_var)    
                
        else:
            if self.conv:
                self.eW,self.eB,self.eH = GAN.define_d_net(x,net_shape,reuse=reuse,conv=self.conv,scope=scope,ac_fn=tf.nn.relu,reg=self.reg)  
                with tf.variable_scope(scope,reuse=reuse):
                    #print('shape check',self.eH[-1].shape,net_shape[-1][-2])
                    self.sigma_w,self.sigma_b,z_sigma = build_dense_layer(tf.layers.flatten(self.eH[-2]),len(net_shape[-1])+1,net_shape[-1][-2],self.z_dim,\
                                                                        ac_fn=tf.nn.softplus,reg=self.reg)
            else:
                self.eW,self.eB,self.eH = GAN.define_d_net(x,[self.x_dim]+net_shape+[self.z_dim],reuse=reuse,conv=self.conv,\
                                                            scope=scope,ac_fn=self.ac_fn,reg=self.reg)  
                with tf.variable_scope(scope,reuse=reuse):
                    self.sigma_w,self.sigma_b,z_sigma = build_dense_layer(self.eH[-2],len(net_shape)+2,net_shape[-1],self.z_dim,\
                                                                            ac_fn=tf.nn.softplus,reg=self.reg)    

        return self.eH[-1],z_sigma



    def encode(self,x,random=True):
        ii = 0
        iters = int(np.ceil(x.shape[0]/self.batch_size))
        rlt = np.zeros([x.shape[0],self.z_dim])
        value = self.qz if random else self.z_mu
        for i in range(iters):
            start = ii
            x_batch,_,ii = get_next_batch(x,self.batch_size,ii)
            end = ii if ii < x.shape[0] and ii > start else x.shape[0]
            feed_dict = {self.x_ph:x_batch}
            rlt[start:end] = self.sess.run(value,feed_dict)[:end-start]
        return rlt

    
    def decoder(self,z,net_shape,scope='decoder',reuse=False):
        if self.bayes:
            if self.conv:
                conv_net_shape = net_shape[1]
                net_shape = net_shape[0]
            self.dW,self.dB,self.dH,_,_,_,d_parm_var = build_nets(net_shape,z,bayes=True,ac_fn=self.ac_fn,\
                                                                    output_ac=tf.nn.sigmoid,bayes_output=False)
            self.parm_var.update(d_parm_var)
            if self.conv:
                self.conv_dW,conv_parm_var,conv_dh = build_bayes_conv_net(self.dH[-1],self.batch_size,conv_net_shape)
                self.dH.append(conv_dh)
                self.conv_parm_var.update(conv_parm_var)

        else:
            if self.conv:
                self.dW,self.dB,self.dH = GAN.define_g_net(z,net_shape,reuse=reuse,conv=self.conv,scope=scope,reg=self.reg) 
            else:
                self.dW,self.dB,self.dH = GAN.define_g_net(z,[self.z_dim]+net_shape+[self.x_dim],reuse=reuse,conv=self.conv,scope=scope,reg=self.reg) 
        return self.dH[-1]   

    
    def reconstruct(self,x):
        ii = 0
        iters = int(np.ceil(x.shape[0]/self.batch_size))
        rlt = np.zeros([x.shape[0],self.x_dim])
        for i in range(iters):
            start = ii
            x_batch,_,ii = get_next_batch(x,self.batch_size,ii,repeat=False)
            end = ii if ii < x.shape[0] and ii > start else x.shape[0]
            feed_dict = {self.x_ph:x_batch}
            print('start',start,'end',end)
            rlt[start:end] = self.sess.run(self.rec_x,feed_dict)[:end-start]
        return rlt

    
    def train(self,X,warm_start=False,standalone=True):
        
        with self.sess.as_default():
            if not warm_start:
                if standalone:
                    tf.global_variables_initializer().run()
                else:
                    reinitialize_scope(self.scope,self.sess)

            ii = 0
            num_iters = int(np.ceil(X.shape[0]/self.batch_size))

            for e in range(self.epochs):
                for t in range(num_iters):
                    x_batch,_,ii=get_next_batch(X,self.batch_size,ii) 
                    feed_dict = {self.x_ph:x_batch}

                    info_dict = self.inference.update(feed_dict=feed_dict,scope=self.scope)

                if (e+1)%self.print_e==0 or (e==0):
                    print(e+1,'loss',info_dict['loss'])


    def save_params(self):
        sess = self.sess
        if self.bayes:
            self.prev_eW, self.prev_eB, self.prev_dW, self.prev_dB = [], [], [], []
            for w,b in zip(self.eW,self.eB):
                self.prev_eW.append(Normal(loc=sess.run(w.loc),scale=sess.run(w.scale)))
                self.prev_eB.append(Normal(loc=sess.run(b.loc),scale=sess.run(b.scale)))
            
            self.prev_eH = forward_mean_nets(self.prev_eW,self.prev_eB,self.x_ph,ac_fn=self.ac_fn,sess=self.sess,output_ac=self.output_ac)
            self.prev_sigma_w = Normal(loc=sess.run(self.sigma_w.loc),scale=sess.run(self.sigma_w.scale))
            self.prev_sigma_b = Normal(loc=sess.run(self.sigma_b.loc),scale=sess.run(self.sigma_b.scale))
            
            self.prev_z_sigma = restore_dense_layer(self.prev_eH[-2],len(self.prev_eW)+1,self.prev_sigma_w,self.prev_sigma_b,ac_fn=tf.nn.softplus,bayes=True)          
            self.prev_qz = Normal(loc=self.prev_eH[-1],scale=self.prev_z_sigma)

            for w,b in zip(self.dW,self.dB):
                self.prev_dW.append(Normal(loc=sess.run(w.loc),scale=sess.run(w.scale)))
                self.prev_dB.append(Normal(loc=sess.run(b.loc),scale=sess.run(b.scale)))

            self.prev_dH = forward_mean_nets(self.prev_dW,self.prev_dB,self.prev_qz,ac_fn=self.ac_fn,sess=self.sess,output_ac=tf.nn.sigmoid)

        else:
            # save encoder params
            self.prev_eW,self.prev_eB = self.sess.run([self.eW,self.eB]) 
            self.prev_eH = GAN.restore_d_net(self.x_ph,self.prev_eW,self.prev_eB) 
            self.prev_sigma_w,self.prev_sigma_b = self.sess.run([self.sigma_w,self.sigma_b])
        
            self.prev_z_sigma = restore_dense_layer(self.prev_eH[-2],len(self.prev_eW)+1,self.prev_sigma_w,self.prev_sigma_b,ac_fn=tf.nn.softplus)          
            self.prev_qz = Normal(loc=self.prev_eH[-1],scale=self.prev_z_sigma)
            # save decoder params
            self.prev_dW,self.prev_dB = self.sess.run([self.dW,self.dB])
            self.prev_dH = GAN.restore_g_net(self.prev_qz,self.prev_dW,self.prev_dB)
        
        self.prev_qx = Normal(loc=self.prev_dH[-1],scale=self.noise_std)

    
    def prev_encode(self,x,random=True):
        ii = 0
        iters = int(np.ceil(x.shape[0]/self.batch_size))
        rlt = np.zeros([x.shape[0],self.z_dim])
        value = self.prev_qz if random else self.prev_eH[-1]
        for i in range(iters):
            start = ii
            x_batch,_,ii = get_next_batch(x,self.batch_size,ii)
            end = ii if ii < x.shape[0] and ii > start else x.shape[0]
            feed_dict = {self.x_ph:x_batch}
            rlt[start:end] = self.sess.run(value,feed_dict)[:end-start]
        return rlt

    
    def prev_reconstruct(self,x):
        ii = 0
        iters = int(np.ceil(x.shape[0]/self.batch_size))
        rlt = np.zeros([x.shape[0],self.x_dim])
        for i in range(iters):
            start = ii
            x_batch,_,ii = get_next_batch(x,self.batch_size,ii)
            end = ii if ii < x.shape[0] and ii > start else x.shape[0]
            feed_dict = {self.x_ph:x_batch}
            rlt[start:end] = self.sess.run(self.prev_qx,feed_dict)[:end-start]
        return rlt


class Discriminant_VAE(VAE):

    def __init__(self,x_dim,z_dim,batch_size,e_net_shape,d_net_shape,train_size=10000,noise_std=0.1,\
                    epochs=50, print_e=20, learning_rate=0.001,conv=False,scope='dvae',\
                    reg=None,lamb=1.,lamb_reg=0.0001,prior_std=.1,divergence='KL',*args,**kargs):
        
        #self.x_nu_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,x_dim],name='x_nu_ph')
        self.x_de_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,x_dim],name='x_de_ph')
        self.lamb = lamb
        self.divergence = divergence
        super(Discriminant_VAE,self).__init__(x_dim,z_dim,batch_size,e_net_shape,d_net_shape,train_size,noise_std,\
                    epochs, print_e,learning_rate,conv,scope,reg,lamb_reg,prior_std)


    def define_model(self,x_dim,z_dim,batch_size,e_net_shape,d_net_shape,noise_std,learning_rate,*args,**kargs):
        super(Discriminant_VAE,self).define_model(x_dim,z_dim,batch_size,e_net_shape,d_net_shape,noise_std,learning_rate,*args,**kargs)
        with tf.variable_scope(self.scope):
            self.z_de_mu, self.z_de_sigma = self.encoder(self.x_de_ph,e_net_shape,reuse=True,scope='encoder')
            self.qz_de = Normal(loc=self.z_de_mu,scale=self.z_de_sigma)
            self.rec_x_de = self.decoder(self.qz_de,d_net_shape,reuse=True,scope='decoder')
            self.qx_de = Normal(loc=self.rec_x_de,scale=noise_std)



    def normal_pdf(self,z,mu,sigma):
        zs = tf.expand_dims(z,axis=1)
        log_pdf = -tf.divide(tf.square(zs-mu),2*tf.square(sigma))-tf.log(sigma*tf.sqrt(np.pi*2.))
        log_pdf = tf.reduce_sum(log_pdf,axis=2)
        log_pdf = tf.clip_by_value(log_pdf,-20.,20.)
        return tf.exp(log_pdf)

    
    def config_inference(self,opt,train_size):
        if self.divergence == 'KL':
            kl = tf.reduce_mean(tf.log(tf.reduce_mean(self.normal_pdf(tf.stop_gradient(self.qz),self.z_mu,self.z_sigma),axis=1)) \
                    - tf.log(tf.reduce_mean(self.normal_pdf(tf.stop_gradient(self.qz),self.z_de_mu,self.z_de_sigma),axis=1)))
            rvkl = tf.reduce_mean(tf.log(tf.reduce_mean(self.normal_pdf(tf.stop_gradient(self.qz_de),self.z_de_mu,self.z_de_sigma),axis=1)) \
                    - tf.log(tf.reduce_mean(self.normal_pdf(tf.stop_gradient(self.qz_de),self.z_mu,self.z_sigma),axis=1)))

            self.ds_loss = -self.lamb * 0.5 * (kl+rvkl)

        elif self.divergence == 'JS':
            p_nu_z = tf.reduce_mean(self.normal_pdf(tf.stop_gradient(self.qz),self.z_mu,self.z_sigma),axis=1)
            p_de_z = tf.reduce_mean(self.normal_pdf(tf.stop_gradient(self.qz_de),self.z_de_mu,self.z_de_sigma),axis=1)
           
            kl_pm = tf.reduce_mean(tf.log(p_nu_z) - tf.log(0.5*(p_nu_z + \
                        tf.reduce_mean(self.normal_pdf(tf.stop_gradient(self.qz),self.z_de_mu,self.z_de_sigma),axis=1))))
                    
            kl_qm = tf.reduce_mean(tf.log(p_de_z) - tf.log(0.5*(p_de_z + \
                        tf.reduce_mean(self.normal_pdf(tf.stop_gradient(self.qz_de),self.z_mu,self.z_sigma),axis=1))))
            self.ds_loss = -self.lamb * 0.5 * (kl_pm+kl_qm)      

        
        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())*self.lamb_reg

        self.inference = Hierarchy_SVI(latent_vars={self.scope:{self.pz:self.qz}},data={self.scope:{self.qx:self.x_ph}})
        if self.lamb > 0.:
            self.inference.initialize(optimizer={self.scope:opt},train_size=train_size,extra_loss={self.scope:self.ds_loss+reg_loss})
        else:
            self.inference.initialize(optimizer={self.scope:opt},train_size=train_size,extra_loss={self.scope:reg_loss})

    
    def log_ratio(self,x_nu,x_de):
        logr = tf.log(tf.reduce_mean(self.normal_pdf(self.qz,self.z_mu,self.z_sigma),axis=1)) \
                    - tf.log(tf.reduce_mean(self.normal_pdf(self.qz,self.z_de_mu,self.z_de_sigma),axis=1))
        batch_size = self.batch_size
        ii = 0
        iters = int(np.ceil(x_nu.shape[0]/batch_size))
        rlt = np.zeros(x_nu.shape[0])
        for i in range(iters):
            start = ii
            ii_bk = ii
            nu_batch,_,ii = get_next_batch(x_nu,batch_size,ii)
            de_batch,_,__ = get_next_batch(x_de,batch_size,ii_bk)
            end = ii if ii < x_nu.shape[0] and ii > start else x_nu.shape[0]
            feed_dict = {self.x_ph:nu_batch,self.x_de_ph:de_batch}
            rlt[start:end] = self.sess.run(logr,feed_dict)
        return rlt


    def ratio(self,x_nu,x_de):
        return np.exp(self.log_ratio(x_nu,x_de))


    def train(self,X_nu,X_de):
        
        with self.sess.as_default():
       
            tf.global_variables_initializer().run()
            
            ii = 0
            num_iters = int(np.ceil(X_de.shape[0]/self.batch_size))

            for e in range(self.epochs):
                for t in range(num_iters):
                    ii_bk = ii
                    x_batch,_,ii = get_next_batch(X_nu,self.batch_size,ii) 
                    x_de_batch,_,__ = get_next_batch(X_de,self.batch_size,ii_bk) 
                    #x_de_batch,_,__ = get_next_batch(X_de,self.batch_size,ii_bk) 
                    feed_dict = {self.x_ph:x_batch,self.x_de_ph:x_de_batch}

                    info_dict = self.inference.update(feed_dict=feed_dict,scope=self.scope)
                    
                    dloss = self.sess.run(self.ds_loss,feed_dict=feed_dict)
                    #print('dloss',dloss)
                    if np.isnan(dloss):
                        z_mu,z_sigma,h = self.sess.run([self.z_mu,self.z_sigma,self.eH[-2]],feed_dict)
                        print('z_mu,min,max',np.min(z_mu),np.max(z_mu))
                        print('z_sigma,min,max',np.min(z_sigma),np.max(z_sigma))
                        print('z_mu,z_sigma is nan',np.isnan(z_mu).any(),np.isnan(z_sigma).any(),np.isnan(h).any())
                        eW=self.sess.run(self.dW)
                        for w in eW:
                            print('check W',np.isnan(w).any())
                            eW=self.sess.run(self.dW)
                            for w in eW:
                                print('min max W',np.min(w),np.max(w)) 
                        raise TypeError(e,t,'dloss is nan')

                if (e+1)%self.print_e==0 or (e==0):
                    dloss = self.sess.run(self.ds_loss,feed_dict=feed_dict)
                    print(e+1,'loss',info_dict['loss'],'dloss',dloss)
