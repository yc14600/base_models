from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import scipy as sp
import six
import tensorflow as tf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from utils.train_util import config_optimizer,get_var_list,plot,concat_cond_data,shuffle_data
from utils.model_util import *


class GAN(ABC):
    
        
    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.0002,op_type='adam',clip=None,reg=None,g_penalty=False,\
                gamma0=1.,alpha=0.01,pooling=False,strides={},*args,**kargs):
        print('x ph',x_ph,'batch size',batch_size)
        self.conv = conv
        print('conv', conv)
        self.batch_norm = batch_norm
        self.pooling = pooling
        print('pooling', pooling)
        self.ac_fn = ac_fn
        self.g_net_shape = g_net_shape
        self.d_net_shape = d_net_shape
        self.batch_size = batch_size
        self.g_penalty = g_penalty
        print('g penalty',g_penalty)
        self.strides = strides
        if g_penalty:
            print('enable discriminator regularizer')
            self.gamma0 = gamma0
            self.alpha = alpha
            if alpha > 0.:
                print('use annealing')
                self.gamma = tf.placeholder(dtype=tf.float32,shape=[], name='d_reg_gamma')
            else:
                self.gamma = gamma0
      
        self.define_model(x_ph,g_net_shape,d_net_shape,batch_size,conv,ac_fn,batch_norm,learning_rate,\
                            op_type,clip,reg,pooling,strides,*args,**kargs)
        if sess is None:
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        else:
            self.sess = sess

    
    def define_model(self,x_ph,g_net_shape,d_net_shape,batch_size,conv,ac_fn,batch_norm,learning_rate,\
                        op_type,clip=None,reg=None,pooling=False,strides={},*args,**kargs):
        
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[]) if batch_norm else None
        self.x_ph = x_ph # true data
        k = g_net_shape[0][0] if conv else g_net_shape[0]   
        print('check e ph',batch_size,'k',k)    
        self.e_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,k])      
        print('strides',strides)
        g_strides, d_strides = strides.get('generator',[]), strides.get('discriminator',[])
        self.g_W,self.g_B,self.g_H = self.define_g_net(self.e_ph,g_net_shape,reuse=False,conv=conv,ac_fn=ac_fn,\
                                                        batch_norm=batch_norm,training=self.is_training,\
                                                        reg=reg,pooling=pooling,strides=g_strides)
        self.d_W,self.d_B,self.d_H = self.define_d_net(self.x_ph,d_net_shape,reuse=False,conv=conv,ac_fn=ac_fn,\
                                                        batch_norm=batch_norm,training=self.is_training,\
                                                        reg=reg,pooling=pooling,strides=d_strides)
        _,__,self.d_fake_H = self.define_d_net(self.g_H[-1],d_net_shape,reuse=True,conv=conv,ac_fn=ac_fn,\
                                                batch_norm=batch_norm,training=self.is_training,\
                                                reg=reg,pooling=pooling,strides=d_strides)
        
        self.g_loss,self.d_loss = self.set_loss()
        self.g_train,self.g_var_list,self.g_opt = self.config_train(self.g_loss,scope='generator',learning_rate=learning_rate*5,clip=clip)
        self.d_train,self.d_var_list,self.d_opt = self.config_train(self.d_loss,scope='discriminator',learning_rate=learning_rate,clip=clip)
        
        return


    @staticmethod
    def define_g_net(e,net_shape,reuse,conv=False,ac_fn=tf.nn.relu,batch_norm=False,training=None,reg=None,\
                    output_ac=tf.nn.sigmoid,scope = 'generator',pooling=False,strides=[]):
        W,B,H = [],[],[]
        h = e
        dense_net_shape = net_shape[0] if conv else net_shape
        print('define generator')
        # deconv layers must after dense layers
        with tf.variable_scope(scope,reuse=reuse):
            for l in range(len(dense_net_shape)-1):
                l_batch_norm = batch_norm
                # if no conv layer and it's output layer
                if l == len(dense_net_shape) - 2 and not conv:
                        ac_fn = output_ac
                        l_batch_norm = False
                w, b, h = build_dense_layer(h,l,dense_net_shape[l],dense_net_shape[l+1],initialization={'w':tf.random_normal_initializer(stddev=0.02),'b':tf.constant_initializer(0.0)},\
                                            ac_fn=ac_fn,batch_norm=l_batch_norm,training=training,scope=scope,reg=reg)
                W.append(w)
                B.append(b)  
                #print(l,w,b)
                H.append(h)    
            if conv:
                h = tf.reshape(h,[-1]+net_shape[1][0]) # net_shape[1][] is a list of input and filter shape of dconv2d
                for l in range(1,len(net_shape[1])):
                    l_batch_norm = False
                    # set batch norm every 2 layers, starting from first layer
                    if batch_norm and l%2 == 0:
                        l_batch_norm = True
                    # output layer
                    if l == len(net_shape[1])-1:
                        ac_fn = output_ac
                        l_batch_norm = False

                    filter_shape = net_shape[1][l]
                    print('input shape',h.shape)
                    strd = strides[l-1] if strides else [1,2,2,1]
                    print('strides',strd)
                    ow, oh = int(h.shape[1].value*strd[1]), int(h.shape[2].value*strd[2]) #stride (2,2)
                    output_shape = [h.shape[0].value,ow,oh,filter_shape[2]]
                    print('output shape',output_shape,'filter shape',filter_shape)
                    w,b,h = build_conv_bn_acfn(h,l,filter_shape,strides=strd,initialization={'w':tf.truncated_normal_initializer(stddev=0.02),'b':tf.constant_initializer(0.0)},\
                                                ac_fn=ac_fn,deconv=True,output_shape=output_shape,batch_norm=l_batch_norm,training=training,scope=scope,reg=reg)
                    if pooling:
                        h = tf.nn.max_pool(value=h,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
                    
                    W.append(w)
                    B.append(b)
                    H.append(h)            
            print('generator output shape',H[-1].shape)
        return W,B,H

    @staticmethod
    def define_d_net(x,net_shape,reuse,conv=False,ac_fn=tf.nn.relu,batch_norm=False,training=None,reg=None,\
                        scope = 'discriminator',pooling=False,strides=[]):
        W,B,H = [],[],[]        
        h = x
        dense_net_shape = net_shape[1] if conv else net_shape
        print('define discriminator')
        # conv layers must before dense layers
        with tf.variable_scope(scope,reuse=reuse):
            if conv:
                for l in range(len(net_shape[0])):                    
                    filter_shape = net_shape[0][l]
                    print('filter shape',filter_shape)
                    strd = strides[l] if strides else [1,2,2,1]
                    print('strides',strd)
                    # set batch norm every 2 layers, starting from second layer
                    l_batch_norm = False
                    if batch_norm and (l+1)%2 == 0:
                        l_batch_norm = True
                    w,b,h = build_conv_bn_acfn(h,l,filter_shape,strides=strd,initialization={'w':tf.random_normal_initializer(stddev=0.02),'b':tf.constant_initializer(0.0)},\
                                                ac_fn=ac_fn,batch_norm=l_batch_norm,training=training,scope=scope,reg=reg)
                    print('output shape',h.shape)
                    #if pooling:
                    #    h = tf.nn.max_pool(value=h,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

                    W.append(w)
                    B.append(b)
                    H.append(h)  
                h = tf.reshape(h,[h.shape[0].value,-1])
            for l in range(len(dense_net_shape)-2):
                w, b, h = build_dense_layer(h,l,dense_net_shape[l],dense_net_shape[l+1],initialization={'w':tf.truncated_normal_initializer(stddev=0.02),'b':tf.constant_initializer(0.0)},\
                                            ac_fn=ac_fn,batch_norm=batch_norm,training=training,scope=scope,reg=reg)
                W.append(w)
                B.append(b)
                H.append(h)  
                print(l,w,b)

            # define output layer without activation function         
            w,b,h = build_dense_layer(h,l+1,dense_net_shape[-2],dense_net_shape[-1],ac_fn=None,batch_norm=False)
            W.append(w)
            B.append(b)
            H.append(h)
            print(l+1,w,b)
        
        return W,B,H


    @staticmethod
    def restore_d_net(x,W,B,conv_L=0,ac_fn=tf.nn.relu,batch_norm=False,training=None,scope='',strides=[]):
        h = x
        H = []
        # including conv layers
        if conv_L > 0:
            conv_W, conv_B = W[:conv_L], B[:conv_L]
            for l in range(conv_L):
                l_batch_norm = False
                if batch_norm and (l+1)%2 == 0:
                    l_batch_norm = True
                strd = strides[l] if strides else [1,2,2,1]
                h = restore_conv_layer(h,l,conv_W[l],conv_B[l],strides=strd,ac_fn=ac_fn,\
                                        batch_norm=l_batch_norm,training=training,scope=scope)
                H.append(h)
            h = tf.reshape(h,[h.shape[0].value,-1])

        # restore dense layers
        for l in range(conv_L,len(W)-1):
            print('restore layer {}: h {}, w {}, b {}'.format(l,h.shape,W[l].shape,B[l].shape))
            h = restore_dense_layer(h,l,W[l],B[l],ac_fn=ac_fn,batch_norm=batch_norm,training=training,scope=scope)
            H.append(h)
        h = restore_dense_layer(h,len(W)-1,W[-1],B[-1],ac_fn=None,batch_norm=False)
        H.append(h)
        return H
 

    @staticmethod
    def restore_g_net(e,W,B,conv_L=0,ac_fn=tf.nn.relu,batch_norm=False,training=None,scope='',strides=[]):
        h = e
        H = []
        print('e shape',e.shape)
        for w,b in zip(W,B):
            print('w shape',w.shape,'b shape',b.shape)
        for l,(w,b) in enumerate(zip(W[:len(W)-conv_L],B[:len(B)-conv_L])):
            
            l_batch_norm = batch_norm

            if l == len(W) - 2 and conv_L > 0:
                ac_fn = tf.nn.sigmoid
                l_batch_norm = False

            h = restore_dense_layer(h,l,w,b,ac_fn=ac_fn,batch_norm=l_batch_norm,training=training,scope=scope)
            H.append(h)
            print('h shape',h.shape)
        if conv_L > 0:
            c = W[len(W)-conv_L].shape[-1]
            k = int(np.sqrt(h.shape[1].value/c))
            print('h,c,k',h.shape,c,k)
            h = tf.reshape(h,[-1,k,k,c])
            for l,(w,b) in enumerate(zip(W[len(W)-conv_L:],B[len(B)-conv_L:])):
                l_batch_norm = False
                # set batch norm every 2 layers, starting from first layer
                if batch_norm and l%2 == 0:
                    l_batch_norm = True
                # output layer
                if l == conv_L-1:
                    ac_fn = tf.nn.sigmoid
                    l_batch_norm = False
                strd = strides[l] if strides else [1,2,2,1]
                ow, oh = h.shape[1].value * strd[1], h.shape[2].value * strd[2]
                output_shape = [h.shape[0].value, ow, oh, w.shape[-2]]               
                h = restore_conv_layer(h,l,w,b,strides=strd,ac_fn=ac_fn,deconv=True,output_shape=output_shape,\
                                        batch_norm=l_batch_norm,training=training,scope=scope)
                H.append(h)
        return H

        
    def generator(self,e,*args,**kargs):       
        if self.is_training is None:
            return self.sess.run(self.g_H[-1],feed_dict={self.e_ph:e})
        else:
            return self.sess.run(self.g_H[-1],feed_dict={self.e_ph:e,self.is_training:False})


    def discriminator(self,x,*args,**kargs):
        if self.is_training is None:
            return self.sess.run(self.d_H[-1],feed_dict={self.x_ph:x})
        else:
            return self.sess.run(self.d_H[-1],feed_dict={self.x_ph:x,self.is_training:False})

    @staticmethod
    def Discriminator_Regularizer(d_real_logits, d_real_x, d_fake_logits, d_fake_x,batch_size):
        d_real = tf.nn.sigmoid(d_real_logits)
        d_fake = tf.nn.sigmoid(d_fake_logits)
        grad_d_real_logits = tf.gradients(d_real_logits, d_real_x)[0]
        grad_d_fake_logits = tf.gradients(d_fake_logits, d_fake_x)[0]
        grad_dr_logits_norm = tf.norm(tf.reshape(grad_d_real_logits, [batch_size,-1]), axis=1, keep_dims=True)
        grad_df_logits_norm = tf.norm(tf.reshape(grad_d_fake_logits, [batch_size,-1]), axis=1, keep_dims=True)

        #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
        assert grad_dr_logits_norm.shape == d_real.shape
        assert grad_df_logits_norm.shape == d_fake.shape

        reg_d_real = tf.multiply(tf.square(1.0-d_real), tf.square(grad_dr_logits_norm))
        reg_d_fake = tf.multiply(tf.square(d_fake), tf.square(grad_df_logits_norm))
        disc_regularizer = tf.reduce_mean(reg_d_real + reg_d_fake)
        return disc_regularizer
    
    def set_loss(self,*args,**kargs):
        d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_H[-1], labels=tf.ones_like(self.d_H[-1])))
        d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_H[-1], labels=tf.zeros_like(self.d_fake_H[-1])))
        
        d_loss = d_real_loss + d_fake_loss
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_H[-1], labels=tf.ones_like(self.d_fake_H[-1])))

        return g_loss, d_loss


    @staticmethod
    def config_train(loss,scope,learning_rate=0.0002,op_type='adam',beta1=0.5,decay=None,clip=None,*args,**kargs):
        
        var_list = get_var_list(scope)
        grads = tf.gradients(loss, var_list)
        if clip is not None:
            grads = [tf.clip_by_value(g, clip[0],clip[1]) for g in grads]
        grads_and_vars = list(zip(grads, var_list))
        print(scope,'learning rate',learning_rate)
        #print('var list',var_list)
        opt = config_optimizer(learning_rate,scope+'_step',grad_type=op_type,decay=decay,beta1=beta1)   

        train = opt[0].apply_gradients(grads_and_vars, global_step=opt[1])

        return train,var_list,opt

    def print_log(self,e,feed_dict,*args,**kargs):
        d_loss = self.sess.run([self.d_loss],feed_dict=feed_dict)
        g_loss = self.sess.run([self.g_loss],feed_dict=feed_dict)
        print('epoch',e,'d loss',d_loss,'g loss',g_loss)
        return 


    def update_feed_dict(self,X,Y,ii,batch_size,p=0.):
        feed_dict = {} if self.is_training is None else {self.is_training:True}
        x_batch,y_batch,ii = get_next_batch(X,batch_size,ii,labels=Y)
        e_batch = np.random.uniform(low=-1.,size=(batch_size,self.g_W[0].shape[0].value)).astype(np.float32)
        #e_batch = np.random.normal(size=(batch_size,self.g_W[0].shape[0].value)).astype(np.float32)

        feed_dict.update({self.x_ph:x_batch,self.e_ph:e_batch})
        if self.g_penalty and self.alpha > 0:
            feed_dict.update({self.gamma:self.gamma0*np.power(self.alpha,p)})
        return feed_dict,ii


    def training(self,X,Y,batch_size,epoch,d_obj=None,g_obj=None,disc_iter=1,vis=False,\
                    result_path='vis_results/',warm_start=False,merged=None,train_writer=None):
        
        if d_obj is None:
            d_obj = self.d_train
        if g_obj is None:
            g_obj = self.g_train

        if vis and not os.path.exists(result_path):
            os.makedirs(result_path)

        with self.sess.as_default():
            print('warm start',warm_start)
            if not warm_start:
                tf.global_variables_initializer().run()
            
            num_iters = int(np.ceil(X.shape[0]/batch_size))
            print('num iters',num_iters)
            T = epoch * num_iters
            for e in range(epoch):
                X,Y = shuffle_data(X,Y)
                ii = 0
                for i in range(num_iters):
                    feed_dict,ii = self.update_feed_dict(X,Y,ii,batch_size,p=i/T)

                    self.sess.run(d_obj,feed_dict=feed_dict)
                    #d_loss = self.sess.run([self.d_loss],feed_dict=feed_dict)

                    if (i+1) % disc_iter == 0:
                        self.sess.run(g_obj,feed_dict=feed_dict)
                        #g_loss = self.sess.run([self.g_loss],feed_dict=feed_dict)
                    if merged is not None:
                        summary = self.sess.run(merged,feed_dict=feed_dict)
                        train_writer.add_summary(summary, i+e*num_iters)


                self.print_log(e,feed_dict)
                #print('epoch',e,'d loss',d_loss,'g loss',g_loss)
                if vis and (e+1)%1==0:
                    e_samples = np.random.uniform(low=-1.,size=(batch_size,self.e_ph.shape[1].value)).astype(np.float32)
                    #e_samples = np.random.normal(size=(batch_size,self.e_ph.shape[1].value)).astype(np.float32)

                    x_samples = self.generator(e_samples)
                    ng = int(np.sqrt(batch_size))
                    fig = plot(x_samples[:ng*ng],shape=[ng,ng])
                    fig.savefig(os.path.join(result_path,'e'+str(e+1)+'.pdf'))
                    plt.close()
                    






class WGAN(GAN):

    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.0002,op_type='adam',clip=None,reg=None,*args,**kargs):
        super(WGAN,self).__init__(x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,\
                                    ac_fn,batch_norm,learning_rate,op_type,clip,reg,*args,**kargs)
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_var_list]
        return


    def set_loss(self,*args,**kargs):
        d_fake_loss = tf.reduce_mean(self.d_fake_H[-1])
        d_real_loss = tf.reduce_mean(self.d_H[-1])
        reg_loss = 0.0001*tf.reduce_sum(tf.losses.get_regularization_losses())
        d_loss = - d_real_loss + d_fake_loss + reg_loss
        g_loss = - d_fake_loss +reg_loss
        return g_loss,d_loss

    def training(self,X,Y,batch_size,epoch,disc_iter=1,vis=False,result_path='vis_results/',warm_start=False,*args,**kargs):
        d_obj = [self.d_train,self.clip_D]
        g_obj = self.g_train
        super(WGAN,self).training(X,Y,batch_size=batch_size,epoch=epoch,d_obj=d_obj,g_obj=g_obj,disc_iter=disc_iter,\
                                    vis=vis,result_path=result_path,warm_start=warm_start,*args,**kargs)

        return

class WGAN_GP(WGAN):

    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.0002,op_type='adam',clip=None,reg=None,lambd=0.25,*args,**kargs):

        self.lambd = lambd
        super(WGAN_GP,self).__init__(x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,\
                batch_norm,learning_rate,op_type,clip,reg,*args,**kargs)
        return


    def set_loss(self,*args,**kargs):
        g_loss,d_loss = super(WGAN_GP,self).set_loss()

        # This is copied from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = tf.random_uniform(shape=self.x_ph.get_shape(), minval=0.,maxval=1.)
        differences = self.g_H[-1] - self.x_ph # This is different from MAGAN
        interpolates = self.x_ph + (alpha * differences)
        _,__,self.d_inter_H = self.define_d_net(interpolates,self.d_net_shape,reuse=True,conv=self.conv,ac_fn=self.ac_fn,\
                                                batch_norm=self.batch_norm,training=self.is_training)

        gradients = tf.gradients(self.d_inter_H[-1], [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        d_loss += self.lambd * gradient_penalty

        return g_loss,d_loss



class fGAN(GAN):

    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.0002,op_type='adam',clip=None,reg=None,divergence='KL',\
                lamb_constr=0.,g_penalty=True,gamma0=1.,alpha=0.01,*args,**kargs):
        self.divergence = divergence
        self.lamb_constr = lamb_constr

        #print('lamb_constr',lamb_constr)

        print('x_ph',x_ph)
        super(fGAN,self).__init__(x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,batch_norm,\
                                learning_rate,op_type,clip,reg,g_penalty,gamma0,alpha,*args,**kargs)
        
        return

    @staticmethod
    def get_f(divergence,tf=False):
        if tf:
            log, exp, sqrt, square = tf.log, tf.exp, tf.sqrt, tf.square

        else:
            log, exp, sqrt, square = np.log, np.exp, np.sqrt, np.square

        if divergence == 'KL':
            def f(u):
                return u*log(u)
        elif divergence == 'rv_KL':
            def f(u):
                return -log(u)
        elif divergence == 'Pearson':
            def f(u):
                return square(u-1.)
        elif divergence == 'Hellinger':
            def f(u):
                return square(sqrt(u) - 1.)
        elif divergence == 'Jensen_Shannon':
            def f(u):
                return -(u+1.)*log((1.+u)*0.5) + u*log(u)
        elif divergence == 'GAN':
            def f(u):
                return u*log(u) - (u+1.)*log(u+1.)
        else:
            raise NotImplementedError('Divergence NOT supported.')      
        return f  

    @staticmethod
    def get_act_conj_fn(divergence):
        if divergence == 'KL':
            def act_fn(v):
                return v

            def conj_f(t):                
                return tf.exp(t-1.)

        elif divergence == 'rv_KL':
            def act_fn(v):
                return -tf.exp(-v)

            def conj_f(t):
                return -1.-tf.log(-t)

        elif divergence == 'Pearson':
            def act_fn(v):
                return v
            
            def conj_f(t):
                return 0.25 * tf.square(t) + t

        elif divergence == 'Hellinger':
            def act_fn(v):
                return 1. - tf.exp(-v)

            def conj_f(t):
                return t/(1.-t)

        elif divergence == 'Jensen_Shannon':
            def act_fn(v):
                return tf.log(2.) - tf.log(1.+tf.exp(-v))

            def conj_f(t):
                return -tf.log(2. - tf.exp(t))

        elif divergence == 'GAN':
            def act_fn(v):
                return - tf.log(1. + tf.exp(v))

            def conj_f(t):
                return - tf.log(1. - tf.exp(t))

        else:
            raise NotImplementedError('Divergence NOT supported.')

        return act_fn, conj_f
    
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

    
    def set_loss(self,d_h=None,d_fakeh=None):
        if d_h is None:
            d_h = self.d_H[-1] 
        if d_fakeh is None:
            d_fakeh = self.d_fake_H[-1]

        act_fn,conj_f = self.get_act_conj_fn(self.divergence)
        idf_gf = self.get_idf_gf(self.divergence)
        F = tf.reduce_mean(act_fn(d_h)) + tf.reduce_mean(-conj_f(act_fn(d_fakeh)))
        reg_loss = 0.0001 * tf.reduce_sum(tf.losses.get_regularization_losses())
        #r_constr =  tf.square(tf.reduce_mean(idf_gf(tf.stop_gradient(d_fakeh[-1]))) - 1.) + tf.square(tf.reduce_mean(1./idf_gf(d_h[-1])) - 1.)

        g_loss = F + reg_loss #+ self.lamb_constr * r_constr
        d_loss = -F + reg_loss #+ self.lamb_constr * r_constr

        if self.g_penalty:
            
            d_reg = self.Discriminator_Regularizer(d_h,self.x_ph,d_fakeh,self.g_H[-1],self.batch_size)
            #d_reg = self.Discriminator_Regularizer(self.d_H[-1],self.x_ph,self.d_fake_H[-1],self.g_H[-1],self.batch_size)
            #d_reg = self.Discriminator_Regularizer(self.d_fake_H[-1],self.g_H[-1],self.batch_size,act_fn)

            d_loss += (self.gamma*0.5)*d_reg
            g_loss = -tf.reduce_mean(act_fn(d_fakeh)) + reg_loss

        return g_loss, d_loss



class CGAN(GAN):

    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.001,op_type='adam',clip=None,reg=None,pooling=False,\
                c_dim=10,*args,**kargs):
        self.c_dim = c_dim
        #self.c_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,c_dim]) 
        super(CGAN,self).__init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,\
                batch_norm,learning_rate,op_type,clip,reg,pooling,*args,**kargs)

        return


    def define_model(self,x_ph,g_net_shape,d_net_shape,batch_size,conv,ac_fn,batch_norm,learning_rate,\
                        op_type,clip=None,reg=None,pooling=False,*args,**kargs):
        self.conv = conv
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[]) if batch_norm else None
        self.x_ph = x_ph # true data
        k = g_net_shape[0][0] if conv else g_net_shape[0] 
        c_dim = self.c_dim  
        print('check e ph',batch_size,'k',k,'c_dim',c_dim)    
        self.e_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,k]) 
        self.c_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,c_dim])     
        print('kargs',**kargs)
        self.g_W,self.g_B,self.g_H = self.define_g_net(self.e_ph,g_net_shape,reuse=False,conv=conv,ac_fn=ac_fn,\
                                                    batch_norm=batch_norm,training=self.is_training,reg=reg,\
                                                    pooling=pooling,**kargs)
        
        self.d_W,self.d_B,self.d_H = self.define_d_net(self.x_ph,d_net_shape,reuse=False,conv=conv,ac_fn=ac_fn,\
                                                    batch_norm=batch_norm,training=self.is_training,reg=reg,\
                                                    pooling=pooling,**kargs)
        
        self.fake_x = concat_cond_data(self.g_H[-1],self.c_ph,one_hot=False,dim=c_dim,conv=conv)
        _,__,self.d_fake_H = self.define_d_net(self.fake_x,d_net_shape,reuse=True,conv=conv,ac_fn=ac_fn,batch_norm=batch_norm,training=self.is_training,\
                                            reg=reg,pooling=pooling,**kargs)
        
        self.g_loss,self.d_loss = self.set_loss()
        self.g_train,self.g_var_list,self.g_opt = self.config_train(self.g_loss,scope='generator',learning_rate=learning_rate*5,clip=clip)
        self.d_train,self.d_var_list,self.d_opt = self.config_train(self.d_loss,scope='discriminator',learning_rate=learning_rate,clip=clip)
        
        return


    @staticmethod
    def gen_feed_dict(x_ph,e_ph,c_ph,X,Y,ii,batch_size,is_training,k,c_dim,conv=False,one_hot=False):
        #print('condition gan')
        feed_dict = {} if is_training is None else {is_training:True}
        x_batch,y_batch,ii = get_next_batch(X,batch_size,ii,labels=Y)
        e_batch = np.random.uniform(low=-1.,size=(batch_size,k)).astype(np.float32)
        #e_batch = np.random.normal(size=(batch_size,k)).astype(np.float32)

        x_batch = concat_cond_data(x_batch,y_batch,one_hot=one_hot,dim=c_dim,conv=conv)
        e_batch = concat_cond_data(e_batch,y_batch,one_hot=one_hot,dim=c_dim,conv=False)
        
        feed_dict.update({x_ph:x_batch,e_ph:e_batch,c_ph:y_batch})

        return feed_dict,ii

    def update_feed_dict(self,X,Y,ii,batch_size,one_hot=False,p=0.):
        #print('X SHAPE',X.shape,'Y SHAPE',Y.shape)
        k = self.g_W[0].shape[0].value-self.c_dim
        feed_dict,ii = self.gen_feed_dict(self.x_ph,self.e_ph,self.c_ph,X,Y,ii,batch_size,\
                                    self.is_training,k,self.c_dim,self.conv,one_hot)
        if self.g_penalty and self.alpha > 0:
            feed_dict.update({self.gamma:self.gamma0 * np.power(self.alpha,p)})
        return feed_dict, ii

    @staticmethod
    def Discriminator_Regularizer(d_real_logits, d_real_x, d_fake_logits, d_fake_x,batch_size,c_ph):
        d_real_logits = tf.expand_dims(tf.reduce_sum(d_real_logits * c_ph,axis=1),1)
        d_fake_logits = tf.expand_dims(tf.reduce_sum(d_fake_logits * c_ph,axis=1),1)
        d_real = tf.nn.sigmoid(d_real_logits)
        d_fake = tf.nn.sigmoid(d_fake_logits)
        grad_d_real_logits = tf.gradients(d_real_logits, d_real_x)[0]
        grad_d_fake_logits = tf.gradients(d_fake_logits, d_fake_x)[0]
        print('logits shape',grad_d_fake_logits.shape)
        grad_dr_logits_norm = tf.norm(tf.reshape(grad_d_real_logits, [batch_size,-1]), axis=1, keep_dims=True)
        grad_df_logits_norm = tf.norm(tf.reshape(grad_d_fake_logits, [batch_size,-1]), axis=1, keep_dims=True)
        print('logits norm shape',grad_df_logits_norm.shape)
        print('d_real shape',d_real.shape)
        #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
        assert grad_dr_logits_norm.shape == d_real.shape
        assert grad_df_logits_norm.shape == d_fake.shape

        reg_d_real = tf.multiply(tf.square(1.0-d_real), tf.square(grad_dr_logits_norm))
        reg_d_fake = tf.multiply(tf.square(d_fake), tf.square(grad_df_logits_norm))
        disc_regularizer = tf.reduce_mean(reg_d_real + reg_d_fake)
        return disc_regularizer


class CWGAN(CGAN,WGAN):
    
    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.001,op_type='adam',clip=None,reg=None,c_dim=10,*args,**kargs):
        self.c_dim = c_dim
        #self.c_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,c_dim]) 
        WGAN.__init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,\
                batch_norm,learning_rate,op_type,clip,reg,*args,**kargs)

        return

    def set_loss(self):
        return WGAN.set_loss(self)


class CWGAN_GP(CGAN,WGAN_GP):
    
    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.001,op_type='adam',clip=None,reg=None,c_dim=10,lambd=0.25,*args,**kargs):
        self.c_dim = c_dim
        #self.c_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,c_dim]) 
        WGAN_GP.__init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,\
                batch_norm,learning_rate,op_type,clip,reg,lambd,*args,**kargs)

        return

    def set_loss(self):
        return WGAN_GP.set_loss(self)


class CfGAN(CGAN,fGAN):
    
    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.001,op_type='adam',clip=None,reg=None,divergence='KL',\
                c_dim=10,lamb_constr=0.,g_penalty=True,gamma0=1.,alpha=0.01,*args,**kargs):
        self.c_dim = c_dim
        #print('lamb constr',lamb_constr)
        #self.c_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,c_dim]) 
        fGAN.__init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,\
                batch_norm,learning_rate,op_type,clip,reg,divergence,lamb_constr,g_penalty,gamma0,alpha,*args,**kargs)

        return
    
    def set_loss(self,d_h=None,d_fakeh=None):
        if d_h is None:
            d_h = self.d_H[-1] 
        if d_fakeh is None:
            d_fakeh = self.d_fake_H[-1]

        act_fn,conj_f = self.get_act_conj_fn(self.divergence)
        idf_gf = self.get_idf_gf(self.divergence)
        F = tf.reduce_mean(act_fn(d_h),axis=0) + tf.reduce_mean(-conj_f(act_fn(d_fakeh)),axis=0)
        reg_loss = 0.0001 * tf.reduce_sum(tf.losses.get_regularization_losses())
        #r_constr =  tf.square(tf.reduce_mean(idf_gf(tf.stop_gradient(d_fakeh[-1]))) - 1.) + tf.square(tf.reduce_mean(1./idf_gf(d_h[-1])) - 1.)

        g_loss = F + reg_loss #+ self.lamb_constr * r_constr
        d_loss = -F + reg_loss #+ self.lamb_constr * r_constr

        if self.g_penalty:
            
            d_reg = self.Discriminator_Regularizer(d_h,self.x_ph,d_fakeh,self.g_H[-1],self.batch_size,self.c_ph)
            #d_reg = self.Discriminator_Regularizer(self.d_H[-1],self.x_ph,self.d_fake_H[-1],self.g_H[-1],self.batch_size)
            #d_reg = self.Discriminator_Regularizer(self.d_fake_H[-1],self.g_H[-1],self.batch_size,act_fn)

            d_loss += (self.gamma*0.5)*d_reg*self.c_ph
            g_loss = -tf.reduce_mean(act_fn(d_fakeh),axis=0) + reg_loss
        
        # for multi-dims output
        c_num = tf.reduce_sum(self.c_ph,axis=0)
        c_prop = c_num /tf.reduce_sum(c_num)
        d_loss = tf.reduce_sum(d_loss * c_prop)
        g_loss = tf.reduce_sum(g_loss * c_prop)

        return g_loss, d_loss
