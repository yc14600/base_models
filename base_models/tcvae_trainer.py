from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import six
import os
path = os.getcwd()
import sys
sys.path.append(path+'/../')

import time
import math
import collections
from numbers import Number

import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader

import tcvae.lib.dist as dist
import tcvae.lib.utils as utils
import tcvae.lib.datasets as dset
from tcvae.lib.flows import FactorialNormalizingFlow

from tcvae.elbo_decomposition import elbo_decomposition
from tcvae.plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces
from tcvae.vae_quant import *

from tensorflow.examples.tutorials.mnist import input_data


class VAE_trainer(object):
    def __init__(self,args,use_cuda=True):
        self.args = args
        self.use_cuda = use_cuda
        if use_cuda:
            torch.cuda.set_device(args.gpu)

        

        # setup the VAE
        if args.dist == 'normal':
            self.prior_dist = dist.Normal()
            self.q_dist = dist.Normal()
        elif args.dist == 'laplace':
            self.prior_dist = dist.Laplace()
            self.q_dist = dist.Laplace()
        elif args.dist == 'flow':
            self.prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
            self.q_dist = dist.Normal()

        self.vae = VAE(z_dim=args.latent_dim, use_cuda=use_cuda, prior_dist=self.prior_dist, q_dist=self.q_dist,
            include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss)
        
        # setup the optimizer
        self.optimizer = optim.Adam(self.vae.parameters(), lr=args.learning_rate)

        # setup visdom for visualization
        if args.visdom:
            self.vis = visdom.Visdom(env=args.save, port=4500)
            
    def train(self,X):
        args = self.args
        # data loader
        X = torch.Tensor(X)
        train_loader = setup_data_loaders(args,X,use_cuda=self.use_cuda)

        train_elbo = []
        args = self.args
        vae =self.vae
        # training loop
        dataset_size = len(train_loader.dataset)
        num_iterations = len(train_loader) * args.num_epochs
        iteration = 0
        # initialize loss accumulator
        elbo_running_mean = utils.RunningAverageMeter()
        while iteration < num_iterations:
            for i, x in enumerate(train_loader):
                iteration += 1
                batch_time = time.time()
                self.vae.train()
                anneal_kl(args, vae, iteration)
                self.optimizer.zero_grad()
                # transfer to GPU
                if self.use_cuda:
                    x = x.cuda(async=True)
                # wrap the mini-batch in a PyTorch Variable
                x = Variable(x)
                # do ELBO gradient and accumulate loss
                obj, elbo = vae.elbo(x, dataset_size)
                if utils.isnan(obj).any():
                    raise ValueError('NaN spotted in objective.')
                obj.mean().mul(-1).backward()
                elbo_running_mean.update(elbo.mean().data[0])
                self.optimizer.step()

                # report training diagnostics
                if iteration % args.log_freq == 0:
                    train_elbo.append(elbo_running_mean.avg)
                    print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)' % (
                        iteration, time.time() - batch_time, vae.beta, vae.lamb,
                        elbo_running_mean.val, elbo_running_mean.avg))

                    vae.eval()

                    # plot training and test ELBOs
                    if args.visdom:
                        display_samples(vae, x, self.vis)
                        plot_elbo(train_elbo, self.vis)
                    if args.save_model:
                        utils.save_checkpoint({
                            'state_dict': vae.state_dict(),
                            'args': args}, args.save, 0)
    
    
    def reinit(self):
        args = self.args
        for i,m in enumerate(self.vae.modules()):
            #print(i,m)
            if isinstance(m,torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()
                print('reset params',i,m)
            
        #print('before reinit opt',self.optimizer.state)
        self.optimizer.__init__(self.vae.parameters(), lr=args.learning_rate)
        #print('after reinit opt', self.optimizer.state)
    
    def encode(self,x):
        return self.vae.encode(x)[0]

    
    
    
    
    
    