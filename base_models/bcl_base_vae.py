from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import csv
import copy
import six
import importlib
import os
path = os.getcwd()
import sys
sys.path.append(path+'/../')

import tensorflow as tf
# In[3]:
from abc import ABC, abstractmethod
from utils.model_util import *
from utils.train_util import *
from utils.coreset_util import *

from hsvi.hsvi import Hierarchy_SVI
from hsvi.methods.svgd import SVGD

class BCL_VAE(ABC):
    
    def __init__(self,net_shape,x_ph,num_heads=1,batch_size=512,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,dropout=None,initialization=None,\
                    ac_fn=tf.nn.relu,*args,**kargs):

        
        super(BCL_VAE,self).__init__(net_shape,x_ph,None,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,ac_fn)

        return

    def define_model(self,dropout=None,initialization=None):
        in_dim = self.net_shape[0]