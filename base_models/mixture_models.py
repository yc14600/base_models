import numpy as np
import tensorflow as tf
#import edward as ed


#from edward.models import Normal
from scipy.stats import multivariate_normal, norm

class MixDiagGaussian(object):
    def __init__(self,means,stds,weights,dim):
        self.components = self.gen_components(means,stds,dim)
        self.means = means
        self.stds = stds
        self.weights = weights
        self.dim = dim

    def gen_components(self,means,stds,dim):
        components = []
        if dim > 1:
            for m,s in zip(means,stds):
                #print('check component',m,s)
                dist = multivariate_normal(mean=np.ones(dim)*m,cov=np.ones(dim)*s)
                components.append(dist)
        else:
            dist = norm(loc=m,scale=s)
            components.append(dist)
        return components

    def log_prob(self,x):
        return np.log(self.prob(x))
        
    def prob(self,x):
        cp = 0.
        for w,c in zip(self.weights,self.components):
            #print('check x',x[0])
            #print('check c parm',sess.run(c.loc))
            #print('check c prob',np.sum(sess.run(c.prob(x))==0.))
            cp += w * c.pdf(x)
        '''
        p = cp[:,0]
      
        for d in range(1,self.dim):
            p += cp[:,d]
            #lp = tf.log(p[0])
            print('check logp',sess.run(p))
        '''
        return cp

    def sample(self,size=1):
        ids = np.random.choice(len(self.weights),size=size,p=self.weights)
        ss = np.zeros([size,self.dim],dtype=np.float32)
        for i in range(len(self.weights)):
            idx = ids==i
            ss[idx] = np.random.normal(loc=self.means[i],scale=self.stds[i],size=(int(np.sum(idx)),self.dim)).astype(np.float32)
        #print(ss[:5])
        return ss       