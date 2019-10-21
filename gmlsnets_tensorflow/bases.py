import numpy as np
import itertools
from collections import Counter
import toolz
import scipy
import tensorflow as tf

#generates taylor polynomial basis (TPB) of `order` and dimension, `dim`
class Taylor:
    def __init__(self,dim,order):
        self.dim = dim
        self.order = order
        self.indc = Counter(map(toolz.compose(tuple,sorted),
                            itertools.chain(
                                *[itertools.product(*[range(dim) for _ in range(o)])
                                                                        for o in range(order+1)])
                  ))
        self.len = len(self.indc)

    #evaluates the TPB on an numpy array of points, `x:[point,coordinate]`
    def __call__(self,x):
        return np.concatenate([np.expand_dims(
                            1.*self.indc[ind]/scipy.math.factorial(len(ind))
                              *np.prod([x[:,d]**ind.count(d) for d in range(self.dim)],0),
                                              1)  for ind in sorted(self.indc)],1)

    #evalutes TPB at `x` with coefficients `C`
    def inner(self,C,x):
        return tf.reduce_sum(C*self.__call__(x),-1) 


#generates tensor product basis
class TP:
    def __init__(self,dim,order):
        self.dim = dim
        self.order = order  #for 1d polynomial
        self.indc = list(itertools.product(*[range(order+1) for _ in range(dim)]))
        self.len = len(self.indc)

    def __call__(self,x):
        return np.stack(
                   [np.prod([x[:,d]**ind[d] for d in range(self.dim)],0) for ind in self.indc],-1)

    def inner(self,C,x):
        return tf.reduce_sum(C*self.__call__(x),-1) 


