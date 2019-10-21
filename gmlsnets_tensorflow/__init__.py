import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import scipy.spatial
import scipy
from sklearn.neighbors import NearestNeighbors
import numpy as np
import itertools
import toolz
import gmlsnets_tensorflow.bases as bases
import gmlsnets_tensorflow.weightfuncs as weightfuncs


#creates a keras layer to compute GMLS coefficients at `x2` given point evaluations of a function
#at `x`. Uses basis fP, weight function fW and cutoff eps
class MFLayer(Layer):
    def __init__(self, x, x2, fP, fW, eps, **kwargs):

        super(MFLayer, self).__init__(dtype = x.dtype,**kwargs)

        self.output_dim = x2.shape[0]
        self.dim = x.shape[1]
        self.x = x
        self.x2 = x2
        self.xinds = list(range(len(x)))
        self.fP = fP
        self.fW = fW
        self.eps = eps

        self.x_tree = scipy.spatial.cKDTree(x)
        self.neighbors = self.x_tree.query_ball_point(x2, eps)
        self.maxneigh = np.max(list(map(len,self.neighbors)))

        self.Pija = [self.fP((self.x[self.neighbors[i]] - x2[i])/self.eps)
                                                for i in range(self.output_dim)]
        self.Wij  = [self.fW(self.x2[i], self.x[self.neighbors[i]],self.eps)
                                                for i in range(self.output_dim)]
        self.Miab_inv = [scipy.linalg.pinv(np.sum(
                                           np.expand_dims(self.Pija[i],2)
                                          *np.expand_dims(self.Pija[i],1)
                                          *np.expand_dims(np.expand_dims(self.Wij[i],1),1),0))
                            for i in range(self.output_dim)]

        self.Qija = [np.einsum('ab,jb,j->ja',self.Miab_inv[i],self.Pija[i],self.Wij[i])
                                        for i in range(self.output_dim)]

        merge = toolz.partial(toolz.reduce,lambda x,y:x+y)
        entries = sum(np.array(list(map(lambda Qja:np.shape(Qja)[0],self.Qija)))*fP.len)
        Qind = np.zeros((entries,2),dtype=int)
        Qneigh = np.zeros((entries,),dtype=int)
        Qval = np.zeros((entries,),dtype=self.dtype)

        ic = 0
        for i,Qja in enumerate(self.Qija):
            Qind[ic:ic+len(Qja)*fP.len] = np.array(list(
                            itertools.product(range(i*fP.len,(i+1)*fP.len),range(len(Qja)))))
            Qneigh[ic:ic+len(Qja)*fP.len] = np.tile(self.neighbors[i],fP.len)
            Qval[ic:ic+len(Qja)*fP.len] = np.reshape(Qja.T,(np.prod(np.shape(Qja)),))
            ic = ic+len(Qja)*fP.len
        
        self.sp_ids = tf.sparse.SparseTensor(Qind,Qneigh,
                                            (self.output_dim*fP.len,self.maxneigh))
        self.sp_weights = tf.sparse.SparseTensor(Qind,Qval,
                                            (self.output_dim*fP.len,self.maxneigh))


    def call(self,U):
        if int(U.shape[-1]) != len(self.x):
            raise Exception('Wrong size')
        UT=tf.transpose(U)
        Cia = tf.nn.embedding_lookup_sparse(UT,self.sp_ids,self.sp_weights,combiner='sum')
        CiaT = tf.reshape(tf.transpose(Cia),
                            ([-1 if us is None else us for us in U.shape.as_list()[:-1]]
                               +[self.output_dim,self.fP.len]))
        return CiaT

    def build(self, input_shape):
        super(MFLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


#evaluates a set of parameterized functionals at `x2` applied to functions sampled at `x`.
#Equivalent to (strided) convolutional layers.
class MFConvLayer(MFLayer):
    def __init__(self, x, x2, fP, fW, eps, channels,
                    activation=None,
                    use_bias=True,
                    kernel_initializer = 'glorot_uniform',
                    bias_initializer='zeros', **kwargs):

        super(MFConvLayer, self).__init__(x, x2, fP, fW, eps, **kwargs)

        self.channels = channels
        self.activation = activation
        self.activationf = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self,input_shape):
        super(MFConvLayer, self).build(input_shape)
        kernel_shape = (self.fP.len,int(input_shape[-1]),self.channels)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias')

    def call(self,U):
        permU = list(range(len(U.shape)))
        permC = list(range(len(U.shape)+1))
        permU[-2::] = [permU[-1],permU[-2]]
        permC[-3::] = [permC[-2],permC[-1],permC[-3]]
        C = tf.transpose(super(MFConvLayer, self).call(tf.transpose(U,permU)),permC)
        V = tf.einsum('jkl,...ijk->...il',self.kernel,C)
        if self.use_bias:
            V = V + self.bias
        if self.activation is not None:
            V = self.activationf(V)
        return V




#maps data from one pointcloud to another using `reduce_op`. Equivalent to pool layers.
class MFPoolLayer(Layer):
    def __init__(self,x, x2, eps, reduce_op, **kwargs):

        super(MFPoolLayer, self).__init__(dtype = x.dtype,**kwargs)


        self.output_dim = x2.shape[0]
        self.x_tree = scipy.spatial.cKDTree(x)
        self.x_xold= self.x_tree.query_ball_point(x2, eps)

        self.reduce_op = reduce_op

    def call(self,Uold):
        Uoldi = np.array(tf.unstack(Uold,axis=-2))
        U = tf.stack([self.reduce_op([Uoldi[j] for j in self.x_xold[i]],0)
                            for i in range(self.output_dim)],-2)
        return U

    def build(self, input_shape):
        super(MFPoolLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
