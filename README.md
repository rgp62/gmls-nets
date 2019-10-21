# GMLS-Nets
This code is the Tensorflow implementation of

N. Trask, R. G. Patel, B. J. Gross, and P. J. Atzberger, "GMLS-Nets: A Framework for Learning from Unstructured Data," arXiv:1909.05371, (2019).


#### Installation
`pip install gmlsnets-tensorflow`

#### Requirements
`Python >= 3.5`
`numpy`  
`scipy`  
`matplotlib`  
`scikit-learn`  
`toolz`  
`tensorflow`  

#### Usage
The three classes in `gmlsnets_tensorflow/__init__.py` provide Keras layers used to construct GMLS-Nets architectures. `MFLayer` creates layers that compute GMLS coefficients from functions sampled on a point cloud. `MFConvLayer` and `MFPoolLayer` create for point cloud data the equivalent to (strided) convolutional layers and pool layers, respectively. These classes use the weighting functions in `gmlsnets_tensorflow/weightfuncs.py` and the polynomial bases in `gmlsnets_tensorflow/bases.py`. See the examples folder for MNIST and PDE discovery examples. 

#### Additional information
For the PyTorch implementation, see https://github.com/atzberg/gmls-nets.
