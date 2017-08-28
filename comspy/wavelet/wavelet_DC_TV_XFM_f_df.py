from __future__ import division
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'

import os.path
import transforms as tf
import scipy.ndimage.filters
import gradWavelet as grads
import sampling as samp
import direction as d
import optimize as opt
EPS = np.finfo(float).eps

# ----------------------------------------------------- #
# ------- MAJOR FUNCTIONS FOR USE IN MAIN CODE -------- #
# ----------------------------------------------------- #

def objectiveFunction(x, N, N_im, dims, dimOpt, dimLenOpt, lam1, lam2, data, k, strtag, ph, kern,
                      dirWeight=0, dirs=None, dirInfo=[None,None,None,None], nmins=0, wavelet='db4', mode="per", a=10.):
    '''
    This is the optimization function that we're trying to optimize. We are optimizing x here, and testing it within the funcitons that we want, as called by the functions that we've created
    '''
    #dirInfo[0] is M
    #import pdb; pdb.set_trace()
    tv = 0
    xfm = 0
    data.shape = N_im
    x.shape = N
    if len(N) > 2:
        x0 = np.zeros(N_im)
        for i in xrange(N[0]):
            x0[i,:,:] = tf.iwt(x[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)
    else:
        x0 = tf.iwt(x,wavelet,mode,dims,dimOpt,dimLenOpt)
    
    obj = np.sum(objectiveFunctionDataCons(x0,N_im,ph,data,k))
    
    if lam1 > 1e-6:
        tv = np.sum(objectiveFunctionTV(x0,N_im,strtag,kern,dirWeight,dirs,nmins,dirInfo=dirInfo,a=a))
    
    if lam2 > 1e-6:
        xfm = np.sum((1/a)*np.log(np.cosh(a*x)))
    
    x.shape = (x.size,) # Not the most efficient way to do this, but we need the shape to reset.
    data.shape = (data.size,)
    ##output
    #print('obj: %.2f' % (obj))
    #print('tv: %.2f' % (lam1*tv))
    #print('xfm: %.2f' % (lam2*xfm))
    return abs(obj + lam1*tv + lam2*xfm)


def derivativeFunction(x, N, N_im, dims, dimOpt, dimLenOpt, lam1, lam2, data, k, strtag, ph, 
                       kern, dirWeight=0.1, dirs=None, dirInfo=[None,None,None,None], nmins=0, wavelet="db4", mode="per", a=10.):
    '''
    This is the function that we're going to be optimizing via the scipy optimization pack. This is the function that represents Compressed Sensing
    '''
    #import pdb; pdb.set_trace()
    disp = 0
    gTV = 0
    gXFM = 0
    x.shape = N
    if len(N) > 2:
        x0 = np.zeros(N_im)
        for i in xrange(N[0]):
            x0[i,:,:] = tf.iwt(x[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)
    else:
        x0 = tf.iwt(x,wavelet,mode,dims,dimOpt,dimLenOpt)
    
    gDataCons = tf.wt(grads.gDataCons(x0,N_im,ph,data,k),wavelet,mode,dims,dimOpt,dimLenOpt)[0]
    if lam1 > 1e-6:
        gTV = tf.wt(grads.gTV(x0,N_im,strtag,kern,dirWeight,dirs,nmins,dirInfo=dirInfo,a=a),wavelet,mode,dims,dimOpt,dimLenOpt)[0] # Calculate the TV gradient
    if lam2 > 1e-6:
        gXFM = grads.gXFM(x,a=a)
    
    x.shape = (x.size,)
    
    return (gDataCons + lam1*gTV + lam2*gXFM).flatten() # Export the flattened array
    
    
    
    
    
    
    
# ----------------------------------------------------- #
# -------- Individual Calculations for clarity -------- #
# ----------------------------------------------------- #

def objectiveFunctionDataCons(x, N, ph, data, k):
    obj_data = k*(data - tf.fft2c(x,ph))
    return obj_data*obj_data.conj() #L2 Norm

def objectiveFunctionTV(x, N, strtag, kern, dirWeight=0, dirs=None, nmins=0,
                        dirInfo=[None,None,None,None], a=10):
    return (1/a)*np.log(np.cosh(a*tf.TV(x,N,strtag,kern,dirWeight,dirs,nmins,dirInfo)))
    
def objectiveFunctionXFM(x, a=10):
    return np.sum((1/a)*np.log(np.cosh(X)))
    