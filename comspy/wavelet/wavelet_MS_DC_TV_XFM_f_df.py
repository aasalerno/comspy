from __future__ import division
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'

import os.path
import transforms as tf
import scipy.ndimage.filters
import gradWaveletMS as grads
import sampling as samp
import direction as d
EPS = np.finfo(float).eps

# ----------------------------------------------------- #
# ------- MAJOR FUNCTIONS FOR USE IN MAIN CODE -------- #
# ----------------------------------------------------- #

def objectiveFunction(x, N, N_im, sz, dims, dimOpt, dimLenOpt, lam1, lam2, data, k, strtag, ph,     
                      kern, dirWeight=0, dirs=None, dirInfo=[None,None,None,None], nmins=0, wavelet='db4', mode="per", a=10.):
    '''
    This is the optimization function that we're trying to optimize. We are optimizing x here, and testing it within the funcitons that we want, as called by the functions that we've created
    '''
    #dirInfo[0] is M
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
    
    obj = np.sum(objectiveFunctionDataCons(x0,N_im,ph,data,k,sz))
    
    if lam1 > 1e-6:
        tv = np.sum(objectiveFunctionTV(x0,N_im,strtag,kern,dirWeight,dirs,nmins,dirInfo=dirInfo,a=a))
    
    if lam2 > 1e-6:
        xfm = np.sum((1/a)*np.log(np.cosh(a*x)))
    
    x.shape = (x.size,) # Not the most efficient way to do this, but we need the shape to reset.
    data.shape = (data.size,)
    #import pdb; pdb.set_trace()
    ##output
    #print('obj: %.2f' % (obj))
    #print('tv: %.2f' % (lam1*tv))
    #print('xfm: %.2f' % (lam2*xfm))
    return abs(obj + lam1*tv + lam2*xfm)


def derivativeFunction(x, N, N_im, sz, dims, dimOpt, dimLenOpt, lam1, lam2, data, k, strtag, ph, 
                       kern, dirWeight=0.1, dirs=None, dirInfo=[None,None,None,None], nmins=0, wavelet="db4", mode="per", a=10.):
    '''
    This is the function that we're going to be optimizing via the scipy optimization pack. This is the function that represents Compressed Sensing
    '''
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
    
    gdc = grads.gDataCons(x0,N_im,ph,data,k,sz)
    #import pdb; pdb.set_trace()
    if lam1 > 1e-6:
        gtv = grads.gTV(x0,N_im,strtag,kern,dirWeight,dirs,nmins,dirInfo=dirInfo,a=a)
    
    gDataCons = np.zeros(N)
    gTV = np.zeros(N)
    gXFM = np.zeros(N)
    
    for i in xrange(N[0]):
        gDataCons[i,:,:] = tf.wt(gdc[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)[0]
        if lam1 > 1e-6:
            gTV[i,:,:] = tf.wt(gtv[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)[0] # Calculate the TV gradient
        if lam2 > 1e-6:
            gXFM[i,:,:] = grads.gXFM(x[i,:,:],a=a)
    
    x.shape = (x.size,)
    
    return (gDataCons + lam1*gTV + lam2*gXFM).flatten() # Export the flattened array
    
    
    
    
    
    
    
# ----------------------------------------------------- #
# -------- Individual Calculations for clarity -------- #
# ----------------------------------------------------- #

def objectiveFunctionDataCons(x, N, ph, data, k, sz):
    #import pdb; pdb.set_trace()
    obj_data = k*(data - tf.fftnc(x,ph,sz=sz))
    return obj_data*obj_data.conj() #L2 Norm

def objectiveFunctionTV(x, N, strtag, kern, dirWeight=0, dirs=None, nmins=0,
                        dirInfo=[None,None,None,None], a=10):
    return (1/a)*np.log(np.cosh(a*tf.TV(x,N,strtag,kern,dirWeight,dirs,nmins,dirInfo)))
    
def objectiveFunctionXFM(x, a=10):
    return np.sum((1/a)*np.log(np.cosh(X)))
    