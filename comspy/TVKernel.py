from __future__ import division
import numpy as np
import numpy.fft as fft
import pywt
import direction as d
from scipy.ndimage.filters import correlate

def TV(im, N, strtag, kern, dirWeight=1, dirs=None, nmins=0, dirInfo=[None,None,None,None]):
    res = np.zeros(np.hstack([len(strtag), im.shape]))
    inds = dirInfo[3]
    Nkern = np.hstack([1,kern.shape[-2:]])
    for i in xrange(len(strtag)):
        if strtag[i] == 'spatial':
            res[i] = correlate(im,kern[i].reshape(Nkern),mode='wrap')
        elif strtag[i] == 'diff':
            res[i] = dirWeight*d.least_Squares_Fitting(im,N,strtag,dirs,inds,dirInfo[0]).real
    return res.reshape(np.hstack([len(strtag), N]))
    
    
def gTV(x, N, strtag, kern, dirWeight, dirs=None, nmins=0, dirInfo=[None,None,None,None], a=10):

    if nmins:
        M = dirInfo[0]
        dIM = dirInfo[1]
        Ause = dirInfo[2]
        inds = dirInfo[3]
    else:
        M = None
        dIM = None
        Ause = None
        inds = None
    
    if len(x.shape) == 2:
        N = np.hstack([1,N])
        
    x0 = x.reshape(N)
    grad = np.zeros(np.hstack([N[0], len(strtag), N[1:]]),dtype=float)
    Nkern = np.hstack([1,kern.shape[-2:]])
    
    TV_data = TV(x0,N,strtag,kern,dirWeight,dirs,nmins,dirInfo)
    for i in xrange(len(strtag)):
        if strtag[i] == 'spatial':
            kernHld = np.flipud(np.fliplr(kern[i])).reshape(Nkern)
            grad[:,i,:,:] = correlate(np.tanh(a*TV_data[i]),kernHld,mode='wrap')
            
    return grad