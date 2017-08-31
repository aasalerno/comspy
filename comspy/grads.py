#
#
# grads.py
#
#
from __future__ import division
import numpy as np
import transforms as tf
import matplotlib.pyplot as plt
from scipy.ndimage.filters import correlate


def gXFM(x, a=10):
    '''
    In this code, we apply an approximation of what the 
    value would be for the gradient of the XFM (usually wavelet)
    on the data. The approximation is done using the form:
        
    |x| =. sqrt(x.conj()*x)
    
    Using this, we are tryin to come up with a form that is cts about
    all x.
    
    Because of how this is done, we need to ensure that it is applied on
    a slice by slice basis.
    
    Inputs:
    [np.array] x - data that we're looking at
    [int]      p - The norm of the value that we're using
    [float]  l1smooth - Smoothing value
    
    Outputs:
    [np.array] grad - the gradient of the XFM
    
    '''
    
    
    #x0 = x.reshape(N)
    #grad = np.zeros(N)
    #    for i in xrange(x.shape[2]):
    #        x1 = x[...,...,i]
    #        grad[...,...,i] = p*x1*(x1*x1.conj()+l1smooth)**(p/2-1)
    #grad = p*x0*(x0*x0.conj()+l1smooth)**(p/2.0-1)
    grad = np.tanh(a*x)
    
    #import pdb; pdb.set_trace()
    return grad

def gDataCons(x, N, ph, data, k):
    '''
    Here, we are attempting to get the objective derivative from the
    function. This gradient is how the current data compares to the 
    data from the scanner in order to try and enforce data consistency.
    
    Inputs:
    [np.array] x - data that we're looking at
    [np.array] data_from_scanner - the original data from the scanner
    [int/boolean] samp_mask - Mask so we only compare the data from the regions of k-space that were sampled
    [int]      p - The norm of the value that we're using
    [float]  l1smooth - Smoothing value
    
    Outputs:
    [np.array] grad - the gradient of the XFM
    
    '''
    if len(N) == 2:
        N = np.hstack([1,N])
    #grad = np.zeros([x.shape])

    # Here we're going to convert the data into the k-sapce data, and then subtract
    # off the original data from the scanner. Finally, we will convert this data 
    # back into image space
    x0 = x.reshape(N)
    data.shape = N
    grad = np.zeros(N,complex)
    ph0 = ph.reshape(N)
    #samp_mask = samp_mask.reshape(N)
    #import pdb; pdb.set_trace()
    
    xdata = tf.fft2c(x0,ph0,axes=(-2,-1))
    #import pdb; pdb.set_trace()
    grad = tf.ifft2c(2 * k * (xdata - data),ph0,axes=(-2,-1))
   
    #for kk in range(N[0]):
        #x_data = tf.fft2c(x0[kk,:,:],ph0[kk,:,:],sz)
    
        #grad[kk,:,:] = -2*tf.ifft2c(samp_mask[kk,:,:]*(data_from_scanner[kk,:,:] - x_data),ph0[kk,:,:],sz=sz).real; # -1* & ,real
    return grad
    

def gTV(x, N, strtag, kern, dirWeight, dirs=None, nmins=0, dirInfo=[None,None,None], a=10):

    if nmins:
        Ahat = dirInfo[0]
        dI = dirInfo[1]
        inds = dirInfo[2]
    else:
        Ahat = None
        dI = None
        inds = None
    
    if len(x.shape) == 2:
        N = np.hstack([1,N])
        
    x0 = x.reshape(N)
    grad = np.zeros(np.hstack([N[0], len(strtag), N[1:]]))#,dtype=complex)
    if kern.shape == 3:
        Nkern = np.hstack(1,[kern.shape[1:]])
    else:
        Nkern = kern.shape[1:]
        
    TV_data = tf.TV(x0,N,strtag,kern,dirWeight,dirs,nmins,dirInfo).real
    for i in xrange(len(strtag)):
        if strtag[i] == 'spatial':
            kernHld = np.flipud(np.fliplr(kern[i])).reshape(Nkern)
            grad[:,i,:,:] = correlate(np.tanh(a*TV_data[i]),kernHld,mode='wrap')
            #grad[:,i,:,:] = np.tanh(a*TV_data[i])
            #grad[:,i,:,:] = fftconvolve(np.tanh(a*TV_data[i]),kern[i].reshape(Nkern),mode='same')
        elif strtag[i] == 'diff':
            for nDir in xrange(N[0]):
                rows = np.where(dI[nDir] == 1)[0]
                dDir = np.zeros([len(rows) + 1, Ahat.shape[1], N[-2]*N[-1]],complex)
                r = inds[nDir,:]
                Iq = x0[nDir,:]
                Ir = x0[inds[nDir],:]
                Irq = (Ir - Iq).reshape(nmins,-1)
                dDir[0] = -1*np.tanh(a*np.dot(Ahat[nDir],Irq))
                for q in range(1,dDir.shape[0]):
                    #import pdb; pdb.set_trace()
                    diffHld = x0[nDir].flatten() - x0[rows[q-1]].flatten()
                    dDir[q] = np.tanh(a*np.dot(Ahat[rows[q-1]],np.dot(dI[nDir,rows[q-1],:].reshape(nmins,1),diffHld.reshape(1,-1))))
                grad[nDir,i,:,:] = np.sum(dDir,axis=(0,1)).reshape(N[-2],N[-1])
                
            
    grad = np.sum(grad,axis=1)
    return grad
    
    
# def gTV(x, N, strtag, dirWeight, dirs=None, nmins=0, dirInfo=[None,None,None,None], a=10):
#     #import pdb; pdb.set_trace()
#     if nmins:
#         M = dirInfo[0]
#         dIM = dirInfo[1]
#         Ause = dirInfo[2]
#         inds = dirInfo[3]
#     else:
#         M = None
#         dIM = None
#         Ause = None
#         inds = None
#
#     if len(x.shape) == 2:
#         N = np.hstack([1,N])
#
#     x0 = x.reshape(N)
#     grad = np.zeros(np.hstack([N[0], len(strtag), N[1:]]))
#
#     TV_data = tf.TV_old(x0,N,strtag,dirWeight,dirs,nmins,dirInfo)
#     for i in xrange(len(strtag)):
#         if strtag[i] == 'spatial':
#             TV_dataRoll = np.roll(TV_data[i,:,:],1,axis=i)
#             #import pdb; pdb.set_trace()
#             grad[:,i,:,:] = -np.tanh(a*(TV_data[i,:,:])) + np.tanh(a*(TV_dataRoll))
#             #grad[i,:,:] = -np.sign(TV_data[i,:,:]) + np.sign(TV_dataRoll)
#         elif strtag[i] == 'diff':
#             for d in xrange(N[i]):
#                 dDirx = np.zeros(np.hstack([N,M.shape[1]])) # dDirx.shape = [nDirs,imx,imy,nmins]
#                 for ind_q in xrange(N[i]):
#                         for ind_r in xrange(M.shape[1]):
#                             dDirx[ind_q,:,:,ind_r] = x0[inds[ind_q,ind_r],:,:].real - x0[ind_q,:,:].real
#                 for comb in xrange(len(Ause[d])):
#                         colUse = Ause[d][comb]
#                         for qr in xrange(M.shape[1]):
#                             grad[d,i,:,:] += np.dot(dIM[d,qr,colUse],dDirx[d,:,:,qr])
#             grad[:,i,:,:] *= dirWeight
#
#     #import pdb; pdb.set_trace()
#     #grad = np.sum(grad,axis=1)
#     return grad
