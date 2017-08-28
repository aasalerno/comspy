# !/usr/bin/env python -tt
#
#
#   transforms.py
#
#
#
#   Code to be a wrapper for all of the transforms that are done in order to clean up the central codes

from __future__ import division
import numpy as np
import numpy.fft as fft
import pywt
import direction as d
#from scipy.ndimage.filters import correlate
from scipy.signal import correlate

EPS = np.finfo(float).eps



def fft2c(im_to_fft, ph, norm="ortho", axes=(-2,-1)):
    FFTdata = fft.fft2(im_to_fft*ph,axes=axes,norm=norm);
    return FFTdata

def ifft2c(data_to_ifft, ph, norm="ortho", axes=(-2,-1)):
    if sz is None:
        IFFTdata = fft.ifft2(data_to_ifft,axes=axes,norm=norm)*np.conj(ph);
    return IFFTdata
    
def fftnc(im_to_fft, ph, norm="ortho", axes=(-2,-1)):
    FFTdata = fft.fftn(im_to_fft*ph,axes=axes,norm=norm);
    return FFTdata

def ifftnc(data_to_ifft, ph, norm="ortho", axes=(-2,-1)):
    IFFTdata = fft.ifftn(data_to_ifft,axes=axes,norm=norm)*np.conj(ph);
    return IFFTdata

def xfm(data_to_xfm,wavelet = 'db4',mode='per'):
    XFMdata = pywt.wavedec2(data_to_xfm.reshape(data_to_xfm.shape[-2:]),wavelet,mode)
    return XFMdata

def ixfm(data_to_ixfm,wavelet = 'db4',mode='per'):
    IXFMdata = pywt.waverec2(data_to_ixfm,wavelet,mode)
    return IXFMdata

def wt(data_to_wt, wavelet='db4', mode='per', dims=None, dimOpt=None, dimLenOpt=None):
    return wvlt2mat(xfm(data_to_wt, wavelet, mode), dims, dimOpt, dimLenOpt)

def iwt(data_to_iwt, wavelet='db4', mode='per', dims=None, dimOpt=None, dimLenOpt=None):
    return ixfm(mat2wvlt(data_to_iwt, dims, dimOpt, dimLenOpt), wavelet, mode)

def wtn(data_to_wt, wavelet='db4', mode='per', dims=None, dimOpt=None, dimLenOpt=None):
    xfmData = pywt.wavedecn(data_to_wt,wavelet,mode)
    return wvlt2mat(xfmData, dims, dimOpt, dimLenOpt)

def iwtn(data_to_iwt, wavelet='db4', mode='per', dims=None, dimOpt=None, dimLenOpt=None):
    return pywt.waverecn(mat2wvlt(data_to_iwt, dims, dimOpt, dimLenOpt), wavelet, mode)

    
# -------------------------------------------- #
# ---------------- NEW TV -------------------- #
# -------------------------------------------- #


def TV(im, N, strtag, kern, dirWeight=1, dirs=None, nmins=0, dirInfo=[None,None,None]):
    res = np.zeros(np.hstack([len(strtag), im.shape]))
    inds = dirInfo[2]
    Nkern = np.hstack(kern.shape[1:])
    for i in xrange(len(strtag)):
        if strtag[i] == 'spatial':
            #res[i] = correlate(im,kern[i].reshape(Nkern),mode='wrap')
            res[i] = correlate(im,kern[i].reshape(Nkern),mode='same')
        elif strtag[i] == 'diff':
            dirDataHold = dirWeight*d.least_Squares_Fitting(im,N,strtag,dirs,inds,dirInfo[0]).real
            res[i] = np.sqrt(np.sum(dirDataHold**2,axis=1)).reshape(N)
    return res.reshape(np.hstack([len(strtag), N]))
    

def TV_old(im, N, strtag, dirWeight=1, dirs=None, nmins=0, dirInfo=[None,None,None]):
    
    '''
    A finite differences sampling operation done on datasets to spply some 
    smoothing techniques.
    
    Note that the output comes back such that the stacking dimension is dimension 0
    '''
    #axisvals = []
    #for i in xrange(len(strtag)):
        #if strtag[i].lower() == 'spatial':
            #nstacks += 1
            #axisvals.append(cnt)
            #cnt += 1
        #elif strtag[i].lower() == 'diff':
            #nstacks += nmins
            #axisvals.append(0)
    
    #import pdb; pdb.set_trace()
    res = np.zeros(np.hstack([len(strtag), im.shape]))
    #im = np.squeeze(im)
    inds = dirInfo[2]
    cnt = 0
    for i in xrange(len(strtag)):
        if strtag[i] == 'spatial':
            #res[cnt,:,:] = np.roll(data,1,axis = axisvals[i]) - data
            res[i] = np.roll(im,-1,axis = i) - im
            #cnt += 1
        elif strtag[i] == 'diff':
            #res[cnt:cnt+nmins,:,:] = TVDir(data)
            res[i] = dirWeight*d.least_Squares_Fitting(im,N,strtag,dirs,inds,dirInfo[0]).real
            #cnt += nmins
    
    return res.reshape(np.hstack([len(strtag), N]))

def matlab_style_gauss2D(im,shape=(3,3),sigmaX = 0):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    import cv2
    
    filtdata = cv2.GaussianBlur(im.real*1,shape,sigmaX) + cv2.GaussianBlur(im.imag*1,shape,sigmaX)*1j;
    ph = np.angle(filtdata)
    
    return ph
    
def laplacianUnwrap(P,N,res):
    # P is the wrapped phase 3d volume, ny nx and nz are the matrix dimensions, Ly Lx Lz are
    # the spatial dimensions of the volume in cm.
    
    ph_ones = np.ones(N)

    P = P.reshape(N)
    cP = np.cos(P)
    sP = np.sin(P)
    FcP = fft2c(cP,ph_ones)
    FsP = fft2c(sP,ph_ones)
    Lx = res[0]*N[0]*10000
    Ly = res[1]*N[1]*10000
    [k1,k2]= np.meshgrid(np.arange(-N[1]/2+.5,N[1]/2)/Ly,np.arange(-N[0]/2+.5,N[0]/2)/Ly)
    ksq = k1**2 + k2**2
    iFkFcP = ifft2c(ksq*FcP,ph_ones)
    iFkFsP = ifft2c(ksq*FsP,ph_ones)
    fP = fft2c(cP*iFkFsP - sP*iFkFcP,ph_ones)/np.fft.fftshift(ksq)
    #P = ifft2c(fP,ph_ones).real
    P = np.angle(ifft2c(fP,ph_ones))
    return P
    
#def iDx(data,shp):
#res = data[np.hstack([0,range(shp[0]-1)]),:] - data
#res[0,:] = -data[0,:]
#res[-1,:] = data[-2,:]
#return res

#def iDy(data,shp):
#res = data[:,np.hstack([0,range(shp[1]-1)])] - data
#res[:,0] = -data[:,0]
#res[:,-1] = data[:,-2]
#return res

#def iDz(data,shp):
#res = data[:,:,np.hstack([0,range(shp[2]-1)])] - data
#res[:,:,0] = -data[:,:,0]
#res[:,:,-1] = data[:,:,-2]
#return res

#def iTV(data):
#'''
#Inverse of the finite differences sampling operation done. Attempting to build back
#the data after it's been TV'd
#Note that the input must be put in such that the stacking dimension is dimension 0
#'''

#shp = data.shape
#res = iDx(data[0,:,:],shp[1:])+ iDy(data[1,:,:],shp[1:])

#if len(shp) == 4:
    #res = res + iDz(data[2,:,:,:],shp[1:])
#return res

def fermifilt(rawdata,cutoff=0.98,transwidth=0.06):
    data_shape = N.shape(rawdata)
    nro = data_shape[-1]
    nv1 = data_shape[-2]
    nv2 = data_shape[-3]
    for j in range(nv2):
        r_vals = N.sqrt( (2.0*j/float(nv2)-1.0)**2.0 + \
                    ((2.0*N.arange(nv1)/float(nv1)-1.0)**2.0)[:,N.newaxis] + \
                    ((2.0*N.arange(nro)/float(nro)-1.0)**2.0)[N.newaxis,:]  \
                    )
        filt = 1.0/(1.0+N.exp(-(cutoff-r_vals)/transwidth))
        if len(data_shape)==3:
            rawdata[j,:,:]=(rawdata[j,:,:]*filt).astype(N.complex)
        else:
            rawdata[...,j,:,:]=(rawdata[...,j,:,:]*filt).astype(N.complex)
    return rawdata
    
# ----------- Rearrange Functions ----------- #
    
def zpad(orig_data,res_sz):
    res_sz = np.array(res_sz)
    orig_sz = np.array(orig_data.shape)
    padval = np.ceil((res_sz-orig_sz)/2)
    res = np.pad(orig_data,([int(padval[0]),int(padval[0])],[int(padval[1]),int(padval[1])]),mode='constant')
    return res

def wvlt2mat(wvlt, dims=None, dimOpt=None, dimLenOpt= None):
    if dims is None:
        dims = np.zeros(np.hstack([len(wvlt), len(wvlt[0].shape)]))
        if dims.shape[-1]==2:
            for i in range(len(wvlt)):
                if i == 0:
                    dims[i,...] = wvlt[i].shape
                else:
                    dims[i,...] =  wvlt[i][0].shape
        elif dims.shape[-1]==3:
            wvKeys = wvlt[1].keys()
            for i in range(len(wvlt)):
                if i == 0:
                    dims[i,...] = wvlt[i].shape
                else:
                    dims[i,...] = wvlt[i][wvKeys[0]].shape
    if np.any(dims[0,...] != np.zeros(len(wvlt[0].shape))):
        dims = np.vstack([np.zeros(len(wvlt[0].shape)), dims]).astype(int)
        
    if dimOpt is None:
        dimOpt = np.zeros(np.hstack([len(wvlt), len(wvlt[0].shape)]))
        dimOpt[0,...] = wvlt[0].shape
        for i in range(len(wvlt)):
            dimOpt[i,...] = np.sum(dimOpt,axis=0)
    if np.any(dimOpt[0,...] != np.zeros(len(wvlt[0].shape))):
        dimOpt =  np.vstack([np.zeros(len(wvlt[0].shape)), dimOpt]).astype(int)
        
    if dimLenOpt is None:
        dimLenOpt = np.zeros(dimOpt.shape)
        for i in range(dimOpt.shape[0]):
            dimLenOpt[i,...] = np.sum(dimOpt[0:i+1,...],axis=0)
    dimLenOpt = dimLenOpt.astype(int)
    
    sz = np.sum(dimOpt,axis=0,dtype=int)
    mat = np.zeros(sz,complex)
    
    if dims.shape[-1]==2:
        for i in range(1,dims.shape[0]):
            if i==1: 
                mat[0:dims[i,0],0:dims[i,1]] = wvlt[i-1]
            else: # Here we have to do the other parts, as they are split in three
                mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1]] = wvlt[i-1][0] # to the right
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1]] = wvlt[i-1][1] # below
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1]] = wvlt[i-1][2] # diagonal
    
    elif dims.shape[-1]==3:
        for i in range(1,dims.shape[0]):
            if i==1: 
                mat[0:dims[i,0],0:dims[i,1],0:dims[i,2]] = wvlt[i-1]
            else:
                mat[0:dims[i,0],0:dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]] = wvlt[i-1][wvKeys[0]]
                mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],0:dims[i,2]] = wvlt[i-1][wvKeys[1]]
                mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]] = wvlt[i-1][wvKeys[2]]
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1],0:dims[i,2]] = wvlt[i-1][wvKeys[3]]
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],0:dims[i,2]] = wvlt[i-1][wvKeys[4]]
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]] = wvlt[i-1][wvKeys[5]]
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]] = wvlt[i-1][wvKeys[6]]
        
    return mat, dims, dimOpt, dimLenOpt
        
def mat2wvlt(mat, dims, dimOpt, dimLenOpt):
    wvlt = [[] for i in range(len(dims)-1)]
    for i in range(1,len(wvlt)):
        wvlt[i] = [[] for kk in range(3)]
    if dims.shape[-1]==2:
        for i in range(1,dims.shape[0]):
            if i==1: 
                wvlt[i-1] = mat[0:dims[i,0],0:dims[i,1]]
            else: # Here we have to do the other parts, as they are split in three
                wvlt[i-1][0] = mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1]] # to the right
                wvlt[i-1][1] = mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1]] # below
                wvlt[i-1][2] = mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1]] # diagonal
    elif dims.shape[-1]==3:
        for i in range(1,dims.shape[0]):
            wvKeys=['dad', 'aad', 'daa', 'add', 'ada', 'dda', 'ddd']
            if i==1: 
                wvlt[i-1] = mat[0:dims[i,0],0:dims[i,1],0:dims[i,2]]
            else:
                wvlt[i-1] = {wvKeys[0]: mat[0:dims[i,0],0:dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]],
                wvKeys[1]: mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],0:dims[i,2]],
                wvKeys[2]: mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]],
                wvKeys[3]: mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1],0:dims[i,2]],
                wvKeys[4]: mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],0:dims[i,2]],
                wvKeys[5]: mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]],
                wvKeys[6]: mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]]}
        
    return wvlt

def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array" % (axis, m.ndim))
    return m[tuple(indexer)]
