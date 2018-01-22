from __future__ import division
import numpy as np
import comspy as cs
import comspy.transforms as tf
import sys, os

def preobjective(x,dx, N, N_im, dims, dimOpt, dimLenOpt, lam1, lam2, data, k, strtag, ph, 
                       kern, dirWeight=0.1, dirs=None, dirInfo=[None,None,None,None], nmins=0, wavelet="db4", mode="per", level=3, a=10.):
    x.shape = N
    dx.shape = N
    data.shape = N_im
    FTXFMtx = np.zeros(N_im,complex)
    FTXFMtdx = np.zeros(N_im,complex)
    DXFMtx = np.zeros(N_im,complex)
    DXFMtdx = np.zeros(N_im,complex)
    for i in range(N[0]):
        FTXFMtx = k*tf.fft2c(tf.iwt(x[i],wavelet,mode,dims,dimOpt,dimLenOpt),ph_scan)
        FTXFMtdx = k*tf.fft2c(tf.iwt(dx[i],wavelet,mode,dims,dimOpt,dimLenOpt),ph_scan)
        DXFMtx = tf.TV(tf.iwt(x[i],wavelet,mode,dims,dimOpt,dimLenOpt), N, strtag, kern, dirWeight, dirs, nmins, dirInfo)
        DXFMtdx = tf.TV(tf.iwt(dx[i],wavelet,mode,dims,dimOpt,dimLenOpt), N, strtag, kern, dirWeight, dirs, nmins, dirInfo);
    x.shape = x.size
    dx.shape = dx.size
    data.shape = data.size
    return FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx

def objective(FTXFMtx,FTXFMtdx,DXFMtx,DXFMtdx, N, N_im, dims, dimOpt, dimLenOpt, lam1, lam2, data, k, strtag, ph, 
                       kern, dirWeight=0.1, dirs=None, dirInfo=[None,None,None,None], nmins=0, wavelet="db4", mode="per", level=3, a=10.):
    p = 1
    obj = FTX