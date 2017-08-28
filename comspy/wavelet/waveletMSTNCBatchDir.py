#!/usr/bin/env python
# Imports
from __future__ import division
import numpy as np
import scipy as sp
import os.path
import matplotlib.pyplot as plt
import sys
from sys import path as syspath
syspath.append("/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/")
syspath.append("/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/source/")
syspath.append("/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/source/wavelet")
os.chdir('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/')
import transforms as tf
import scipy.ndimage.filters
import gradWaveletMS as grads
import sampling as samp
import direction as d
import scipy.optimize as opt
import optimizationFunctions as funs
import read_from_fid as rff
import saveFig
import visualization as vis
from scipy.ndimage.filters import gaussian_filter
from time import gmtime, strftime
#from pyminc.volumes.factory import *

import argparse

parser = argparse.ArgumentParser(description=
'''
Batch solutions of the CS example.

Produce solutions on a *per slice* basis for the reconstruction of undersampled scans. 
''')

parser.add_argument('filename', metavar='filename', type=str, help='The file where we can find the fully sampled dataset -- expects .npy')
parser.add_argument('dirFile', metavar='gradlist', type=str, help='The list of the directions we are using in cartesian coords')
# WILL NEED TO ADD AN AXIS CHOICE SO THAT WE CAN SPECIFY WHICH AXIS IS THE RO
parser.add_argument('pctg', metavar='samplePerc', type=float, help='The percentage that we are pseudo-sampling. This is 1/R')
parser.add_argument('xtol', metavar='tolerance', type=float, help='Accepted tolerance for the solver')
# Should add in a "solver" argument that allows a choice of solver
parser.add_argument('TV', metavar='lam_TV', type=float, help='Tunable term to influence strength of the TV term in the objective function')
parser.add_argument('XFM', metavar='lam_XFM', type=float, help='Tunable term to influence strength of the XFM term in the objective function')
parser.add_argument('dirWeight', metavar='lam_DIR', type=float, help='Tunable term to influence strength of the DIRECTION term in the objective function -- this is relative to the spatial TV terms')
parser.add_argument('lam_trust', metavar='lam_TRUST', type=float, help='Tunable term to influence strength of how much we "trust" the data that we get from the multi-step solution')
args = parser.parse_args()

np.random.seed(534)
plt.rcParams['image.cmap'] = 'gray'
f = funs.objectiveFunction
df = funs.derivativeFunction

#filename = '/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/brainData/P14/data/fullySampledBrain.npy'
#sliceChoice = 150
#strtag = ['','spatial', 'spatial']
#xtol = 0.001
#TV = 0.0025
#XFM = 0.0025
#dirWeight = 0
#lam_trust = 0.5
#SNR = 30
#filename = '/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/data/120dir_DTI_Phantom-SNR' + str(int(SNR)) + '-blur.npy'
#dirFile =  '/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/gradvecs/120dirAnneal.txt'
#pctg = 0.25
#strtag = ['diff','spatial', 'spatial']
#xtol = 0.0001
#TV = 0.001
#XFM = 0.001
#dirWeight = 1
#lam_trust = 0.7

if __name__ == "__main__":
    filename = args.filename
    dirFile = args.dirFile
    pctg = args.pctg
    xtol = args.xtol
    TV = args.TV
    XFM = args.XFM
    dirWeight = args.dirWeight
    lam_trust = args.lam_trust
    
    strtag = ['diff','spatial', 'spatial']
    ItnLim = 30
    lineSearchItnLim = 30
    wavelet = 'db4'
    mode = 'per'
    
    kern = np.zeros([3,3,3,3])
    for i in range(kern.shape[0]):
        kern[i,1,1,1] = -1

    kern[0,2,1,1] = 0
    kern[1,1,2,1] = 1
    kern[2,1,1,2] = 1
    
    if dirFile:
        nmins = 5
        dirs = np.loadtxt(dirFile)
        dirInfo = d.calcAMatrix(dirs,nmins)
    #dirInfo = [None]*3
    radius = 0.2
    pft=False
    alpha_0 = 0.1
    c = 0.6
    a = 10.0

    pctg = 0.25
    radius = 0.2

    im = np.load(filename)
    N = np.array(im.shape)  # image Size
    szFull = im.size

    P = 2
    nSteps = 4

    pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=[1, N[-2], N[-1]], style='mult', pft=pft,ext=0.5)
    
    #k = np.zeros([dirs.shape[0], 294, 294])
    #pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=[1, N[-2], N[-1]], style='mult', pft=pft,ext=0.5)
    #k = d.dirPDFSamp([int(dirs.shape[0]), 294, 294], P=2, pctg=0.25, radius=0.2, dirs=dirs, cyl=True, taper=0.25)
    #k = np.load('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/data/comb_k_25.npy')
    k = np.zeros([dirs.shape[0], N[-2], N[-1]])
    pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=[1, N[-2], N[-1]], style='mult', pft=pft,ext=0.5)
    k = d.dirPDFSamp([int(dirs.shape[0]), N[-2], N[-1]], P=2, pctg=0.25, radius=0.2, dirs=dirs, cyl=True, taper=0.25)[0]

    
    if len(N) == 2:
        N = np.hstack([1, N])
        k = k.reshape(N)
        im = im.reshape(N)
    elif len(N) == 3:
        if k.ndim == 2:
            k = k.reshape(np.hstack([1,N[-2:]])).repeat(N[0],0)

        
    # Now we initialize to build up "what we would get from the
    # scanner" -- as well as our phase corrections
    #ph_scan = np.zeros(N, complex)
    #data = np.zeros(N,complex)
    #dataFull = np.zeros(N,complex)

    # We need to try to make this be as efficient and accurate as 
    # possible. The beauty of this, is if we are using data that is
    # anatomical, we can use the RO direction as well
    # NOTE: Something that we can do later is make this estimation of
    # phase inclue the RO direction, and then do a split later. This is 
    # post-processing, but pre-CS
    #k = np.fft.fftshift(k, axes=(-2,-1))
        
    ph_ones = np.ones(N, complex)
    dataFull = np.fft.fftshift(tf.fft2c(im, ph=ph_ones,axes=(-2,-1)),axes=(-2,-1))
    data = k*dataFull
    #k = np.fft.fftshift(k, axes=(-2,-1))
    #im_scan_wph = tf.ifft2c(data,ph=ph_ones)
    #ph_scan = np.angle(gaussian_filter(im_scan_wph.real,0) +  1.j*gaussian_filter(im_scan_wph.imag,0))
    #ph_scan = np.exp(1j*ph_scan)
    #im_scan = tf.ifft2c(data,ph=ph_scan,sz=szFull)


    # Now, we can use the PDF (for right now) to make our starting point
    # NOTE: This won't be a viable method for data that we undersample
    #       because we wont have a PDF -- or if we have uniformly undersampled
    #       data, we need to come up with a method to have a good SP
    pdfDiv = pdf.copy()
    pdfZeros = np.where(pdf==0)
    pdfDiv[pdfZeros] = 1


    # Here, we look at the number of "steps" we want to do and step 
    # up from there. The "steps" are chose based on the percentage that 
    # we can sample and is based on the number of steps we can take.
    x, y = np.meshgrid(np.linspace(-1,1,N[-1]),np.linspace(-1,1,N[-2]))
    locs = (abs(x)<=radius) * (abs(y)<=radius)
    minLoc = np.min(np.where(locs==True))
    pctgSamp = np.zeros(minLoc+1)
    for i in range(1,minLoc+1):
        kHld = k[0,i:-i,i:-i]
        pctgSamp[i] = np.sum(kHld)/kHld.size
    pctgLocs = np.arange(1,nSteps+1)/(nSteps)
    locSteps = np.zeros(nSteps)
    locSteps[0] = minLoc
    for i in range(nSteps):
        locSteps[i] = np.argmin(abs(pctgLocs[i]-pctgSamp))
    # Flip it here to make sure we're starting at the right point
    locSteps = locSteps[::-1].astype(int)
    locSteps = np.hstack([locSteps,0])

    for j in range(nSteps+1):
        # we need to now step through and make sure that we 
        # take care of all the proper step sizes
        NSub = np.array([N[0], N[1]-2*locSteps[j], N[2]-2*locSteps[j]]).astype(int)
        ph_onesSub = np.ones(NSub, complex)
        ph_scanSub = np.zeros(NSub, complex)
        dataSub = np.zeros(NSub,complex)
        im_scanSub = np.zeros(NSub,complex)
        im_FullSub = np.zeros(NSub,complex)
        kSub = np.zeros(NSub)
        if locSteps[j]==0:
            kSub = k.copy()
            dataSub = np.fft.fftshift(kSub*dataFull,axes=(-2,-1))
            im_FullSub = tf.ifft2c(np.fft.fftshift(dataFull,axes=(-2,-1)),ph=ph_onesSub,sz=szFull)
        else:
            kSub = k[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
            dataSub = np.fft.fftshift(kSub*dataFull[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]],axes=(-2,-1))
            im_FullSub = tf.ifft2c(np.fft.fftshift(dataFull[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]],axes=(-2,-1)),ph=ph_onesSub,sz=szFull)
                
        im_scan_wphSub = tf.ifft2c(dataSub, ph=ph_onesSub, sz=szFull)
        ph_scanSub = np.angle(gaussian_filter(im_scan_wphSub.real,0) +  1.j*gaussian_filter(im_scan_wphSub.imag,0))
        #ph_scanSub[i,:,:] = tf.matlab_style_gauss2D(im_scan_wphSub,shape=(5,5))
        ph_scanSub = np.exp(1j*ph_scanSub)
        im_scanSub = tf.ifft2c(dataSub, ph=ph_scanSub, sz=szFull)
        
        if j == 0:
            kMasked = kSub.copy()
        else:
            kHld = np.zeros(NSub)
            for msk in range(N[0]):
                padMask = tf.zpad(np.ones(kMasked[msk].shape),NSub[-2:])
                kHld[msk] = ((1-padMask)*kSub[msk] + padMask*tf.zpad(kMasked[msk],NSub[-2:])).reshape(np.hstack([1,NSub[-2:]]))
            kMasked = kHld
        #kMasks.append(kMasked)
        
        # Now we need to construct the starting point
        if locSteps[j]==0:
            pdfDiv = pdf.copy()
        else:
            pdfDiv = pdf[locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
        pdfZeros = np.where(pdfDiv < 1e-4)
        pdfDiv[pdfZeros] = 1
        pdfDiv = pdfDiv.reshape(np.hstack([1,NSub[-2:]])).repeat(NSub[0],0)
        
        N_imSub = NSub
        hldSub, dimsSub, dimOptSub, dimLenOptSub = tf.wt(im_scanSub[0].real,wavelet,mode)
        NSub = np.hstack([N_imSub[0], hldSub.shape])
        
        w_scanSub = np.zeros(NSub)
        im_dcSub = np.zeros(N_imSub)
        w_dcSub = np.zeros(NSub)
        
        if j == 0:
            data_dc = np.zeros(N_imSub,complex)
        else:
            data_dc_hld = np.zeros(N_imSub,complex)
            for i in range(N_imSub[0]):
                data_dc_hld[i] = tf.zpad(np.fft.fftshift(data_dc[i],axes=(-2,-1)),N_imSub[-2:])*(1-kSub[i])
            data_dc = np.fft.fftshift(data_dc_hld,axes=(-2,-1))
            
        dataSub += data_dc
        dataSub = d.dirDataSharing(dataSub,dirs,nmins=nmins)
        im_dcSub = tf.ifft2c(dataSub / np.fft.ifftshift(pdfDiv,axes=(-2,-1)), ph=ph_scanSub, axes=(-2,-1)).real.copy()
        for i in xrange(N[0]):
            w_scanSub[i,:,:] = tf.wt(im_scanSub.real[i],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0]
            w_dcSub[i,:,:] = tf.wt(im_dcSub[i],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0]
            
        kSub = np.fft.fftshift(kSub,axes=(-2,-1))
        w_dcSub = w_dcSub.flatten()
        im_spSub = im_dcSub.copy().reshape(N_imSub)
        dataSub = np.ascontiguousarray(dataSub)
        
        mets = ['Density Corrected']#,'Zeros','Ones','Random']
        wdcs = []
        
        #if (j!=0) and (locSteps[j]!=0):
            #kpad = tf.zpad(kStp[0],np.array(kSub.shape[-2:]).astype(int))
            #data_dc = np.fft.fftshift(tf.zpad(np.ones(kStp.shape[-2:])*dataStp[0], np.array(kSub.shape[-2:]).astype(int)) + (1-kpad)*np.fft.fftshift(dataSub))
            #im_dcSub = tf.ifft2c(data_dc[i,:,:] / np.fft.ifftshift(pdfDiv), ph=ph_scanSub[i,:,:]).real.copy().reshape(N_imSub)
        imdcs = [im_dcSub] #,np.zeros(N_im),np.ones(N_im),np.random.randn(np.prod(N_im)).reshape(N_im)]
        #import pdb; pdb.set_trace()
        
        kSamp = np.fft.fftshift(kMasked,axes=(-2,-1))
        
        args = (NSub, N_imSub, szFull, dimsSub, dimOptSub, dimLenOptSub, TV, XFM, dataSub, kSamp, strtag, ph_scanSub, kern, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
        w_result = opt.fmin_tnc(f, w_scanSub.flat, fprime=df, args=args, accuracy=xtol, disp=0)
        w_dc = w_result[0].reshape(NSub)
        #stps.append(w_dc)
        #w_stp.append(w_dc.reshape(NSub))
        #im_hld = np.zeros(N_imSub)
        #for i in range(NSub[0]):
            #im_hld[i] = tf.iwt(w_stp[-1][i],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)
        #imStp.append(im_hld)
        #plt.imshow(imStp[-1],clim=(minval,maxval)); plt.colorbar(); plt.show()
        #w_dc = w_stp[k].flatten()
        #stps.append(w_dc)
        #wdcHold = w_dc.reshape(NSub)
        #dataStp = np.fft.fftshift(tf.fft2c(imStp[-1],ph_scanSub),axes=(-2,-1))
        #kStp = np.fft.fftshift(kSub,axes=(-2,-1)).copy()
        #kMaskRpt = kMasked.reshape(np.hstack([1,N_imSub[-2:]])).repeat(N_imSub[0],0)
        im_hld = np.zeros(N_imSub)
        for i in range(NSub[0]):
            im_hld[i] = tf.iwt(w_dc[i],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)
        data_dc = tf.fft2c(im_hld, ph=ph_scanSub, axes=(-2,-1))
        kMasked = (np.floor(1-kMasked)*pctgSamp[locSteps[j]]*lam_trust + kMasked)
        #kMasked = (np.floor(1-kMasked)*pctgSamp[locSteps[j]]*1.0 + kMasked).reshape(np.hstack([1, N_imSub[-2:]]))

    
    
    wHold = w_dc.copy().reshape(NSub)
    imHold = np.zeros(N_imSub)

    for i in xrange(N[0]):
        imHold[i,:,:] = tf.iwt(wHold[i,:,:],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)
        
    SNR = filename.split('SNR')[1].split('-')[0]
    np.save('temp/' + str(int(100*pctg)) + '_TV_' + str(TV) + '_XFM_' + str(XFM) + '_lamTrust_' + str(int(100*lam_trust)) + '_xtol_' + str(xtol) + '_dirWeight_' + str(dirWeight) + '_dirs_' + str(dirs.shape[0]) + '_SNR_' + SNR + '.npy',imHold)