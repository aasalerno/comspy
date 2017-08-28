from __future__ import division
import numpy as np
import numpy.fft as fft
from scipy.optimize import fmin_tnc
from scipy.ndimage.filters import gaussian_filter
import direction as d
import sampling as samp
import preRecon as pr
import transforms as tf

def wavelet_reconstruct(f, df, x0, N, ph_scan, strtag, method='CG', xtol=1e-6, disp=1, *args):
    '''
    Where the recon actually happens. It requires the f, df to already be defined.
    Currently uses a truncated Newton CG method but will be more useful in future
    '''
    hld, dims, dimOpt, dimLenOpt = tf.wt(x0[0].real,wavelet,mode)
    Nw = np.hstack([N[0], hld.shape])
    w_dc = np.zeros(Nw)
    im0 = tf.ifft2c(x0,ph=ph_scan,axes=(-2,-1))
    for i in range(N[0]):
        w_dc[i] = tf.wt(im0.real[i],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0]
        
    w_result = fmin_tnc(f, w_dc.flat, fprime=df, args=args, accuracy=xtol, disp=disp)[0].reshape(Nw)
    
    for i in xrange(N[0]):
        im_result[i,:,:] = tf.iwt(w_result[i,:,:],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)
    
    return im_result

def multistep_recon(f, df, x0, N, ph_scan, pdf, k, locSteps, strtag, method='CG', xtol=1e-6, disp=1, *args):
    nSteps = len(locSteps)
    
    for j in range(nSteps):
        NSub = np.array([N[0], N[1]-2*locSteps[j], N[2]-2*locSteps[j]]).astype(int)
        if locSteps[j]==0:
            kSub = k.copy()
            dataSub = np.fft.fftshift(kSub*dataFull,axes=(-2,-1))
        else:
            kSub = k[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
            
        im_scan_wphSub = tf.ifft2c(dataSub, ph=ph_onesSub)
        ph_scanSub = np.angle(gaussian_filter(im_scan_wphSub.real,0) + 
                            1.j*gaussian_filter(im_scan_wphSub.imag,0))
        ph_scanSub = np.exp(1j*ph_scanSub)
        im_scanSub = tf.ifft2c(dataSub, ph=ph_scanSub, axes=(-2,-1))

        if j == 0:
            kMasked = kSub.copy()
        else:
            kHld = np.zeros(NSub)
            for msk in range(N[0]):
                padMask = tf.zpad(np.ones(kMasked[msk].shape),NSub[-2:])
                kHld[msk] = ((1-padMask)*kSub[msk] +    
                            padMask*tf.zpad(kMasked[msk],NSub[-2:])).reshape(np.hstack([1,NSub[-2:]]))
            kMasked = kHld
        
        if locSteps[j]==0:
            pdfDiv = pdf.copy()
        else:
            pdfDiv = pdf[locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
            
        pdfZeros = np.where(pdfDiv < 1e-4)
        pdfDiv[pdfZeros] = 1
        pdfDiv = pdfDiv.reshape(np.hstack([1,NSub[-2:]])).repeat(NSub[0],0)
        
        hldSub, dimsSub, dimOptSub, dimLenOptSub = tf.wt(im_scanSub[0].real,wavelet,mode)
        NwSub = np.hstack([NSub[0], hldSub.shape])
        
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
            
        if any('diff' in s for s in strtag):
            data_dc = d.dirDataSharing(data_dc,dirs,nmins=nmins)
            
        dataSub += data_dc
        
        w_result = wavelet_reconstruct(f, df, x0, N, ph_scan, strtag, method='CG', xtol=1e-6, disp=1)