from __future__ import division
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import os.path
os.chdir('/home/asalerno/Documents/pyDirectionCompSense/')
import transforms as tf
import scipy.ndimage.filters
import grads
import sampling as samp
import direction as d
#from scipy import optimize as opt
import optimize as opt

filename = '/home/asalerno/Documents/pyDirectionCompSense/data/SheppLogan256.npy'
strtag = ['spatial','spatial']
TVWeight = 0.01
XFMWeight = 0
dirWeight = 0
#DirType = 2
ItnLim = 150
epsilon = 1e-3
l1smooth = 1e-15
xfmNorm = 1
scaling_factor = 4
L = 2
method = 'CG'
dirFile = None
nmins = None

np.random.seed(2000)
im = np.load(filename)

for i in range(len(strtag)):
    strtag[i] = strtag[i].lower()
    
N = np.array(im.shape) #image Size
tupleN = tuple(N)
pctg = 0.25 # undersampling factor
P = 5 # Variable density polymonial degree
ph = tf.matlab_style_gauss2D(im,shape=(5,5));

pdf = samp.genPDF(N,P,pctg,radius = 0.1,cyl=[0]) # Currently not working properly for the cylindrical case -- can fix at home
# Set the sampling pattern -- checked and this gives the right percentage
k = samp.genSampling(pdf,10,60)[0].astype(int)

# Diffusion information that we need
if dirFile:
    dirs = np.loadtxt(dirFile)
    M = d.calc_Mid_Matrix(dirs,nmins=4)
else:
    dirs = None
    M = None

# Here is where we build the undersampled data
data = np.fft.ifftshift(k)*tf.fft2c(im,ph)
#ph = phase_Calculation(im,is_kspace = False)
#data = np.fft.ifftshift(np.fft.fftshift(data)*ph.conj());

# IMAGE from the "scanner data"
im_scan = tf.ifft2c(data,ph)

# Primary first guess. What we're using for now. Density corrected
im_dc = tf.ifft2c(data/np.fft.ifftshift(pdf),ph).flatten().copy()

# Optimization algortihm -- this is where everything culminates together
im_result = opt.minimize(optfun, im_dc, args = (N,TVWeight,XFMWeight,data,k,strtag,dirWeight,dirs,M,nmins,scaling_factor,L,ph),method=method,jac=derivative_fun,options={'maxiter':ItnLim,'gtol':epsilon,'disp':1})

# im_result gives us a lot of information, what we really need is ['x'] reshaped to the required image size -- N
return im_result, im_dc, im_scan