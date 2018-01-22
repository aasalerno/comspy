from __future__ import division
import numpy as np
import comspy as cs
import matplotlib.pyplot as plt
import comspy.transforms as tf
import sys, os

strtag = ['','spatial', 'spatial']
dirWeight = 0
dirData = False
nDir = None
#filename = '/hpf/largeprojects/MICe/asalerno/comspyData/phantom/phantom.npy'
filename = '/hpf/largeprojects/MICe/asalerno/comspyData/brainData/P14/data/sl190_fullySampledBrain.npy'
dirFile = None
nmins = None

ItnLim = 30
lineSearchItnLim = 30
method = 'CG'

P=2
pctg=0.25
radius=0.2
cyl=[1,256,256]
style='mult'
pft=False
ext=0.5


wavelet = 'db3'
mode = 'per'
level = 4
kern = np.zeros([3,3,3,3])
for i in range(kern.shape[0]):
    if strtag[i] == 'spatial':
        kern[i,1,1,1] = -1

if strtag[0] == 'spatial':
    kern[0,2,1,1] = 1
if strtag[1] == 'spatial':
    kern[1,1,2,1] = 1
if strtag[2] == 'spatial':
    kern[2,1,1,2] = 1

dirFile = None
nmins = None
dirs = None
M = None
dirInfo = [None]*4
radius = 0.2
pft=False
alpha_0 = 0.01
c = 0.6
a = 1.0 # value used for the tanh argument instead of sign

pctg = 0.25
phIter = 0
sliceChoice = 150
xtol = 1e-4#[1e-2, 1e-2, 1e-3, 5e-4, 5e-4]
TV = 0.01#[0.005, 0.005, 0.002, 0.001, 0.001]
XFM = 0.01#[0.005, 0.005, 0.002, 0.001, 0.001]
radius = 0.2





im, N = cs.preRecon.read_and_normalize(filename)
im, N = cs.preRecon.direction_swap_axes(im, N, dirData, nDir)
if len(N) == 2:
    N = np.hstack([1, N])
    im.shape = N

if dirData:
    dirs, dirInfo = cs.preRecon.read_directional_data(dirFile,nmins)
else:
    dirs = None
    dirInfo = [None]*4
    
dataFull, data, datadc, pdf, k, im_scan, ph_scan = cs.preRecon.create_scanner_k_space(im, N, P=P,
                                                                                      pctg=pctg,
                                                                                      dirData=dirData,
                                                                                      dirs=dirs,
                                                                                      radius=radius,
                                                                                      cyl=cyl, 
                                                                                      style=style, 
                                                                                      pft=pft, 
                                                                                      ext=ext)

# Since we're working with data that's artifical and we always work in cyl
x,y = np.meshgrid(np.linspace(-1,1,N[-2]),np.linspace(-1,1,N[-1]))
r = x**2 + y**2 
r[r<=1] = 1
r[r>1] = 0
dataFull = dataFull*r
# Here is where we'd put the functions
w_im_scan, dims, dimOpt, dimLenOpt = tf.wt(im_scan,wavelet,mode,level)
w_im, dims, dimOpt, dimLenOpt = tf.wt(im,wavelet,mode,level)

## test to show that wavelet and FT dont change images
#for i in range(10):
    #if i%2:
        #w_im2 = tf.wt(im_scan,wavelet,mode,level)[0]
        #data2 = tf.fft2c(im_scan2,ph=ph_scan)
        #if np.max(abs(w_im2 - w_im)) > 1e-9:
            #print('WT FAILED')
        #elif np.max(abs(data2 - data)) > 1e-9:
            #print('FFT FAILED')
    #else:
        #im2 = tf.iwt(w_im_scan,wavelet,mode,dims,dimLen,dimLenOpt)
        #im_scan2 = tf.ifft2c(data,ph=ph_scan)
        #if np.max(abs(im2 - im_scan)) > 1e-9:
            #print('IWT FAILED')
        #elif np.max(abs(im_scan2 - im_scan)) > 1e-9:
            #print('IFFT FAILED')
        
            

Nw = np.hstack([N[0], w_im.shape])

im_sp = np.ones(N)
data_sp = tf.fft2c(im_sp, ph=ph_scan, axes=(-2,-1))

w_sp = np.zeros(Nw,complex)
for i in range(N[0]):
    w_sp[i] = tf.wt(im_sp.real[i],wavelet,mode,level,dims,dimOpt,dimLenOpt)[0]
lam1=0.0005
lam2=0.0005
args = (Nw, N, dims, dimOpt, dimLenOpt, lam1, lam2, data, k, strtag, ph_scan, kern, dirWeight, dirs, dirInfo, nmins, wavelet, mode, level, a)


## ----- DEBUG -----#
## Objective Functions
#dc = cs.objectiveFunctions.objectiveFunctionDataCons(im_scan, N, ph_scan, data, k)
#tv = cs.objectiveFunctions.objectiveFunctionTV(im_scan, N, strtag, kern, dirWeight, 
                                               #dirs, nmins, dirInfo, a)
#xfm = cs.objectiveFunctions.objectiveFunctionXFM(w_im_scan, a)
#f = cs.objectiveFunctions.f

## Gradient Functions
gdc = cs.grads.gDataCons(im_scan, N, ph_scan, data, k)
gtv = cs.grads.gTV(im_scan, N, strtag, kern, dirWeight, dirs, nmins, dirInfo, a)
gxfm = cs.grads.gXFM(w_im_scan, a)
#df = cs.objectiveFunctions.df

## Transforms
#TV = tf.TV(im_scan,N,strtag,kern)
#XFM = tf.wt(im_scan,wavelet,mode,level)[0]


# LUSTIGS OPTIMIZER #
x = w_sp.copy()
f = cs.objectiveFunctions.f
df = cs.objectiveFunctions.df

t0 = 1
t = t0
beta = 0.6
alpha = 0.01
g0 = df(w_sp,*args)
dx = -g0

f0 = f(x,*args)
f1 = f(x+t*dx,*args)
maxlsiter = 10

for i in range(10):
    t = t0
    lsiter = 0
    
    while (f1 > (f0 - alpha*t*np.sum(dx**2))) and (lsiter < maxlsiter):
        lsiter += 1
        t = t*beta
        f1 = f(x+t*dx,*args)
    
    if lsiter == maxlsiter:
        print('LSITER MAX REACHED')
    if lsiter > 2:
        t0 = t*beta
    elif lsiter < 1:
        t0 = t/beta
        
    x = x + t*dx
    g1 = df(x,*args)
    bk = np.sum(g1**2)/np.sum(dx**2+1e-15)
    g0 = g1.copy()
    dx = -g1 + bk*dx
    
w_result = x.reshape(Nw)
im_result = np.zeros(N)
for i in xrange(N[0]):
    im_result[i,:,:] = tf.iwt(w_result[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)
    
#plt.imshow(im_result[0],clim=(0,1))
plt.ion()
plt.figure()
plt.subplot(131)
plt.imshow(im_result[0],clim=(0,1))
plt.title('Result')
plt.subplot(132)
plt.imshow(im_scan.real[0],clim=(0,1))
plt.title('Scanner')
plt.subplot(133)
plt.imshow(abs(im[0]))
plt.title('Fully Sampled')
