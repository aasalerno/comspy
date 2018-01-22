#!/usr/bin/env python -tt
#
#
# sampling.py
#
#
# We start with the data from the scanner. The inputs are:
#       - inFile (String) -- Location of the data
#                         -- Direct to a folder where all the data is
#       - 
#
from __future__ import division
#import pyminc.volumes.factory
import numpy as np 
import scipy as sp
import scipy.ndimage as ndimage
import sys
import glob
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
#import matplotlib.pyplot as plt
import os.path
from sys import path as syspath
syspath.append('/home/bjnieman/source/vnmr')
from varian_read_file import parse_petable_file
import transforms as tf

EPS = np.finfo(float).eps

def indata(inFolder):
    '''
    This code reads in data from mnc files in order to be worked on via the code. 
    Reads the data in and outputs it as a list
    '''
    us_data = [] 

    # get the names of the input files from the specified folder
    filenames = glob.glob(inFolder + '/*.mnc')

    # Put the data in a large dataset to be worked with
    for files in filenames:
        cnt = 0
        us_data.append(pyminc.volumes.factory.volumeFromFile(files))
        cnt += 1 
    return us_data

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    
def zpad(orig_data,res_sz):
    res_sz = np.array(res_sz)
    orig_sz = np.array(orig_data.shape)
    padval = np.ceil((res_sz-orig_sz)/2)
    res = np.pad(orig_data,([int(padval[0]),int(padval[0])],[int(padval[1]),int(padval[1])]),mode='constant')
    return res
    
def genPDF(img_sz,
        p,
        pctg,
        l_norm = 2,
        radius = 0,
        cyl = [0],
        disp = 0,
        style='mult',
        pft=False,
        ext=0.5):
    """
    Generates a Probability Density Function for Pseudo-undersampling (and potentially for use with the scanner after the fact. 
    
    This uses a variable density undersampling, allowing for more (or less) data in the centre
    
    Input:
        [int list] img_sz - The size of the input dataset
        [int]         p - polynomial power (1/r^p)
        [float]    pctg - Sampling factor -- how much data we want to collect
        [int]    l_norm - L1 or L2 distance measure
        [float]  radius - fully sampled centre radius
        [int list]  cyl - Is it cylindrical data acquisition? 
        [bool]     disp - Do you want it displayed or not?
    
    Output:
        2D[float] pdf - The probability density function
        
    Based on Lustig's work from 2007
    """
    
    minval = 0.0
    maxval = 1.0
    val = 0.5
    
    # Check if we're doing cylindrical data, if so, change img_sz and note that we need to zero pad
    if cyl[0] == 1:
        img_sz_hold = cyl[1:]
        cir = True
        if np.logical_and(img_sz_hold[-2] == img_sz[-2], img_sz_hold[-1] == img_sz[-1]):
            zpad_mat = False
        else:
            zpad_mat = True
    else:
        cir = False
        img_sz_hold = img_sz
        zpad_mat = False
        outcyl = None

    # If the size vector is only one value, add on another value
    if len(img_sz_hold) == 1:
        img_sz_hold = [img_sz_hold, 1]


    # How many of the datapoints do we have to look at?
    sx = img_sz_hold[-2]
    sy = img_sz_hold[-1]
    #import pdb; pdb.set_trace()
    if cir:
        PCTG = int(np.floor(pctg*sx*sy*np.pi/4.))
        if pft:
            h = 1+ext
            r_ch = np.sqrt(1-(h*radius)**2)
            A_T = h*r_ch
            theta = np.arccos(ext)
            A_p0 = 1-(2*theta-A_T)/np.pi
            PCTG = int(PCTG*A_p0)
    else:
        PCTG = int(np.floor(pctg*sx*sy))

    if np.sum(np.array(img_sz_hold == 1,dtype='int')) == 0: #2D case
        [x,y] = np.meshgrid(np.linspace(-1,1,sy),np.linspace(-1,1,sx))
        if l_norm == 1:
            r = abs(np.array([x,y])).max(0)
        elif l_norm == 2:
            r = np.sqrt(x**2 + y**2)
            # Check if the data is cyl acquisition -- if so, make sure outside datapoints don't get chosen by setting r = 1
            if cir:
                outcyl = np.where(r > 1)
                r[outcyl] = 1
            else:
                r = r/r.max()
    else: #1D
        r = abs(np.linspace(-1,1,max([sx,sy])))
        
    idx = np.where(r < radius)
    pdf = (1-r)**p
    pdf[idx] = 1

    #if len(idx[0]) > PCTG/3:
        #raise ValueError('Radius is too big! Rerun with smaller central radius.')

    # Bisect the data to get the proper PDF values to allow for the optimal sampling pattern generation
    if p==0:
        val = PCTG - len(idx[0])
        pdf = PCTG/(pdf.size-len(idx))*np.ones(pdf.shape)
        pdf[idx] = 1
    else:
        if style=='mult':
            pdf = 1/r**p
            #maxPx = sy/2
            #maxPy = sx/2
            alpha = 10
            maxPx = 10
            maxPy = 10
            c = 0.90
            while alpha>1 or (abs(np.sum(pdf)-PCTG) > (0.01*PCTG)):
                maxPx = c*maxPx
                maxPy = c*maxPy
                [px,py] = np.meshgrid(np.linspace(-maxPx,maxPx,sy),np.linspace(-maxPy,maxPy,sx))
                rpx = np.sqrt(px**2+py**2)
                r0 = rpx[idx[0][0],idx[1][0]]
                rpx = rpx - r0 + 1
                rpx[idx] = 1
                pdf = 1/(rpx**p)
                if cir:
                    [x,y] = np.meshgrid(np.linspace(-1,1,sy),np.linspace(-1,1,sx))
                    r = np.sqrt(x**2 + y**2)
                    outcyl = np.where(r > 1)
                    pdf[outcyl] = 0
                if pft:
                    pdf_pftLoc = pdf*(r >= (1+ext)*radius)*(r<=1)
                    loc = np.where(pdf_pftLoc[(img_sz[1]/2).astype(int),:]==0)[0][1]
                    pdf[:,0:loc] = 0
                val = PCTG - len(idx[0])
                sumval = np.sum(pdf) - len(idx[0])
                alpha = val/sumval
                pdf = alpha*pdf
                pdf[idx] = 1
        else:
            while(1):
                val = minval/2 + maxval/2;
                pdf = (1-r)**p + val
                if outcyl:
                    pdf[outcyl] = 0
                pdf[np.where(pdf > 1)] = 1
                pdf[idx] = 1
                N = np.floor(np.sum(pdf));
                if N > PCTG:
                    maxval = val
                elif N < PCTG:
                    minval = val;
                else:
                    break;
    
    if cir:
        [x,y] = np.meshgrid(np.linspace(-1,1,sy),np.linspace(-1,1,sx))
        r = np.sqrt(x**2 + y**2)
        outcyl = np.where(r > 1)
        pdf[outcyl] = 0
    
    pdf = ndimage.filters.gaussian_filter(pdf,3)
    
    if zpad_mat:
        if (img_sz[-2] > img_sz_hold[-2]) or (img_sz[-1] > img_sz_hold[-1]):
            pdf = zpad(pdf,img_sz)
        else:
            xdiff = int((img_sz[-2] - img_sz_hold[-2])/2)
            ydiff = int((img_sz[-1] - img_sz_hold[-1])/2)
            pdf = pdf[xdiff:-xdiff,ydiff:-ydiff]

    if disp:
        plt.figure
        plt.imshow(pdf)
        
    return pdf
    
def genSampling(pdf, n_iter, tol):
    '''
    Quick Monte-Carlo Algorithm to generate a sampling pattern, to try and have minimal peak interference. Number of samples is np.sum(pdf) +/- tol.
    
    Inputs:
    [np.array]  pdf - Probability density function to choose from
    [float]  n_iter - number of attempts
    [int]       tol - Deviation from the desired number of samples
    
    Outputs:
    [bool array]  mask - sampling pattern
    [float]  actpctg - actual undersampling factor
    
    This code is ported from Michael Lustig 2007
    '''
    
    pdf[np.where(pdf > 1)] = 1
    K = np.sum(pdf)
    
    minIntr = 1e6;
    minIntrVec = np.zeros(pdf.shape)
    stat = []
    
    for n in xrange(n_iter):
        tmp = np.zeros(pdf.shape)
        while abs(np.sum(tmp) - K) > tol:
            tmp = np.random.random(pdf.shape) < pdf
            
        TMP = np.fft.ifft2(tmp/(pdf+EPS))
        if np.max(abs(TMP[1:])) < minIntr:
            minIntr = np.max(abs(TMP[1:]))
            minIntrVec = tmp;
            
        stat.append(np.max(abs(TMP[1:])))
        
    actpctg = np.sum(minIntrVec)/minIntrVec.size
    return minIntrVec, actpctg

def genSamplingDir(img_sz=[180,180], 
                dirFile='/home/asalerno/Documents/pyDirectionCompSense/GradientVectorMag.txt',
                pctg=0.25,
                cyl=[0],
                radius=0.2,
                nmins=5,
                endSize=None,
                engfile=None):
    '''
    This was the original method for sampling directionally based data -- now we use dirSampling within the functin direction
    '''
    if not endSize:
        endSize = img_sz
    import itertools
    # load the directions
    print('Loading Directions...')
    dirs = np.loadtxt(dirFile) 
    n = int(dirs.shape[0])
    r = np.zeros([n,n])

    # Push everything onto one half sphere
    #    for i in xrange(n):
    #        if dirs[i,2] < 0:
    #            dirs[i,:] = -dirs[i,:]

    print('Calculating Distances...')
    # Calculate the distance. Do it for both halves of the sphere
    for i in xrange(n):
        for j in xrange(n):
            r[i,j] = min(np.sqrt(np.sum((dirs[i,:] + dirs[j,:])**2)),np.sqrt(np.sum((dirs[i,:] - dirs[j,:])**2)))

    invR = 1/(r+EPS)

    print('Finding all possible direction combinations...')
    # Find all of the possible combinations of directions
    k = int(np.floor(n*pctg)) # How many "directions" will have a point in k space
    combs = np.array(list(itertools.combinations(range(0,n),k))) # All the different vector combinations
    vecs = np.array(list(itertools.combinations(range(0,k),2))) # All the different combos that need to be checked for the energy
    engStart = np.zeros([combs.shape[0]]) # Initialization for speed of the energy


    print('Running PE Electrostatics system...')
    # Run the "Potential energy" for each of the combinations
    if not engfile:
        for i in xrange(combs.shape[0]):
            for j in xrange(vecs.shape[0]):
                engStart[i] = engStart[i] + invR[combs[i,vecs[j,0]],combs[i,vecs[j,1]]]
    else:
        engStart = np.load(engfile)
        # npy file
    #import pdb; pdb.set_trace()
    #np.save('/micehome/asalerno/Documents/pyDirectionCompSense/phantom/engFile30dir_'+ str(int(nmins)) + 'mins.npy',engStart)

    print('Producing "best cases..."')
    # Build the best cases of trying to get the vectors together
    ind = engStart.argsort() # Sort from lowest energy (farthest apart) to highest
    eng = engStart[ind] # Again, sort
    vecsInd = combs[ind,] # This tells us the vectors that we're going to be using for our mins
    locs = np.zeros([n,nmins]) # This gives us the mins for our individual vectors
    vecsMin = np.zeros([k,n*nmins]) 

    # Look for the locations where the indicies exist first (that is, they are the smallest)
    # and input those as the vectors we want to use
    for i in range(n):
        locs[i,] = np.array(np.where(vecsInd == i))[0,0:nmins]
        vecsMin[:,nmins*i:nmins*(i+1)] = vecsInd[locs[i,].astype(int),:].T

    # Only keep those rows that are unique
    vecsMin = unique_rows(vecsMin.T).astype(int)
    amts = np.zeros(n)

    # Count how often each direction gets pulled
    for i in xrange(n):
        amts[i] = vecsMin[vecsMin == i].size
    srt = amts.argsort()
    cts = amts[srt]
        
    print('Check lowest 20%')
    # Make sure we only look at the lowest 20% of the energies
    qEng = np.percentile(eng,20)

        # if theres a huge difference, tack more of the lower counts on, but make sure that we aren't hitting too high energy sets

        #vecsUnique,vecs_idx = np.unique(vecsInd,return_index = True)
        
    while cts[-1]/cts[0] >= 1.2:
        #import pdb; pdb.set_trace()
        srt_hold = srt.copy().reshape(1,len(srt))[0,:k]
        srt_hold.sort()
        # We need to add one here as index and direction number differ by a value of one
        indx = np.where(np.all(srt_hold == vecsInd,axis = 1)) 
        #import pdb; pdb.set_trace();
        if eng[indx] < qEng:
            vecsMin = np.vstack([vecsMin,srt_hold])
        else:
            while eng[indx] >= qEng: # Take this if the bottom ones are too big!
                #import pdb; pdb.set_trace();
                arr = np.zeros(k)-1 # Create a holder array
                cnt = 1 # Create an iterator
                tooManyTimesCnt = 0
                while np.any(arr == -1):
                    arr[0] = srt[0]
                    st = np.floor(n*np.random.random(1)) # A holder for the value
                    if not np.any(arr == st): # Make sure that value doesn't already exist in the holder array
                        if np.any(srt[-int(np.floor(n/4)):] == st):
                            tooManyTimesCnt += 1
                            if tooManyTimesCnt == int(n/2):
                                indx = np.where(np.all(srt_hold == vecsInd,axis = 1))
                                vecsMin = np.vstack([vecsMin,srt_hold])
                        else:
                            arr[cnt] = st; # If it doesn't, add it
                            cnt += 1 # Move to the next location
                arr_hold = arr.copy().reshape((1,len(arr)))[0,:k]
                arr_hold.sort() # Sort it out. Making sure it didn't end up too long
                indx = np.where(np.all(arr_hold == vecsInd,axis = 1)) # find the index
                if eng[indx] < qEng: # Make sure we oly add it if the indx is low enough
                    vecsMin = np.vstack([vecsMin,arr_hold])
            
        for i in xrange(dirs.shape[0]):
            amts[i] = vecsMin[vecsMin == i].size
        srt = amts.argsort()
        cts = amts[srt]
        #if np.random.random(1)<.1:
            #print(cts)
    
    # Now we have to finally get the sampling pattern from this!
    print('Obtaining sampling pattern...')
    [x,y] = np.meshgrid(np.linspace(-1,1,img_sz[-2]),np.linspace(-1,1,img_sz[-1]))
    r = np.sqrt(x**2 + y**2)
    
    # If not cylindrical, we need to have vals < 1
    if not cyl[0]:
        print('Not cylindrical, so we need r<=1')
        r = r/np.max(abs(r))
        
    [rows,cols] = np.where((r <= 1).astype(int)*(r > radius).astype(int) == 1)
    [rx,ry] = np.where(r <= radius)
    
    # Create our sampling mask
    samp = np.zeros(np.hstack([img_sz]))
    nSets = np.hstack([vecsMin.shape, 1])
    
    # Start making random choices for the values that require them
    #import pdb; pdb.set_trace()
    for i in xrange(len(rows)):
        val = np.floor(nSets[0]*np.random.random(1)).astype(int)
        choice = vecsMin[val,].astype(int)
        samp[choice,rows[i],cols[i]] = 1
        
    for i in xrange(len(rx)):
        samp[:,rx[i],ry[i]] = 1
        
    if np.any(endSize != img_sz):
        print('Zero padding...')
        samp_final = np.zeros(np.hstack([n,endSize]))
        
        for i in xrange(n):
            samp_hold = zpad(samp[i,:,:].reshape(img_sz),endSize)
            samp_final[i,:,:] = samp_hold.reshape(np.hstack([1,endSize]))
        
        samp = samp_final

    return samp

def radialHistogram(k,rmax=1,bins=50,pdf=None,sl=None,disp=1):
    
    maxxy = rmax
    [x,y] = np.meshgrid(np.linspace(-maxxy,maxxy,k.shape[0]), np.linspace(-maxxy,maxxy,k.shape[1]))
    r = np.sqrt(x**2+y**2)
    r *= k
    r = r[r!=0]
    bins = int(len(k[0,:])/2)
    cnts, binEdges = np.histogram(r.flat,bins=bins,normed=False)
    rads = np.linspace(binEdges[1],binEdges[-1],int(len(k[0,:])/2))
    rsq = rads**2
    areas = np.zeros(rsq.shape)
    areas[0] = np.pi*rsq[0]
    areas[1:] = np.pi*np.diff(rsq)

    #fig = plt.figure()
    plt.bar(binEdges[:-1],cnts,width=binEdges[1]-binEdges[0])
    ymax = np.max(cnts)*1.1
    #plt.bar(binEdges[:-1],cnts,width=binEdges[1]-binEdges[0])
    #ymax = np.max(cnts)*1.1
    plt.xlim(0,rmax)
    plt.ylim(0,ymax)
    #plt.title('Radial Histogram')
    plt.xlabel('Radius')
    plt.ylabel('Counts')
    
    return cnts, binEdges

    #if sl:
        #pltSliceHalf(pdf,sl,rads)
            
    #if disp:
        #plt.show()

def pltSliceHalf(im,sl,rads,col='b'):

    plt.plot(rads,im[sl,int(len(im[sl,:])/2):],col)
    plt.xlabel('Radius')
    plt.ylabel('Counts')
    
def pltSlice(im,sl,rads,col='b'):
    #if not fig:
        #fig = plt.figure()
    plt.plot(rads,im[sl,:],color=col)
    
def signalMask(im, thresh=0.1, iters = None):
    if not iters:
        iters = int(0.1*im.shape[0])
    
    mask = np.zeros(im.shape)
    highVal = np.where(im>thresh)
    mask[highVal] = 1

    yLen,xLen = mask.shape
    output = mask.copy()
    for iter in xrange(iters):
        #plt.imshow(output)
        #plt.show()
        for y in xrange(yLen):
            for x in xrange(xLen):
                if (y > 0     and mask[y-1,x]) or \
                (y < yLen - 1 and mask[y+1,x]) or \
                (x > 0        and mask[y,x-1]) or \
                (x < xLen - 1 and mask[y,x+1]):
                    output[y,x] = 1
        mask = output.copy()
        
    #mask = ndimage.binary_fill_holes(mask).astype(int)
    mask = ndimage.filters.gaussian_filter(mask,int(iters*0.5))
    return mask
    
def loRes(im,pctg):
    N = im.shape
    ph_ones=np.ones(N)
    [x,y] = np.meshgrid(np.linspace(-1,1,N[1]),np.linspace(-1,1,N[0]))
    rsq = x**2 + y**2
    loResMaskLocs = np.where(rsq < pctg)
    loResMask = np.zeros(N)
    loResMask[loResMaskLocs] = 1
    loResMask = sp.ndimage.filters.gaussian_filter(loResMask,3)
    data = np.fft.fftshift(loResMask)*tf.fft2c(im, ph=ph_ones)
    im_lr_wph = tf.ifft2c(data,ph=ph_ones)
    ph_lr = tf.matlab_style_gauss2D(im_lr_wph,shape=(5,5))
    ph_lr = np.exp(1j*ph_lr)
    im_lr = tf.ifft2c(data, ph=ph_lr)
    return im_lr
    
def makePEtable(k,filename):
    xLoc,yLoc = np.where(k)
    sx,sy = k.shape
    xLoc += int(-sy/2 + 1)
    xLocArrStr = np.char.mod('%i', xLoc)
    xLocStr = "\n    ".join(xLocArrStr)

    yLoc += int(-sx/2 + 1)
    yLocArrStr = np.char.mod('%i', yLoc)
    yLocStr = "\n    ".join(yLocArrStr)

    print('Saving petable to: ' + filename)
    with open(filename,'w') as out:
        t1 = 't1 = '
        t2 = 't2 = '
        sp = '    '
        out.write('{}\n{}{}\n{}\n{}{}'.format(t1,sp,xLocStr,t2,sp,yLocStr))

def readPEtable(inputfile):
    t1list=parse_petable_file(inputfile,'t1')
    t2list=parse_petable_file(inputfile,'t2')
    t1len = np.max(t1list) - np.min(t1list) + 1
    t2len = np.max(t2list) - np.min(t2list) + 1
    t1list -= np.min(t1list)
    t2list -= np.min(t2list)
    k = np.zeros([t1len,t2len])
    k[t1list,t2list] = 1
    return k

def genPEtable(t1,t2,filename):
    import re
    np.set_printoptions(threshold=1e99)
    t1s = 't1 = \n ' + re.sub('[\[\]]', '', np.array_str(t1))
    t2s = 't2 = \n ' + re.sub('[\[\]]', '', np.array_str(t2))
    ts = t1s + '\n' + t2s
    out = open(filename,'w')
    out.write(ts)
    out.close()
    np.set_printoptions(threshold=1e3)
    