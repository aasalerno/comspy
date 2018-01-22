#!/usr/bin/env python -tt
#
#
# direction.py
#
#

#!/usr/bin/env python -tt
#
#
# direction.py
#
#

from __future__ import division
import numpy as np
import numpy.fft as fft
import scipy.optimize as sciopt
from numpy.linalg import inv
import sampling as samp
import scipy.ndimage as ndimage



def dot_product_threshold_with_weights(filename,
                        threshold = 0.1,
                        sigma = 0.35/2):
    dirs = np.loadtxt(filename)
    num_vecs = dirs.shape[0]
    cnt = 0
    
    locs = np.array([])
    vals = []
    
    for i in xrange(0,num_vecs):
        for j in xrange(1,num_vecs):
            dp = np.dot(dirs[i,:],dirs[j,:])
            
            if dp >= threshold:
                cnt = cnt + 1
                locs[cnt,:] = np.vstack([locs, [i, j]])
                vals[cnt] = np.vstack([vals, np.exp((dp**2 - 1)/(2*sigma**2))])
    
    return locs, vals

def dot_product_with_mins(dirs,
                        nmins = 5):
    '''
    This code exists to quickly calculate the closest directions in order to quickly get the values we need to calculate the mid matrix for the least squares fitting
    '''
    #dirs = np.loadtxt(filename) # Load in the file
    num_vecs = dirs.shape[0] # Get the number of directions
    
    dp = np.zeros([num_vecs,num_vecs]) # Preallocate for speed
        
    for i in xrange(num_vecs):
        for j in xrange(num_vecs):
            dp[i,j] = np.dot(dirs[i,:],dirs[j,:]) # Do all of the dot products
    
    inds = np.fliplr(np.argsort(abs(dp))) # Sort the data based on *rows*
    return inds[:,1:nmins+1]

def func(x,a,b):
    return a + b*x

def calcAMatrix(dirs,nmins):
    '''
    The purpose of this code is to create the middle matrix for the calculation:
        Gdiff**2 = del(I_{ijkrq}).T*M*del(I_{ijkrq})
        
    By having the M matrix ready, we can easily parse through the data trivially.
    
    We calculate M as [A*(A.T*A)**(-1)][(A.T*A)**(-1)*A.T]
    
    Where A is from (I_{ijkr} - I_{ijkq}) = A_rq * B_{ijkq}
    Note that there is a different M for each direction that we have to parse through
    
    The return is an nDirs x nMins x nMins matrix where m is the number of directions that we have in the dataset.
    '''
    n = dirs.shape[0]
    for i in range(n):
        inds = dot_product_with_mins(dirs,nmins)
        #dirs = np.loadtxt(filename)
        
    M = np.zeros([n,nmins,nmins])
    A = np.zeros([n,nmins,dirs.shape[1]])
    Ahat = np.zeros([n,dirs.shape[1],nmins])
    
    for qDir in xrange(n):
        for i in range(len(inds[qDir])):
            a = np.argmin([np.sum((dirs[inds[qDir,:],:][i]-dirs[qDir,:])**2),np.sum((dirs[inds[qDir,:],:][i]+dirs[qDir,:])**2)])
            if a == 0:
                A[qDir,i,:] = dirs[inds[qDir,:][i],:] - dirs[qDir,:]
            elif a == 1:
                A[qDir,i,:] = dirs[inds[qDir,:][i],:] - dirs[qDir,:]
        #datHold = datHold/np.linalg.norm(datHold) # Should I do this? Normalizes the vectors
        #A = np.hstack([A, datHold])
        # Calculate Ahat, which is the solution to Ax = b, [(A'A)^(-1)*A']b = x
        Ahat[qDir] = np.dot(inv(np.dot(A[qDir].T,A[qDir])),A[qDir].T)
        #M[qDir] = np.dot(Ahat[qDir].T,Ahat[qDir])
    
    
    # We need to take care of the positive and negatives of the images that we are using, since this system has a +/- aspect to the comparisons that occur
    
    # Make lists to hold the positive indicies and the negative ones
    indsNeg = range(n)
    indsPos = range(n)
        
    for kk in xrange(n):
        indsNeg[kk] = [np.repeat(kk,nmins), range(nmins)]
        indsPos[kk] = np.where(inds==kk)
        
    # dI, the derivative with respect to the Image. We need to now apply the +/-
    dI = np.zeros([n,n,nmins])
    #dIM = np.zeros([n,nmins,n])
    Ause = range(n)
    
    for kk in range(n):
        dI[kk,indsNeg[kk][0],indsNeg[kk][1]] = -1
        dI[kk,indsPos[kk][0],indsPos[kk][1]] = 1
        #Ause[kk] = np.where(np.any(dI[kk,:,:] != 0,axis=1))[0]
        #for d in xrange(len(Ause[kk])):
            #colUse = Ause[kk][d]
            #dIM[kk,:,colUse] = np.dot(dI[kk,colUse,:],M[colUse,:,:])
    
    dirInfo = [Ahat]
    dirInfo.append(dI)
    dirInfo.append(inds)
    
    return dirInfo


def least_Squares_Fitting(x,N,strtag,dirs,inds,Ahat):
    
    #import pdb; pdb.set_trace()
    x0 = x.copy().reshape(N[0],-1)
    nmins = inds.shape[1]
    #dirloc = strtag.index("diff")
    #x0 = np.rollaxis(x0,dirloc)
    Gdiff = np.zeros([N[0],3,N[1]*N[2]])
    
    for q in xrange(dirs.shape[0]):
        r = inds[q,:]
        
        # Assume the data is coming in as image space data and pull out what we require
        Iq = x0[q,:]
        Ir = x0[r,:]
        #nrow, ncol = Iq.shape
        
        #A = np.zeros(np.hstack([r.shape,3]))
        Irq = Ir - Iq # Iq will be taken from Ir for each part of axis 0
        #Aleft = np.linalg.solve((A.T*A),A.T)
        #beta = np.zeros(np.hstack([Iq.shape,3]))
        
        Gdiff[q] = np.dot(Ahat[q],Irq)
        #for i in xrange(nrow):
            #for j in xrange(ncol):
                #Gdiffsq[q,i,j] = np.dot(np.dot(Irq[:,i,j],M[q,:,:]),Irq[:,i,j])
    
    # This line puts the data back into the orientation that it was in before
    #Gdiffsq = np.rollaxis(Gdiffsq,0,dirloc)
    #Gdiff = np.sum(Gdiff,axis=1)
    return Gdiff
    
def dirDataSharing(data,dirs,origDataSize=0,nmins=5,bymax=1):
    if np.all(origDataSize) == False:
        origDataSize = data.shape[-2:]
    
    N = data.shape
    
    [x,y] = np.meshgrid(np.linspace(-1,1,origDataSize[0]),np.linspace(-1,1,origDataSize[1]))
    r = np.sqrt(x**2+y**2)
    
    if np.all(N[-2:] == origDataSize) == False:
        r = zpad(r,N[-2:])
    
    x,y = np.where(np.logical_and(r>0,r<1))
    
    if len(N) == 2:
        N = np.hstack([1, N])
        
    # Dot product matrix!
    dp = np.zeros([N[0],N[0]])
    
    for i in range(N[0]):
        for j in range(N[0]):
            dp[i,j] = abs(np.inner(dirs[i,:],dirs[j,:]))
    
    # Sort from least to greatest
    inds = np.argsort(dp)
    d = np.sort(dp)
    
    if bymax:
        d = np.fliplr(d)
        inds = np.fliplr(inds)
        
    data_tog = data.copy()
    
    for i in range(N[0]):
        for j in range(len(x)):
            cnt=0
            if abs(data[i,x[j],y[j]]) < 1e-6:
                #import pdb; pdb.set_trace()
                while cnt < nmins and (abs(data[inds[i,cnt],x[j],y[j]])<1e-6):
                    cnt += 1
                data_tog[i,x[j],y[j]] = data[inds[i,cnt],x[j],y[j]]
    
    return data_tog
    

def makeDirSetsPE(dirs,pctg):
    import collections
    if isinstance(dirs,str):
        dirs = np.loadtxt(dirs)
    nDirs = len(dirs)
    dp = np.zeros([nDirs,nDirs])
    dist = np.zeros([nDirs,nDirs])
    
    for i in range(nDirs):
        for j in range(nDirs):
            dp[i,j] = abs(np.dot(dirs[i,:],dirs[j,:]))
            dist[i,j] = np.min([np.sqrt(np.sum((dirs[i,:] - dirs[j,:])**2)), np.sqrt(np.sum((dirs[i,:] + dirs[j,:])**2))])
    
    # Calculate the distances so that we can look at the best cases
    dist += 1e-6
    dist_inv = 1/dist
    dist_inv[np.arange(nDirs),np.arange(nDirs)] = 0
    
    # Lets first make the different "starting points" that we need 
    nSets = int(np.round(1/pctg))+1
    nVals = int(np.floor(nDirs/nSets))
    startPoints = np.argsort(dp)[:,-(nSets):]
    allCombos = []
    finEnergies = []
    
    
    for dtest in xrange(nDirs):
        dirLocs = np.arange(nDirs)
        valsHeld = startPoints[dtest]
        dpHold = dp.copy()
        combs = []
        for i in xrange(nSets):
            combs.append([valsHeld[i]])
            valsRemaining = np.delete(dirLocs,valsHeld)
        for nVal in xrange(nVals-1):
            energies = np.zeros([nSets,len(valsRemaining)])
            minLocs = np.zeros([nSets,len(valsRemaining)])
            minVals = np.zeros([nSets,len(valsRemaining)])
            for nSet in xrange(nSets):
                row_idx = np.array(combs[nSet])
                col_idx = np.array(valsRemaining)
                energies[nSet,:] = np.sum(dist_inv[row_idx[:,None],col_idx].reshape(len(combs[nSet]),len(valsRemaining)),axis=0)
                minLocs[nSet,:] = valsRemaining[np.argsort(energies[nSet,:])]
                minVals[nSet,:] = np.sort(energies[nSet,:])
            listMins = list(minLocs[:,0])
            sets = collections.defaultdict(list)
            dups = collections.defaultdict(list)
            for index, item in enumerate(listMins):
                sets[item].append(index)
            for key in sets:
                if len(sets[key])>1:
                    dups[key] = sets[key]
            if not dups:
                minLocsUse = minLocs[:,0]
            else:
                while dups:
                    contestSets = dups[dups.keys()[0]]
                    setWinner = contestSets[np.argmin(minVals[contestSets,0])]
                    setLosers = np.delete(contestSets,np.argwhere(np.array(contestSets) == setWinner))
                    minLocs[setLosers,:] = np.roll(minLocs[setLosers,:],-1,axis=1)
                    minVals[setLosers,:] = np.roll(minVals[setLosers,:],-1,axis=1)
                    listMins = list(minLocs[:,0])
                    sets = collections.defaultdict(list)
                    dups = collections.defaultdict(list)
                    for index, item in enumerate(listMins):
                        sets[item].append(index)
                    for key in sets:
                        if len(sets[key])>1:
                            dups[key] = sets[key]
            
            for nSet in xrange(nSets):
                combs[nSet].append(int(listMins[nSet]))
                
            valsRemaining = np.delete(valsRemaining,np.where(np.in1d(valsRemaining,np.array(listMins).astype(int))))
        
        if np.any(valsRemaining):
            # we need to find where the last ones should go
            energies = np.zeros([nSets,len(valsRemaining)])
            minLocs = np.zeros([nSets,len(valsRemaining)])
            minVals = np.zeros([nSets,len(valsRemaining)])
            for nSet in xrange(nSets):
                row_idx = np.array(combs[nSet])
                col_idx = np.array(valsRemaining)
                energies[nSet,:] = np.sum(dist_inv[row_idx[:,None],col_idx].reshape(len(combs[nSet]),len(valsRemaining)),axis=0)
                #minLocs[nSet,:] = valsRemaining[np.argsort(energies[nSet,:])]
                #minVals[nSet,:] = np.sort(energies[nSet,:])
            
            minVals = np.sort(energies,axis=0)
            minLocs = np.argsort(energies,axis=0)
            listMins = list(minLocs[0,:])
            sets = collections.defaultdict(list)
            dups = collections.defaultdict(list)
            for index, item in enumerate(listMins):
                sets[item].append(index)
            for key in sets:
                if len(sets[key])>1:
                    dups[key] = sets[key]

            if not dups:
                    minLocsUse = minLocs[:,0]
            else:
                while dups:
                    contestSets = dups[dups.keys()[0]]
                    setWinner = contestSets[np.argmin(minVals[0,contestSets])]
                    setLosers = np.delete(contestSets,np.argwhere(np.array(contestSets) == setWinner))
                    minLocs[:,setLosers] = np.roll(minLocs[:,setLosers],-1,axis=0)
                    minVals[:,setLosers] = np.roll(minVals[:,setLosers],-1,axis=0)
                    listMins = list(minLocs[0,:])
                    sets = collections.defaultdict(list)
                    dups = collections.defaultdict(list)
                    for index, item in enumerate(listMins):
                        sets[item].append(index)
                    for key in sets:
                        if len(sets[key])>1:
                            dups[key] = sets[key]
            
            valsDelete = []
            for val in range(len(listMins)):
                combs[listMins[val]].append(valsRemaining[val])
                valsDelete.append(valsRemaining[val])
                
            valsRemaining = np.delete(valsRemaining,np.where(np.in1d(valsRemaining,np.array(valsDelete).astype(int))))
            
            if valsRemaining:
                print('Something went wrong. Values still left')
                #import pdb; pdb.set_trace()
        
        allCombos.append(combs)
    
    sumEnergiesCombs = []
    allEnergies = np.zeros(len(allCombos))
    for i in xrange(nDirs):
        sumEnergies = np.zeros(nSets)
        sumCombo = np.zeros(nSets)
        for j in xrange(nSets):
            row_idx = np.array(allCombos[i][j])
            col_idx = row_idx
            sumEnergies[j] = np.sum(dist_inv[row_idx[:,None],col_idx])/2
            sumCombo[j] = sumEnergies[j]/len(allCombos[i][j])
        sumEnergiesCombs.append(sumCombo)
        allEnergies[i] = np.sum(sumEnergies)
        
    bestChoice = np.argmin(allEnergies)
    
    return allCombos[bestChoice], allCombos, bestChoice
    
    

    
def dirSampPattern(N,P,pctg,radius,dirs,radFac=1.5):
    #import pdb; pdb.set_trace()
    N = np.array(N[-2:])
    smallImgSize = np.floor(radFac*(radius*N)).astype(int)
    if smallImgSize[0]%2 == 1:
        smallImgSize[0] -= 1
    if smallImgSize[1]%2 == 1:
        smallImgSize[1] -= 1
    sq = ((N-smallImgSize)/2).astype(int)
    pctgSmall = np.min([2*pctg,0.75])
    pdf = samp.genPDF(smallImgSize,P,pctgSmall,radius=1/radFac,cyl=[1,smallImgSize[-2],smallImgSize[-1]],pft=False,ext=0.5)
    #pdf = samp.genPDF(smallImgSize,P,pctgSmall,radius=1/radFac,cyl=[0],pft=False,ext=0.5)
    # We want to make sure that we're only looking at the circle now...
    x,y = np.meshgrid(np.linspace(-1,1,smallImgSize[1]),np.linspace(-1,1,smallImgSize[0]))
    r = np.sqrt(x**2+y**2)
    pdf[pdf<=pctg] = 0
    pdf[(pdf == 0)*(r <= 1)] = pctg
    k = np.zeros(np.hstack([len(dirs), N]))
    for i in range(len(dirs)):
        k[i,sq[0]:-sq[0],sq[1]:-sq[1]] = samp.genSampling(pdf, int(1e-2*pdf.size), 2)[0].astype(int)
    
    sampSets = makeDirSetsPE(dirs,pctg)[1]
    nDirs = len(dirs)
    nSets = len(sampSets[0])
    #nVals = len(sampSets[1])
    #sampSets = np.array(sampSets)
    
    x,y = np.meshgrid(np.linspace(-1,1,N[1]),np.linspace(-1,1,N[0]))
    rSamp = np.sqrt(x**2+y**2)
    rLocs = (rSamp<=1).astype(int)
    rSmall = zpad((r<=1).astype(int),rSamp.shape)
    sampLocs = np.where(rLocs*(rSmall==0))
    
    for i in range(len(sampLocs[0])):
        randDir = int(np.random.random(1)*nDirs)
        randSet = int(np.random.random(1)*nSets)
        k[sampSets[randDir][randSet],sampLocs[0][i],sampLocs[1][i]] = 1
        
    return k

def dirPDFSamp(N,P,pctg,radius,dirs,cyl=True,taper=0.1):
    print("Create PDF")
    N0 = N
    N = np.array(N[-2:])
    x,y = np.meshgrid(np.linspace(-1+1e-4,1-1e-4,N[1]),np.linspace(-1+1e-4,1-1e-4,N[0]))
    r = np.sqrt(x**2 + y**2)
    pdf = np.zeros(N)
    pdf[r<=radius] = 1
    if cyl:
        totPix = np.pi/4*np.prod(N)
        PCTG = round(pctg*totPix)
    else:
        totPix = np.prod(N)
        PCTG = round(pctg*totPix)
    midPix = round(np.sum(pdf))
    leftPix = PCTG-midPix
    leftPdf = leftPix/totPix
    rTap = radius+taper
    rHold = r.copy()
    rHold = ((r<=rTap)*(r>radius))*r
    rHoldMax = np.max(rHold)
    rHold = abs(rHold - rHoldMax)/taper
    rHold[rHold>(rHoldMax/taper-1e-9)] = 0
    leftPdf = (leftPix - (np.sum((rHold<1)*(rHold>0)*(1-leftPdf))/P))/totPix
    if cyl:
        leftPdf = leftPdf*np.pi/4
    rHold[(rHold<1)*(rHold>0)] += leftPdf #**(1/P)
    rHold = rHold/np.max(rHold)
    rHold[rHold>1] = 0
    #rHold[rHold<1(rHold>=1)*(rHold<=1e3)] = 1
    #rHold[rHold>1e4] = 0
    pdf = pdf + rHold
    tapPix = round(np.sum(pdf))
    
    if cyl:
        pdf[(r<=1)*(r>rTap)] = leftPdf
    else:
        pdf[r>rTap] = leftPdf
    
    pdf = ndimage.filters.gaussian_filter(pdf,1)
    pdf[pdf>1] = 1
    
    if cyl:
        pdf[(r<=1)*(pdf<leftPdf)] = leftPdf
        pdf[r>1] = 0
    else:
        pdf[(r<=1)*(pdf<leftPdf)] = leftPdf
        
    #pdf[abs(pdf-leftPdf)<1e-4] = 0
    print("Create Var. Dens. Pattern")
    k = np.zeros(N0)    
    for i in range(N0[0]):
        k[i] = samp.genSampling(pdf, 10, 2)[0].astype(int)
    
    print("Create teams of directions")
    sampSets = makeDirSetsPE(dirs,leftPdf)[1]
    nDirs = len(dirs)
    nSets = len(sampSets[0])
    
    rLocs = ((r<=1)*(r>=(0.95*rTap))).astype(int)
    sampLocs = np.where(rLocs)
    
    print("Choose groups for each point in k-space")
    for i in range(len(sampLocs[0])):
        randDir = int(np.random.random(1)*nDirs)
        randSet = int(np.random.random(1)*nSets)
        k[sampSets[randDir][randSet],sampLocs[0][i],sampLocs[1][i]] = 1
    
    return k, pdf

def zpad(orig_data,res_sz):
    res_sz = np.array(res_sz)
    orig_sz = np.array(orig_data.shape)
    padval = np.ceil((res_sz-orig_sz)/2)
    res = np.pad(orig_data,([int(padval[0]),int(padval[0])],[int(padval[1]),int(padval[1])]),mode='constant')
    return res