import comspy as cs
from sys import argv
from __future__ import division
import string
import os
import re
import numpy as np
from numpy.random import shuffle
from optparse import OptionParser, Option, OptionValueError
import matplotlib.pyplot as plt



def factor(x):
    factors = []
    for i in range(1,x):
        if i in factors:
            break
        if x/i == x//i:
            factors.append(i)
            factors.append(x//i)
    return np.array(factors)


def petable_file_output(i1,i2,nf,ntables,outputfile,ndir,linelim=8,appendnfni=True):
    file_num_format = "_%d" 
    ni = len(i1)//(nf*ntables)
    ptspertable = len(i1)//ntables
    if (appendnfni):
        outputfile = outputfile + "_nf%d_ni%d_dir%d"%(nf,ni,ndir)
    full_fh = open(outputfile,'w')
    full_fh.write("t1 = \n")
    file_fh_list=[]
    multifid = (ntables>1)
    for j in range(ntables):
        if (multifid):
            curr_fh = open(outputfile+file_num_format%j,'w')
            curr_fh.write("t1 = \n")
            file_fh_list.append(curr_fh)
        for k in range(ni):
            for q in range(nf):
                full_fh.write("    %d"%i1[j*ptspertable+k*nf+q])
                if (multifid): curr_fh.write("    %d"%i1[j*ptspertable+k*nf+q])
                if ((q+1)%linelim==0) and (q<nf-1):
                    full_fh.write("\n")
                    if (multifid): curr_fh.write("\n")
            full_fh.write("\n")
            if (multifid): curr_fh.write("\n")
    full_fh.write("t2 = \n")
    for j in range(ntables):
        if (multifid):
            curr_fh = file_fh_list[j]
            curr_fh.write("t2 = \n")
        for k in range(ni):
            for q in range(nf):
                full_fh.write("    %d"%i2[j*ptspertable+k*nf+q])
                if (multifid): curr_fh.write("    %d"%i2[j*ptspertable+k*nf+q])
                if ((q+1)%linelim==0) and (q<nf-1):
                    full_fh.write("\n")
                    if (multifid): curr_fh.write("\n")
            full_fh.write("\n")
            if (multifid): curr_fh.write("\n")
        if (multifid): curr_fh.close()
    full_fh.close()
    return None


#---------------------------------------------------




writeCheckerboardsSep=False
setseed=True

if setseed:
    np.random.seed(627)

N = np.array([180,180],int)
nv=N[-2]
nv2=N[-1]
radius = 0.2
pft=False
pctg=0.25
etl = 6
nf=6
ni=107
ntables=1
nzeros=5
outfile='AS_cylgen_etl6_k01'

if nf==None:
    nf=etl*10
    ni=17



dirfile = '/hpf/largeprojects/MICe/asalerno/comspyData/gradvecs/120dirAnneal.txt'
dirs = np.loadtxt(dirfile)
ndirs=len(dirs)

pdf = cs.sampling.genPDF(N, 2, pctg=pctg, radius=radius, cyl=[1, N[-2], N[-1]], style='mult', pft=pft,ext=0.5)
    
# Generate the sampling scheme, depending on whether or not 
k = cs.direction.dirPDFSamp([ndirs, N[-2], N[-1]], P=2, pctg=pctg*0.9, radius=radius, dirs=dirs, cyl=True, taper=0.25)[0]

# Our system doesn't produce perfect percentages of data collected but that's ok - if it's greater, then we can just get rid of points if need be, if too few we can get rid of them.
totpctg = np.sum(k)/k.size*4/np.pi

xx,yy=np.meshgrid(np.linspace(-1,1,nv),np.linspace(-1,1,nv2))
r = np.sqrt(xx**2 + yy**2)

#desiredNumPoints = np.round(pctg*np.pi/4*nv2*nv/nf/ntables)*nf*ntables
#desiredNumPointsCB = desiredNumPoints//2

# kludge hack for nf and ni
desiredNumPoints = np.round(pctg*np.pi/4*nv2*nv/nf/ntables/ni)*nf*ntables*ni
desiredNumPointsCB = desiredNumPoints//2

# Let's make the checkerboard
allinds = np.meshgrid(np.arange(0,nv)-nv/2,np.arange(0,nv2)-nv2/2)
gridtest = (allinds[0]%2 + allinds[1]%2)%2
wmap = gridtest==1
bmap = gridtest==0
wlocs = np.where(wmap)
blocs = np.where(bmap)

kwhite = k*wmap[np.newaxis,:,:]
kblack = k*bmap[np.newaxis,:,:]

nwb = np.empty( (2,ndirs), int)
nwb[0,:] = np.sum(kwhite,axis=(-2,-1))
nwb[1,:] = np.sum(kblack,axis=(-2,-1))


ptsChangeWB = nwb-desiredNumPointsCB
rMutation = r>radius


# Need to make ak distribution that will effectively determine likelihood of a point being chosen - now the caveat is that it needs to be exact!
for i in range(ndirs):
    # start with white sq
    # If it's less than zero, we have too few points
    if ptsChangeWB[0,i] < 0:
        potentialPoints = np.where((1-kwhite[i])*wmap*(r<1)*rMutation)
        ptsChoice = np.random.choice(len(potentialPoints[0]), int(abs(ptsChangeWB[0,i])),
                                     replace=False)
        kwhite[i,potentialPoints[0][ptsChoice],potentialPoints[1][ptsChoice]] = 1
        
    # If it's greater than zero, we have too many points
    elif ptsChangeWB[0,i] > 0:
        potentialPoints = np.where(kwhite[i]*wmap*(r<1)*rMutation)
        ptsChoice = np.random.choice(len(potentialPoints[0]), int(abs(ptsChangeWB[0,i])),
                                     replace=False)
        kwhite[i,potentialPoints[0][ptsChoice],potentialPoints[1][ptsChoice]] = 0
    
    # Now with black sq
    # If it's less than zero, we have too few points
    if ptsChangeWB[1,i] < 0:
        potentialPoints = np.where((1-kblack[i])*bmap*(r<1)*rMutation)
        ptsChoice = np.random.choice(len(potentialPoints[0]), int(abs(ptsChangeWB[1,i])),
                                     replace=False)
        kblack[i,potentialPoints[0][ptsChoice],potentialPoints[1][ptsChoice]] = 1
    # If it's greater than zero, we have too many points
    elif ptsChangeWB[1,i] > 0:
        potentialPoints = np.where(kblack[i]*bmap*(r<1)*rMutation)
        ptsChoice = np.random.choice(len(potentialPoints[0]), int(abs(ptsChangeWB[1,i])),
                                     replace=False)
        kblack[i,potentialPoints[0][ptsChoice],potentialPoints[1][ptsChoice]] = 0


x,y = np.meshgrid(np.arange(nv2)-nv2/2,np.arange(nv)-nv/2)
desiredNumPointsCB = int(desiredNumPointsCB)

t1White = np.empty( (ndirs,desiredNumPointsCB), int)
t1Black = np.empty( (ndirs,desiredNumPointsCB), int)
t2White = np.empty( (ndirs,desiredNumPointsCB), int)
t2Black = np.empty( (ndirs,desiredNumPointsCB), int)
rWhite = np.empty( (ndirs,desiredNumPointsCB), float)
rBlack = np.empty( (ndirs,desiredNumPointsCB), float)
thetaWhite = np.empty( (ndirs,desiredNumPointsCB), float)
thetaBlack = np.empty( (ndirs,desiredNumPointsCB), float)


for i in range(ndirs):
    kwhiteLocs = np.where(kwhite[i])
    kblackLocs = np.where(kblack[i])
    rWhite[i] = r[kwhiteLocs]
    rBlack[i] = r[kblackLocs]
    rWhiteLocs= np.argsort(rWhite[i])
    rBlackLocs= np.argsort(rBlack[i])
    rWhite[i] = rWhite[i,rWhiteLocs]
    rBlack[i] = rBlack[i,rBlackLocs]
    
    t1White[i] = x[kwhiteLocs][rWhiteLocs]
    t1Black[i] = x[kblackLocs][rBlackLocs]
    t2White[i] = y[kwhiteLocs][rWhiteLocs]
    t2Black[i] = y[kblackLocs][rBlackLocs]
    thetaWhite[i] = np.arctan2(t2White[i],t1White[i])
    thetaBlack[i] = np.arctan2(t2Black[i],t1Black[i])
    
    
# Now that we have all of the data and the order it should be in, lets get it in a logical order.    
npts = int(desiredNumPointsCB/etl)

t1WhitePrint = np.empty( (ndirs,etl,npts), int)
t1BlackPrint = np.empty( (ndirs,etl,npts), int)
t2WhitePrint = np.empty( (ndirs,etl,npts), int)
t2BlackPrint = np.empty( (ndirs,etl,npts), int)
rWhitePrint = np.empty( (ndirs,etl,npts), float)
rBlackPrint = np.empty( (ndirs,etl,npts), float)
thetaWhitePrint = np.empty( (ndirs,etl,npts), float)
thetaBlackPrint = np.empty( (ndirs,etl,npts), float)

for i in range(ndirs):
    for j in range(etl):
        thetaWhitePrint[i,j,:] = thetaWhite[i,j*npts:(j+1)*npts]
        t1WhitePrint[i,j,:] = t1White[i,j*npts:(j+1)*npts]
        t2WhitePrint[i,j,:] = t2White[i,j*npts:(j+1)*npts]
        rWhitePrint[i,j,:] = rWhite[i,j*npts:(j+1)*npts]
        
        thetaBlackPrint[i,j,:] = thetaBlack[i,j*npts:(j+1)*npts]
        t1BlackPrint[i,j,:] = t1Black[i,j*npts:(j+1)*npts]
        t2BlackPrint[i,j,:] = t2Black[i,j*npts:(j+1)*npts]
        rBlackPrint[i,j,:] = rBlack[i,j*npts:(j+1)*npts]
        
        thetaWhitePrintLocs = np.argsort(thetaWhitePrint[i,j])
        thetaBlackPrintLocs = np.argsort(thetaBlackPrint[i,j])
        
        thetaWhitePrint[i,j,:] = thetaWhitePrint[i,j,thetaWhitePrintLocs]
        t1WhitePrint[i,j,:] = t1WhitePrint[i,j,thetaWhitePrintLocs]
        t2WhitePrint[i,j,:] = t2WhitePrint[i,j,thetaWhitePrintLocs]
        rWhitePrint[i,j,:] = rWhitePrint[i,j,thetaWhitePrintLocs]
        
        thetaBlackPrint[i,j,:] = thetaBlackPrint[i,j,thetaBlackPrintLocs]
        t1BlackPrint[i,j,:] = t1BlackPrint[i,j,thetaBlackPrintLocs]
        t2BlackPrint[i,j,:] = t2BlackPrint[i,j,thetaBlackPrintLocs]
        rBlackPrint[i,j,:] = rBlackPrint[i,j,thetaBlackPrintLocs]
        
        
        
t1interleavedArrays = np.empty( (ndirs,npts*2,etl), int)
t2interleavedArrays = np.empty( (ndirs,npts*2,etl), int)

for i in range(ndirs):
    for j in range(npts*2):
        if (j%2 == 1):
            t1interleavedArrays[i,j,:] = t1WhitePrint[i,:,j//2]
            t2interleavedArrays[i,j,:] = t2WhitePrint[i,:,j//2]
        elif (j%2==0):
            t1interleavedArrays[i,j,:] = t1BlackPrint[i,:,j//2]
            t2interleavedArrays[i,j,:] = t2BlackPrint[i,:,j//2]

'''
Now we have to do the work for each fid file
1) Ensure that we handle zeros properly - just to act as a counter.
2) Splitting into multiple fids 
3) Getting a good ni (likely handled by 2)
4) Writing the files out properly
    a) Writing out a "checkerboard1", "checkerboard2", and "interleaved" (if user wants)
    b) Writing out the three cases for each direction - counted properly
    c) Ensuring that checkerboards are output properly
'''

# Step 1 - Handle zeros.
ndirsTotal = ndirs+nzeros
zeroLocs = [0]
for i in range(1,nzeros):
    zeroLocs.append(int(np.percentile(np.arange(ndirsTotal),100*i/(nzeros-1))))


# Steps 2 and 3 - Split into a good ni and ntables
factors = factor(npts*2)
ntabloc = np.argmin(abs(factors-ntables))//2*2
ntables, ni = factors[ntabloc:ntabloc+2] # looks weird but remember python is last digit exclusive

# Step 4 - Write out the files....

outfile_cb1 = outfile+'_cb1_%i_%i'%(N[-2],N[-1])
outfile_cb2 = outfile+'_cb2_%i_%i'%(N[-2],N[-1])
outfile_ileave = outfile+'_ileave_%i_%i'%(N[-2],N[-1])

cnt=0
for i in range(ndirsTotal):
    if i not in zeroLocs:
        # case where we write them all
        if writeCheckerboardsSep == True:
            petable_file_output(t1WhitePrint[cnt].T.flatten(),t2WhitePrint[cnt].T.flatten(),nf,ntables,outfile_cb1,ndir=i,linelim=etl)
            petable_file_output(t1BlackPrint[cnt].T.flatten(),t2BlackPrint[cnt].T.flatten(),nf,ntables,outfile_cb2,ndir=i,linelim=etl)
        # write the interleaved regardless
        petable_file_output(t1interleavedArrays[cnt].flatten(),t2interleavedArrays[cnt].flatten(),nf,ntables,outfile_ileave,ndir=i+1,linelim=etl)
        cnt+=1
        
            
print('ndir: %i \nnzero: %i \netl:%i \nnf: %i \nni: %i \nnfid: %i'%(ndirs,nzeros,etl,nf,ni,ntables))












#if __name__ == '__main__':

    #usage = """%s <matrix1> <matrix2> <output file>
   #or  %s --help
   
#%s is a script for generating petable files for varian cylindrical ge image acquisition
#and reconstruction.
#"""
    #usage = usage % ((program_name, )*3)

    #parser = OptionParser(usage)
    #parser.add_option("--clobber", action="store_true", dest="clobber",
                      #default=0, help="overwrite output file")
        #parser.add_option("--cylindrical", action="store_true", dest="cylindrical",
                      #default=1, help="shave corners of k-space in PE1 and PE2")
    #parser.add_option("--angorder", action="store_true", dest="angorder", default=0,
                      #help="order elements by angle in PE2-PE1 plane")
    #parser.add_option("--angdist", action="store_true", dest="angdist", default=0,
                      #help="order elements by angle in PE2-PE1 plane and distance in PE2")
    #parser.add_option("--checkerboard",type="int",dest="checkerboard",
                      #default=0, help="grab only encodes on a checkerboard (0=False,1=black squares, 2=white squares")    
    #parser.add_option("--pseudocyl",action="store_true",dest="pseudocyl",default=0,
                      #help="linearly arrange etl along pe1 axis within cylinder, similar to cartesian images")
    #parser.add_option("--cyl_concentric_ellipses", 
                      #action="store_true",dest="cyl_concentr_ellipses",default=0,
                      #help="sort multiple 'echoes' by concentric ellipses (major axis on pe2) with etl divisions")
    #parser.add_option("--nf",type="int",dest="nf",
                       #default=1, help="desired nf for each computed table")
    #parser.add_option("--individ_table_limit",type="int",dest="individ_table_limit",
                       #default=4096, help="limit of number of elements in each table")
    #parser.add_option("--etl",type="int",dest="etl",
                      #default=1, help="Echo train length")
    #parser.add_option("--k0echo",type="int",dest="k0echo",
                      #default=1, help="echo number for centre of k-space in cylpe2dist option - This option uses 1-counting")
    #parser.add_option("--output_phase_corr_file", action="store_true", 
                      #dest="output_phase_corr_file",default=0, help="output a reduced sampling phase correction file (covers principle axes and diagonals only)")
    #options, args = parser.parse_args()
    
    #outputfile = args[-1]