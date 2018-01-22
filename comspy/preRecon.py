from __future__ import division
import numpy as np
import numpy.fft as fft
from scipy.ndimage.filters import gaussian_filter
import direction as d
import sampling as samp
import transforms as tf


def read_and_normalize(filename):
    '''
    Read in the file and ensure that it's normalized for our algorithm
    '''
    im = np.load(filename)
    im = im/np.max(abs(im))
    return im, im.shape

def direction_swap_axes(im, dirData=False):
    '''
    Read in the data and swapaxes in order to make sure that we have it
    in the right organizational aspect (RO, DIR, PE1, PE2)
    '''
    N = np.array(im.shape)
    # If we have directional data, first we check to make sure the req numbers exist
    if dirData is not False:
        nDir = dirData.shape[0]
        try: 
            iDir = list(N).index(nDir)
        except ValueError:
            raise ValueError('Hmm... The number of directions isn''t present in your data -- check again')
        '''
        Then we check the dimensions of the data in order to ensure that we have it as:
        (RO, DIR, PE1, PE2)
        Because we will be hitting RO in the outermost loop, and dealing with DIR, PE1 and PE2 all 
        together.
        
        The other case is when the RO is one - handled first in the if statements.
        '''
        print('Number of directions handled')
        Narg = np.argsort(-N)
        Nsort = N[Narg]
        if Nsort[-1] == 1:
            print('Dealing with one slice')
            nRO = Nsort[-1]
            iRO = Narg[-1]
        elif Nsort[0] == nDir:
            nRO = Nsort[1]
            iRO = Narg[1]
        else:
            nRO = Nsort[0]
            iRO = Narg[0]
        im = np.swapaxes(im,0,iRO)
        N = im.shape
        iDir = list(N).index(nDir)
        im = np.swapaxes(im,1,iDir)
    else:
        Narg = np.argsort(-N)
        Nsort = N[Narg]
        if Nsort[-1] == 1:
            print('Dealing with one slice')
            nRO = Nsort[-1]
            iRO = Narg[-1]
        else:
            nRO = Nsort[0]
            iRO = Narg[0]
        im = np.swapaxes(im,0,iRO)
    
    N = im.shape
    return im, N

def read_directional_data(dirFile,nmins):
    '''
    Read in the directions and create the directional data that we need for the scan
    '''
    if type(dirFile) is str:
        dirs = np.loadtxt(dirFile)
    else:
        dirs = dirFile
    dirInfo = d.calcAMatrix(dirs,nmins)
    return dirs, dirInfo

def create_scanner_k_space(im, N, P=2, pctg=0.25, dirData=False, dirs=None,
                        radius=0.2, cyl =[0], style='mult', pft=False, ext=0.5):
    '''
    Read in the data, size, and the directions (if they exist) so that we can create a
    retrospectively sampled set of data for testing.
    '''
    
    # Create a pdf so that we can use it to make a starting point
    pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=[1, N[-2], N[-1]], style='mult', pft=pft,ext=0.5)
    
    # Generate the sampling scheme, depending on whether or not 
    if dirData:
        if dirs is None:
            raise ValueError('If we have directional data, you need to feed this into the function')
        k = d.dirPDFSamp([int(dirs.shape[0]), N[-2], N[-1]], P=2, pctg=pctg, radius=radius, dirs=dirs, cyl=True, taper=0.25)[0]
    else:
        k = samp.genSampling(pdf, 50, 2)[0].astype(int)
    
    # Since our functions are built to work in 3D datasets, here we
    # make sure that N and things are all in 3D
    if len(N) == 2:
        N = np.hstack([1, N])
        k = k.reshape(N)
        im = im.reshape(N)
    elif len(N) == 3:
        if k.ndim == 2:
            k = k.reshape(np.hstack([1,N[-2:]])).repeat(N[0],0)
    
    k = np.fft.fftshift(k,axes=(-2,-1))
    # Convert the image data into k-space
    ph_ones = np.ones(N, complex)
    dataFull = tf.fft2c(im, ph=ph_ones,axes=(-2,-1))
    # Apply our sampling
    data = k*dataFull
    # Now we need to calculate the phase in order to deal with the undersampled image and the 
    # non perfect cancellation of terms 
    #filtdata = gaussian_filter(im_scan_wph.real,0,0) + 1j*gaussian_filter(im_scan_wph.imag,0,0)
    #ph_scan = np.exp(1j*np.angle(filtdata.conj()))
    im_scan_wph = tf.ifft2c(data,ph=ph_ones)
    ph_scan = np.angle(gaussian_filter(im_scan_wph.real,0) +  1.j*gaussian_filter(im_scan_wph.imag,0))
    ph_scan = np.exp(1j*ph_scan)
    im_scan = tf.ifft2c(data,ph=ph_scan)

    
    pdfDiv = pdf.copy()
    pdfZeros = np.where(pdf< 1e-4)
    pdfDiv[pdfZeros] = 1
    datadc = data/pdfDiv
    
    return dataFull, data, datadc, pdf, k, im_scan, ph_scan


def meas_phase(data):
    ph_ones = np.ones(data.shape, complex)
    im_scan_wph = tf.ifft2c(data,ph=ph_ones)
    ph_scan = np.angle(gaussian_filter(im_scan_wph.real,0) +  1.j*gaussian_filter(im_scan_wph.imag,0))
    ph_scan = np.exp(1j*ph_scan)
    im_scan = tf.ifft2c(data,ph=ph_scan)
    return im_scan, ph_scan
    


def pre_multistep(N, pctgSamp, k, radius, nSteps):
    '''
    Create the system to build the boxes that represent certain
    samplings of k-space where you will perform a multistep method.
    '''
    # Here, we look at the number of "steps" we want to do and step 
    # up from there. The "steps" are chose based on the percentage that 
    # we can sample and is based on the number of steps we can take.
    if nSteps < 1:
        raise ValueError('The minimum number for nSteps is 1')
    elif (nSteps - int(nSteps)) > 0:
        raise ValueError('Please use an integer number for nSteps')
    if np.sum([k[:,0,0],k[:,-1,0],k[:,0,-1],k[:,-1,-1]])/N[0] == 4:
        k = np.fft.fftshift(k,axes=(-2,-1))
    nSteps = int(nSteps)
    x, y = np.meshgrid(np.linspace(-1,1,N[-1]),np.linspace(-1,1,N[-2]))
    locs = (abs(x)<=radius) * (abs(y)<=radius)
    minLoc = np.min(np.where(locs==True))
    pctgSamp = np.zeros(minLoc)
    for i in range(1,minLoc):
        kHld = k[0,i:-i,i:-i]
        pctgSamp[i] = np.sum(kHld)/kHld.size
    pctgLocs = np.arange(1,nSteps)/(nSteps-1)
    locSteps = np.zeros(nSteps-1)
    locSteps[0] = minLoc
    # Find the points where the values are as close as possible
    for i in range(nSteps-1):
        locSteps[i] = np.argmin(abs(pctgLocs[i]-pctgSamp))
    # Flip it here to make sure we're starting at the right point
    locSteps = locSteps[::-1].astype(int)
    locSteps = np.hstack([locSteps,0])
    
    return locSteps