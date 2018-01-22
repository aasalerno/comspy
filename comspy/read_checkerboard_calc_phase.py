from __future__ import division
from optparse import OptionParser, Option, OptionValueError, OptionGroup
import imp
import sys
import os
import numpy as np
import numpy.fft as fft
sys.path.append('/home/bjnieman/source/mri_recon')
from mri_recon import dummyopt
import recon_genfunctions as rgf
import varian_read_file as vrf
from mnc_output import write_to_mnc_file
import matplotlib.pyplot as plt
from fse3dmice_recon import ROshift 
import comspy as cs
import comspy.preRecon as preRecon
import comspy.transforms as tf

inputdirectory='/hpf/largeprojects/MICe/asalerno/fid/20sep17.fid'
imouse=6
recon3d=True



def read_from_fid(inputdirectory,imouse=None,recon3d=False):
    # Ensure that it ends with a / as a directory
    if inputdirectory[-1]!='/':
        inputdirectory=inputdirectory+'/'
    if imouse==None:
        imouse=range(16)
    
    # Now we need to get the number of directors as per the procpar file
    inputAcq = vrf.VarianAcquisition(inputdirectory+'dir1')
    ndir=len(inputAcq.param_dict["setdro"])
    
    # Now we need to make a list to simplify how we do our checks
    inputdirectory_list=[os.path.join(inputdirectory, o) for o in os.listdir(inputdirectory) 
                                        if os.path.isdir(os.path.join(inputdirectory,o))]
    inputdirectory_list.sort()
    
    # Let's prep some necessary values now
    if not isinstance(imouse,list):
        imouse=[imouse]
    nmouse=len(imouse)
        
    nro=inputAcq.param_dict["np"]//2
    nf=int(inputAcq.param_dict["nf"])
    ni=int(inputAcq.param_dict["ni"])
    nfid=int(inputAcq.param_dict["nfid"])
    etl=int(inputAcq.param_dict["etl"])
    nv = int(inputAcq.param_dict["nv"])
    nv2 = int(inputAcq.param_dict["nv2"])
    
    carray=np.zeros( (nmouse,ndir,nfid*ni,nf,nro), complex)
    kacq=np.zeros( (nmouse,ndir,nv2,nv,nro), complex)
    dir_list = np.vstack([inputAcq.param_dict["setdro"],inputAcq.param_dict["setdpe"],inputAcq.param_dict["setdsl"]]).T
    
    petable_list=[]
    t1_arr = np.empty((ndir,nfid*ni*nf),int)
    t2_arr = np.empty((ndir,nfid*ni*nf),int)
    
    # Inport the data
    for k in range(ndir):
        inputAcq = vrf.VarianAcquisition(inputdirectory_list[k])
        petable_list.append(inputdirectory_list[k]+'/'+inputAcq.param_dict["petable"].split('/')[1])
        t1_arr[k]=vrf.parse_petable_file(petable_list[k],'t1')
        t2_arr[k]=vrf.parse_petable_file(petable_list[k],'t2')
        # Read in the data
        for j in range(nfid*ni):
            for mouse in range(nmouse):
                fid_data,errflag = inputAcq.getdatafids(j*nf,(j+1)*nf,rcvrnum=imouse[mouse])
                carray[mouse,k,j,:,:] = fid_data.copy()
        # Run the system to pull kacq
        for mouse in range(nmouse):
            kacq[mouse,k] = rgf.petable_orderedpair_reordering(carray[mouse,k],matrix=(nv2,nv),t1array=t1_arr[k],t2array=t2_arr[k])
        
        
    ##sort all images into k-space
    img = np.zeros((nmouse,ndir,nv2,nv,nro),complex)
    nv2shift = nv2//2
    nvshift = nv//2
    nroshift = nro//2
    kmod1 = np.zeros( (nmouse,ndir,nv2,nv,nro),complex )
    kmod2 = np.zeros( (nmouse,ndir,nv2,nv,nro),complex )
    imgmod1 = np.zeros( (nmouse,ndir,nv2,nv,nro),complex )
    imgmod2 = np.zeros( (nmouse,ndir,nv2,nv,nro),complex )
    
    
    for k in range(ndir):
        ct1 = np.reshape(t1_arr[k],(len(t1_arr[k])//etl,etl))
        ct2 = np.reshape(t2_arr[k],(len(t2_arr[k])//etl,etl))
        for mouse in range(nmouse):
            kmod1[mouse,k,ct2[0::2,:].flatten()+nv2//2-1,ct1[0::2,:].flatten()+nv//2-1,:] = \
                        kacq[mouse,k,ct2[0::2,:].flatten()+nv2//2-1,ct1[0::2,:].flatten()+nv//2-1,:]
            kmod2[mouse,k,ct2[1::2,:].flatten()+nv2//2-1,ct1[1::2,:].flatten()+nv//2-1,:] = \
                        kacq[mouse,k,ct2[1::2,:].flatten()+nv2//2-1,ct1[1::2,:].flatten()+nv//2-1,:]
            if recon3d==True:
                imgmod1[mouse,k] = rgf.recon_3d(kmod1[mouse,k])
                imgmod2[mouse,k] = rgf.recon_3d(kmod2[mouse,k])
    
                    
    # ------- Perhaps start a new function here ------- #
    kmod1 = np.squeeze(kmod1)
    kmod2 = np.squeeze(kmod2)
    
    kmod1,N1 = preRecon.direction_swap_axes(kmod1,dir_list)
    kmod2,N2 = preRecon.direction_swap_axes(kmod2,dir_list)
    ph_ones = np.ones(kmod1.shape, complex)
    
    im_scan1 = np.zeros( kmod1.shape, complex)
    ph_scan1 = np.zeros( kmod1.shape, complex)
    im_scan2 = np.zeros( kmod2.shape, complex)
    ph_scan2 = np.zeros( kmod2.shape, complex)
    
    for d in range(ndir):
        print('Working on direction %d'%(d+1)+' (using 1 counting)')
        kmod1vals = preRecon.meas_phase(kmod1[:,d,:,:])
        im_scan1[:,d,:,:] = kmod1vals[0]
        ph_scan1[:,d,:,:] = kmod1vals[1]
        kmod2vals = preRecon.meas_phase(kmod2[:,d,:,:])
        im_scan2[:,d,:,:] = kmod1vals[0]
        ph_scan2[:,d,:,:] = kmod1vals[1]
        
        
        
    
