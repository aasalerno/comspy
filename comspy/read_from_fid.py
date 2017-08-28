from sys import path as syspath
#syspath.append("/home/bjnieman/source/mri_recon")
import numpy as np
from recon_genfunctions import FatalError,get_dict_value,default_recon
import varian_read_file as vrf
import bruker_read_file as brf
#from mnc_output import write_to_mnc_file
import imp
import flash_cyl_MICe as seqmodule
import recon_genfunctions as rgf
from scipy.optimize import leastsq


class dummyopt():
    def __init__(self,petable=None,petable_ordered_pairs=False,fov_shift_ro="0,0,0,0",fov_shift_pe1="0,0,0,0",fov_shift_pe2="0,0,0,0",
                 outputreps=False,complexavg=False,petable_pe1=False,petable_pe2=False,fermi_ellipse=False,
                 noshift_ppe=False,noshift_ppe2=False,large_data_recon=False,mouse_list=None,
                 nofft=False,fft1d=False,fft2d=False,fft3d=True,real=False,imag=False,phase=False,
                 image_range_min=0.0,image_range_max=0.0,max_range=False,vType=None,apowidth=0.8,
                 echoamp_alpha=0.02,phasecorr_data=None,phasecorr_table=None,phasecorr_plot=False,
                 no_echo_roshift=False,no_pe_phaseshift=False,no_Echo_shift_apply=False,no_echo_phaseshift=False,separate_multi_acq=False,Pftacq=False):
        self.petable=petable
        self.petable_ordered_pairs=petable_ordered_pairs
        self.fov_shift_ro=fov_shift_ro
        self.fov_shift_pe1=fov_shift_pe1
        self.fov_shift_pe2=fov_shift_pe2
        self.outputreps=outputreps
        self.complexavg=complexavg
        self.petable_pe1=petable_pe1
        self.petable_pe2=petable_pe2
        self.fermi_ellipse=fermi_ellipse
        self.noshift_ppe=noshift_ppe
        self.noshift_ppe2=noshift_ppe2
        self.large_data_recon=large_data_recon
        self.mouse_list=mouse_list
        self.nofft=nofft
        self.fft1d=fft1d
        self.fft2d=fft2d
        self.fft3d=fft3d
        self.real=real
        self.imag=imag
        self.phase=phase
        self.image_range_min=image_range_min
        self.image_range_max=image_range_max
        self.max_range=max_range
        self.vType=vType
        self.apowidth=apowidth
        self.echoamp_alpha=echoamp_alpha
        self.phasecorr_data=phasecorr_data
        self.phasecorr_table=phasecorr_table
        self.phasecorr_plot=phasecorr_plot
        self.no_echo_roshift=no_echo_roshift
        self.no_echo_phaseshift=no_echo_phaseshift
        self.no_pe_phaseshift=no_pe_phaseshift
        self.no_Echo_shift_apply=no_Echo_shift_apply
        self.separate_multi_acq=separate_multi_acq
        self.Pftacq=Pftacq


def getDataFromFID(petable,inputdirectory,imouse):
    
    options=dummyopt(complexavg=False,petable=petable,petable_ordered_pairs=True,outputreps=True)
    inputAcq = brf.BrukerAcquisition(inputdirectory)
    seqrec = seqmodule.seq_reconstruction(inputAcq,options,"./temp.mnc")
    seqrec.gen_kspace(imouse=imouse)
    seqrec.Pftacq = False
    seqrec.recon()
    return seqrec.image_data[-1,1]