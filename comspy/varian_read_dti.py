from sys import path as syspath
syspath.append("/home/bjnieman/source/mri_recon")
import numpy as np
from recon_genfunctions import FatalError,get_dict_value,default_recon
import varian_read_file as vrf
#import bruker_read_file as brf
import dti_fse_varian as seqmodule
from mnc_output import write_to_mnc_file
import imp
import recon_genfunctions as rgf
from mri_recon import dummyopt

def getDTIDataFromFID(inputdirectory,petable,imouse):
    #inputdirectory='/hpf/largeprojects/MICe/jacob/fid/26apr16.fid'
    #options=dummyopt(petable='/projects/souris/jacob/fid/table_test/JE_Table_Angdist_nf60_ni17',
    #                petable_ordered_pairs=True)
    options=dummyopt(petable=petable,petable_ordered_pairs=True)
    options.echoampcorr=False
    inputAcq = vrf.VarianAcquisition(inputdirectory)
    seqrec = seqmodule.seq_reconstruction(inputAcq,options,"./temp.mnc")
    
    seqrec.gen_kspace(imouse=imouse)

    return seqrec.kspace
    #inputAcq.procpar_dict['dro']

# Need to add a line that will output the diffusion directions based on the procpar files 