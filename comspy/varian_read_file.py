# functions for opening, reading, closing varian fid files
import array as A
import os
import exceptions
from numpy import *
#from struct import calcsize as sizeof
import re
import struct

program_name = 'varian_read_file.py'

class FatalError(exceptions.Exception):
    def __init__(self,args=None):
        self.msg = args

class VarianAcquisition():
    def __init__(self,inputpath,procpar_file=None):
        self.platform="Varian"
        #resolve file names and check existence
        if (procpar_file is None):
            procpar_file = os.path.join(inputpath,'procpar')
        else:
            if not (os.path.exists(procpar_file)):
                procpar_file = os.path.join(inputpath,procpar_file)
        try:
            if not ( os.path.exists(procpar_file) ):
                raise FatalError, "procpar file not found..."
        except FatalError, e:
            print 'Error(%s):' % 'open_vnmrfid_file', e.msg
            raise SystemExit
        self.inputpath=inputpath
        param_dict,procpar_text_lines = generate_procpar_dict(procpar_file)
        self.param_dict = param_dict
        self.nfid = int(get_dict_value(param_dict,'nfid',1))
        if (self.nfid==1):
            fid_file_list = [os.path.join(inputpath,'fid')]
        else:
            fid_file_list = [os.path.join(inputpath,'fid'+str(x)) for x in range(self.nfid)]
        try:
            for j in range(self.nfid):
                if not ( os.path.exists(fid_file_list[j]) ):
                    raise FatalError, "fid%d file not found (%s)..."%(j,fid_file_list[j])
        except FatalError, e:
            print 'Error(%s):' % 'open_vnmrfid_file', e.msg
            raise SystemExit
        self.fidfilelist = [open(fid_file_list[j],'r') for j in range(self.nfid)]
        self.header_tuple = struct.unpack('>iiiiiihhi',self.fidfilelist[0].read(32)) 
        #(nblocks,ntraces,np,ebytes,tbytes,bbytes,vers_id,status,nbheaders)
        bstr_nonneg = lambda n: n>0 and bstr_nonneg(n>>1).lstrip('0')+str(n&1) or '0'
        status_bits = bstr_nonneg(self.header_tuple[7])
        if (status_bits[-4]=='1'):
            datatype='f'
        elif (status_bits[-3]=='1'):
            datatype='i'
        else:
            datatype='h'
        self.header_info=[self.header_tuple[0],self.header_tuple[1],self.header_tuple[2],self.header_tuple[3],self.header_tuple[4],
                          self.header_tuple[5],self.header_tuple[6],datatype,self.header_tuple[8]]
        self.nmice=int(get_dict_value(param_dict,'nmice',1))
        self.nrcvrs = len(re.findall('y',get_dict_value(param_dict,'rcvrs','ynnn')))
        if (self.nmice>self.nrcvrs):
            print "propcar file specifies more mice than channels!!!"
            self.nmice = self.nrcvrs
        self.npe=int(get_dict_value(param_dict,'nv',1))
        self.npe2=int(get_dict_value(param_dict,'nv2',1))
        if (self.npe2<=0): self.npe2=1
        self.nslices=int(get_dict_value(param_dict,'nslices',1))
        self.nro=self.header_info[2]/2
        if (self.header_info[0]/(self.nmice*self.npe2)==1):
            self.data_shape=array((self.nmice,self.npe2,self.npe,self.nro),int)
        else:
            self.data_shape=array((self.header_info[0]/(self.nmice*self.npe2),self.nmice,self.npe2,self.npe,self.nro),int)
        self.nf=int(get_dict_value(param_dict,'nf',self.npe))
        self.ni=int(get_dict_value(param_dict,'ni',self.npe2/self.nfid))
        if (self.ni<0): self.ni=1
        self.nD=int(get_dict_value(param_dict,'nD',3))
        self.rcvrmouse_mapping=array(arange(self.nmice),int) #only convenient since Bruker needs this mapping

    def close(self):
        for x in self.fidfilelist: x.close()

    def getdatafids(self,fid_start,fid_end,rcvrnum=None,startpt=0,endpt=0):
        if (rcvrnum is None):
            mouse_num=0
        else:
            mouse_num = rcvrnum
        nblocks = long(self.header_info[0])
        ntraces = long(self.header_info[1])
        np = self.header_info[2]
        tbytes = long(self.header_info[4])
        bbytes = long(self.header_info[5])
        nbheaders = self.header_info[8]
        nfids=fid_end-fid_start
        if (startpt<0): startpt = np/2-startpt
        if (endpt<=0): endpt = np/2-endpt
        if (endpt<startpt) or (endpt-startpt>np/2):
            endpt = np/2; startpt = 0
        npts = 2*(endpt - startpt)
        data_elems = nfids*npts/2    
        data_error = 0
        complex_data = zeros((data_elems,),complex)
        for j in range(nfids):
            ifid = j+fid_start
            filenum=ifid/(ntraces*nblocks/self.nrcvrs)
            startblocknum=mouse_num+self.nrcvrs*((ifid-filenum*nblocks*ntraces/self.nrcvrs)/ntraces)
            starttracenum=ifid%ntraces
            #print "get_vnmr_datafids: ",mouse_num,fid_start,fid_end,ifid,filenum,startblocknum,starttracenum,len(vnmrfidfilelist)
            self.fidfilelist[filenum].seek(32+startblocknum*bbytes+28+starttracenum*tbytes+startpt*self.header_info[3]*2)
            bindata=A.array(self.header_info[7])
            try:
                bindata.read(self.fidfilelist[filenum],npts)
                bindata.byteswap()
            except EOFError:
                print 'Error(%s): Missing data in file!' % program_name
                print '        Trying to fetch %d fid in mouse %d (filenum %d, block %d, trace %d, nrcvrs %d)'%(ifid,mouse_num,filenum,startblocknum,starttracenum,nrcvrs)
                data_error = True
                break
            #complex_data[j*npts/2:(j+1)*npts/2]=array(bindata[0:npts:2],float16)+1.j*array(bindata[1:npts+1:2],float16)
            complex_data[j*npts/2:(j+1)*npts/2]=array(bindata[0:npts:2],float)+1.j*array(bindata[1:npts+1:2],float)
        data_shape = array((nfids,npts/2))
        complex_data = reshape(complex_data,tuple(data_shape))
        return complex_data,data_error
    
def generate_procpar_dict(procpar_file):
    procpar_fh = open(procpar_file,'r')
    text_lines = procpar_fh.readlines()
    procpar_fh.close()
    param_dict={}
    ntext = 0
    curr_line = 0
    nlines = len(text_lines)
    while (curr_line<nlines):
        line = text_lines[curr_line]
        m = re.search('^[a-z]',line)
        if not m:
            curr_line += 1
            continue
        else:
            words = line.split()
            varname = words[0]
            subtype = int(words[1])
            basictype = int(words[2])
            curr_line += 1
            line = text_lines[curr_line]
            words = line.split()
            nvals = int(words[0])
            if (basictype==1) and (subtype==7) and (not varname=='filter') and (not varname=='dres'):
                try:
                    varval = array([int(x) for x in words[1::]],int)
                except ValueError:
                    varval = array([float(x) for x in words[1::]],float)
            elif (basictype==1):
                varval = array([float(x) for x in words[1::]],float)
            else:
                varval = [ re.search('".*"',line).group()[1:-1] ]
                while (len(varval)<nvals):
                    curr_line+=1
                    line = text_lines[curr_line]
                    varval.append( re.search('".*"',line).group()[1:-1] )
            if (nvals==1): varval=varval[0]
            param_dict[varname] = varval
            curr_line+=1
    return param_dict,text_lines
    

def get_dict_value(param_dict,key,default):
    retvalue=default
    if param_dict.has_key(key):
        retvalue=param_dict[key]
    return retvalue
    
def parse_petable_file(petable_name,array_str):
    petable_fh = open(petable_name,'r')
    text_lines = petable_fh.readlines()
    curr_line = 0
    nlines = len(text_lines)
    #find array_str text --> 't1' or 't2' typically
    while 1:
        line = text_lines[curr_line]
        m = re.search(array_str,line)
        if not m:
            curr_line += 1
            continue
        else:
            break
    #parse text
    array_list=[]
    curr_line += 1
    while (curr_line<nlines):
        line = text_lines[curr_line]
        m = re.search('[a-z]',line)
        if m:
            break
        words = line.split()
        for x in words:
            array_list.append(int(x))
        curr_line+=1
    return array(array_list,int)
    
    
#######################################################################################3
#keep the following functions for compatibility with much older code
#shouldn't need to be used anymore though
#######################################################################################3
 
def open_vnmrfid_file(inputpath,procpar_file=None):
    if (procpar_file==None):
        procpar_file = os.path.join(inputpath,'procpar')
    else:
        if not (os.path.exists(procpar_file)):
            procpar_file = os.path.join(inputpath,procpar_file)
    try:
        if not ( os.path.exists(procpar_file) ):
            raise FatalError, "procpar file not found..."
    except FatalError, e:
        print 'Error(%s):' % 'open_vnmrfid_file', e.msg
        raise SystemExit
    param_dict,procpar_text_lines = generate_procpar_dict(procpar_file)
    nfid = int(get_dict_value(param_dict,'nfid',1))
    if (nfid==1):
        fid_file_list = [os.path.join(inputpath,'fid')]
    else:
        fid_file_list = [os.path.join(inputpath,'fid'+str(x)) for x in range(nfid)]
    try:
        for j in range(nfid):
            if not ( os.path.exists(fid_file_list[j]) ):
                raise FatalError, "fid%d file not found (%s)..."%(j,fid_file_list[j])
    except FatalError, e:
        print 'Error(%s):' % 'open_vnmrfid_file', e.msg
        raise SystemExit
    vnmrfidfilelist = [open(fid_file_list[j],'r') for j in range(nfid)]
    header_tuple = struct.unpack('>iiiiiihhi',vnmrfidfilelist[0].read(32)) 
    #(nblocks,ntraces,np,ebytes,tbytes,bbytes,vers_id,status,nbheaders)
    bstr_nonneg = lambda n: n>0 and bstr_nonneg(n>>1).lstrip('0')+str(n&1) or '0'
    status_bits = bstr_nonneg(header_tuple[7])
    if (status_bits[-4]=='1'):
        datatype='f'
    elif (status_bits[-3]=='1'):
        datatype='i'
    else:
        datatype='h'
    header_info=[header_tuple[0],header_tuple[1],header_tuple[2],header_tuple[3],header_tuple[4],
                 header_tuple[5],header_tuple[6],datatype,header_tuple[8]]
    nmice=get_dict_value(param_dict,'nmice',1)
    nv=get_dict_value(param_dict,'nv',1)
    nv2=get_dict_value(param_dict,'nv2',1)
    if (nv2<=0): nv2=1
    nro=header_info[2]/2
    if (header_info[0]/(nmice*nv2)==1):
        data_shape=array((nmice,nv2,nv,nro),int)
    else:
        data_shape=array((header_info[0]/(nmice*nv2),nmice,nv2,nv,nro),int)
    return vnmrfidfilelist,data_shape,header_info,param_dict,procpar_text_lines

def close_vnmrfid_file(vnmrfidfilelist):
    for fidfile in vnmrfidfilelist:
        fidfile.close()

def get_vnmr_datafids(vnmrfidfilelist,fid_start,fid_end,header_info,mouse_num=0,nrcvrs=1,startpt=0,endpt=0):
    nblocks = long(header_info[0])
    ntraces = long(header_info[1])
    np = header_info[2]
    tbytes = long(header_info[4])
    bbytes = long(header_info[5])
    nbheaders = header_info[8]
    nfids=fid_end-fid_start
    if (startpt<0): startpt = np/2-startpt
    if (endpt<=0): endpt = np/2-endpt
    if (endpt<startpt) or (endpt-startpt>np/2):
        endpt = np/2; startpt = 0
    npts = 2*(endpt - startpt)
    data_elems = nfids*npts/2
    data_error = 0
    complex_data = zeros((data_elems,),complex)
    for j in range(nfids):
        ifid = j+fid_start
        filenum=ifid/(ntraces*nblocks/nrcvrs)
        startblocknum=mouse_num+nrcvrs*((ifid-filenum*nblocks*ntraces/nrcvrs)/ntraces)
        starttracenum=ifid%ntraces
        #print "get_vnmr_datafids: ",mouse_num,fid_start,fid_end,ifid,filenum,startblocknum,starttracenum,len(vnmrfidfilelist)
        vnmrfidfilelist[filenum].seek(32+startblocknum*bbytes+28+starttracenum*tbytes+startpt*header_info[3]*2)
        bindata=A.array(header_info[7])
        try:
            bindata.read(vnmrfidfilelist[filenum],npts)
            bindata.byteswap()
        except EOFError:
            print 'Error(%s): Missing data in file!' % program_name
            print '        Trying to fetch %d fid in mouse %d (filenum %d, block %d, trace %d, nrcvrs %d)'%(ifid,mouse_num,filenum,startblocknum,starttracenum,nrcvrs)
            data_error = True
            break
        #complex_data[j*npts/2:(j+1)*npts/2]=array(bindata[0:npts:2],float16)+1.j*array(bindata[1:npts+1:2],float16)
        complex_data[j*npts/2:(j+1)*npts/2]=array(bindata[0:npts:2],float)+1.j*array(bindata[1:npts+1:2],float)
    data_shape = array((nfids,npts/2))
    complex_data = reshape(complex_data,tuple(data_shape))
    return complex_data,data_error



