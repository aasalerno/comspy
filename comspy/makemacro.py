import numpy as np
from __future__ import division

nzeros=5
includeStartParams=True
includeEndParams=True
includePhaseCorrection=True

dirfile = '/hpf/largeprojects/MICe/asalerno/comspyData/gradvecs/120dirAnneal.txt'
dirs = np.loadtxt(dirfile)
ndir = len(dirs)
bvals=np.array([30]*ndir)
nbvals = len(np.unique(bvals))

# Rotate the data
rotLoc = np.array([1,0,0],float).reshape(1,3)
I = np.eye(3)
v = np.cross(rotLoc,dirs[0,:]).T
s = np.linalg.norm(v)
c = np.dot(rotLoc,dirs[0,:])
vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
R = I + vx + np.dot(vx,vx)*(1-c)/s**2
dirs = np.dot(dirs,R)

# Figure out where we should intersperse the zeros!
ndirsTotal = ndir+nzeros
zeroLocs = [0]
for i in range(1,nzeros):
    zeroLocs.append(int(np.percentile(np.arange(ndirsTotal),100*i/(nzeros-1))))

# Now intersperse them
for i in range(nzeros):
    dirs = np.insert(dirs,zeroLocs[i],[0,0,0],axis=0)
    bvals = np.insert(bvals,zeroLocs[i],0,axis=0)
    

# Setup the file 
filename = '/home/asalerno/temp/as_' + str(ndir) + 'diff_' + str(nbvals) + 'bval'
file = open(filename,'w')


# Write the initialization 
if includeStartParams == True:
    file.writelines(["exists('setdro','parameter','current'):$params \n",
                    "\n",
                    "if $params < 0.5 then \n",
                    "    create('setdro','real') \n",
                    "    create('setdpe','real') \n",
                    "    create('setdsl','real') \n",
                    "    create('setgdiff2','real') \n",
                    "    create('setpetable','string') \n",
                    "    create('cnt','real')\n",
                    "    create('gdiffcorr','real')\n",
                    "endif \n",
                    "\n",
                    "if $ddoe < 0.5 then \n",
                    "    create('destdirorig','string') \n",
                    "endif \n",
                    "\n",
                    "// Required scan parameters \n",
                    "cnt=1 \n",
                    "cfid=0 \n",
                    "nfid=7 \n",
                    "nfmod=6 \n",
                    "ni=107\n",
                    "nf=60\n",
                    "cyl_ni=107\n",
                    "cyl_nf=60\n",
                    "destdirorig='/4tdata/'+operator+'/'+pslabel+'/'+time_submitted+'/'\n",
                    "shell('mkdir -p ' + destdirorig):$ret\n",
                    "\n",
                    "gdiffcorr=0,0,0\n",
                    "seqfil='fse3dmice_asbn_gshape_diff_gdiffcorr_checkerboard'\n","\n"])


zeropetable="'AS/AS_cylgen_etl6_k01_c1c2ileave_180_180_nf60_ni107'"
petable="'AS/AS_cylgen_etl6_k01_180_180_ileave_180_180_nf6_ni1070_dir'"
phaseCorrPetable = "'AS/AS_test_k1_6_180_180_less_phasecorr_nf6_ni720'"

# Now lets start writing up the preamble
setdro=list(dirs[:,0])
setdpe=list(dirs[:,1])
setdsl=list(dirs[:,2])
setgdiff2=list(bvals)
setpetable='setpetable='+zeropetable+','+petable

if includePhaseCorrection==True:
    setdro.append(0.0)
    setdpe.append(0.0)
    setdsl.append(0.0)
    setgdiff2.append(0.0)
    setpetable = setpetable+','+phaseCorrPetable

# Ensure we don't have a =,
setdrostr=['%.3f'%setdro[i] for i in range(1,len(setdro))]
setdro='setdro='+",".join(setdrostr)

setdpestr=['%.3f'%setdpe[i] for i in range(1,len(setdpe))]
setdpe='setdpe='+",".join(setdpestr)

setdslstr=['%.3f'%setdsl[i] for i in range(1,len(setdsl))]
setdsl='setdsl='+",".join(setdslstr)

setgdiff2str=['%.2f'%setgdiff2[i] for i in range(1,len(setgdiff2))]
setgdiff2='setgdiff2='+",".join(setgdiff2str)




#for b in range(nbvals):
    #for d in range(ndir):
        #setdro.append('%.3f'%dirs[d,0])
        #setdro.append(',')
        #setdpe.append('%.3f'%dirs[d,1])
        #setdpe.append(',')
        #setdsl.append('%.3f'%dirs[d,2])
        #setdsl.append(',')
        #setgdiff2.append(str(bvals[b]))
        #setgdiff2.append(',')
        #setpetable.append(petable)
        #setpetable.append(',')
        
#setdro=setdro[:-1]
#setdpe=setdpe[:-1]
#setdsl=setdsl[:-1]
#setgdiff2=setgdiff2[:-1]
#setpetable=setpetable[:-1]


file.writelines([setdro,
                 "\n\n",
                 setdpe,
                 "\n\n",
                 setdsl,
                 "\n\n",
                 setgdiff2,
                 "\n\n",
                 setpetable,
                 "\n\n",
                 "array=''",
                 "\n\n\n"])


if includeEndParams==True:
    file.writelines(["// Set up the first scan\n",
                     "dro=setdro[cnt]\n",
                     "dpe=setdpe[cnt]\n",
                     "dsl=setdsl[cnt]\n",
                     "gdiff2=setgdiff2[cnt]\n",
                     "petable=setpetable[cnt]\n",
                     "wexp='as_dtifid_wpc'"])
    
file.close()

