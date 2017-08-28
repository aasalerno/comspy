#!/usr/bin/env python
# Imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
from sys import path as syspath
import os
syspath.append("/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/")
syspath.append("/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/source/")
syspath.append("/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/source/wavelet")
os.chdir('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/')
import visualization as vis
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['image.cmap'] = 'gray'
import saveFig

def rmse(xk,x):
    return np.sqrt(np.sum((xk-x)**2)/len(xk))

filename = '/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/brainData/P14/data/fullySampledBrain.npy'

pctg = .25
SL = [75,150,200]
TV = [0.005,0.0025,0.001,0.0005,0.0001]
XFM  = [0.005,0.0025,0.001,0.0005,0.0001]
lam_trust = [0.1,0.25,0.33,0.5,0.67,0.75,0.9]

im = abs(np.load(filename)[SL[0],:,:])
imRMSE = np.zeros([len(SL), len(TV), len(XFM), len(lam_trust), im.shape[0], im.shape[1]])
rmses = np.zeros([len(SL), len(TV), len(XFM), len(lam_trust)])
clim = []

for i in range(len(SL)):
    im = abs(np.load(filename)[SL[i],:,:])
    clim.append(np.min(im),np.max(im))
    for j in range(len(TV)):
        for k in range(len(XFM)):
            for m in range(len(lam_trust)):
                imRMSE[i,j,k,m,:,:] = np.load('temp/' + str(int(100*pctg)) + '_slice_' + str(SL[i]) + '_TV_' + str(TV[j]) + '_XFM_' + str(XFM[k]) + '_lamTrust_' + str(int(100*lam_trust[m])) + '.npy')
                rmses[i,j,k,m] = rmse(imRMSE[i,j,k,m,:,:],im)
                

titles = []
for j in range(len(TV)):
    for k in range(len(XFM)):
        titles.append('TV = ' + str(TV[j]) + '   XFM = ' + str(XFM[k]))

        
        
for i in range(len(SL)):
    for m in range(len(lam_trust)):
        labs = ['Slice ' + str(SL[i]) + ' with $\lambda_{trust}$ = ' + str(lam_trust[m]), 'TV', 'XFM']
        vis.figSubplots(np.concatenate(imRMSE[i,:,:,m,:,:],axis=0),titles=titles,labs=labs)
        saveFig.save('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/tests/rmseTests/ims_sl_' + str(SL[i]) + '_lamTrust_' + str(lam_trust[m]) + '_allTVXFM')
        fig, ax = plt.subplots()
        img = ax.imshow(rmses[i,:,:,m],cmap='jet',interpolation='None')
        ax.set_title('RMSE: Slice ' + str(SL[i]) + ' $\lambda_{trust}$ = ' + str(lam_trust[m]))
        strTV = ['']
        strXFM = ['']
        [strTV.append(TV[q]) for q in range(len(TV))]
        [strXFM.append(XFM[q]) for q in range(len(XFM))]
        ax.set_xticklabels(strTV)
        ax.set_yticklabels(strXFM)
        fig.colorbar(img)
        saveFig.save('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/tests/rmseTests/rmses_sl_' + str(SL[i]) + '_lamTrust_' + str(lam_trust[m]) + '_allTVXFM')