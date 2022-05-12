# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:56:38 2022
Surface tension and adsorption fits
@author: willg
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for analytical solutions
import scipy
from scipy import optimize
from scipy.special import erfc as erfc
import math
from matplotlib import rc
import matplotlib.ticker as mtick
from labellines import labelLine, labelLines
rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams['font.size'] = 16
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
#%%
def conc_star(conc_pfas, anion, cation): 
    I = 0.5*(conc_pfas + anion + cation) # - Ionic strength calculations are in molarity
    logy = (0.507*np.sqrt(I))/(1+np.sqrt(I))+0.1*I #activity
    y = 10**(-logy)
    conc_star = np.sqrt((y*(conc_pfas+cation))*(y*conc_pfas))*1000 #C* in mol/m3
    return conc_star
    
def langmuir(wtr, T_max, a, conc_star, conc):
    sft = wtr - (8310*294)*T_max*np.log(1 + a*conc_star) 
    surf_ex = (T_max*a*conc_star)/(1+ a*conc_star)
    kaw = surf_ex/conc
    return sft, kaw
#%%-concentrations
conc = np.array([0.1, 0.5, 1, 5, 10, 50, 100]) #mg/L
conc_model = np.linspace(10**-8, 0.001, num = 500) #molarity

conc_salt = np.array([0.09, 0.45, 0.9, 4.5, 9, 45, 90])

bruss_conc = np.array([0.00000024, 0.00000115, 0.00000237, 0.00001136, 0.00002334, 0.00011681, 0.00023012, 0.00115173, 0.00226897, 0.00268812, 0.00601369, 0.01135572, 0.0223714])
#%%PFOA
pfoa_mw = 414070 #mg/mol

#measured data
pfoa_conc = np.multiply(conc, 1/pfoa_mw)
pfoa_meas = np.array([72.599, 72.552, 72.672, 72.482, 71.724, 70.507, 66.834])

#modeled data
pfoa_cstar_model = conc_star(conc_model, 0, 0)
pfoa_sft, pfoa_kaw = langmuir(72.8, 6.92*10**-6, 1.79, pfoa_cstar_model, conc_model*1000)

#PFOA nacl measured
pfoa_nacl_conc = np.multiply([90, 45, 9, 4.5, 0.45, 0.09], 1/pfoa_mw)
pfoa_bruss_conc = np.array([0.00000024, 0.00000115, 0.00000237, 0.00001136, 0.00002334, 0.00011681, 0.00023012, 0.00115173, 0.00226897, 0.00268812, 0.00601369, 0.01135572, 0.0223714])
pfoa_nacl_meas = np.array([57.383, 61.458, 67.885, 69.988, 70.023, 72.972])
pfoa_bruss_meas = np.array([71.19, 70.595, 70.298, 67.917, 66.131, 59.881, 56.012, 44.702, 38.006, 34.137, 29.97, 22.53, 21.935])

#modeled data
pfoa_cstar_nacl_model = conc_star(conc_model, 0.1, 0.1)**2
pfoa_nacl_sft, pfoa_nacl_kaw = langmuir(71.8, 3.08*10**-6, 0.5, pfoa_cstar_nacl_model, conc_model*1000)

#%%-PFOS
pfos_mw = 538220 #mg/mol

#measured
pfos_conc = np.multiply(conc, 1/pfos_mw)
pfos_meas = np.array([72.678, 72.682, 72.546, 72.06, 71.393, 68.904, 65.561])

#modeled
pfos_cstar_model = conc_star(conc_model, conc_model, 0)
pfos_sft, pfos_kaw = langmuir(72.9, 3.08*10**-6, 6.36, pfos_cstar_model, conc_model*1000)

#PFOS nacl
pfos_nacl_conc = np.flip(np.multiply([90, 45, 9, 4.5, 0.9, 0.45, 0.09], 1/pfos_mw))
pfos_nacl_meas = np.array([72.562, 71.991, 70.006, 65.637, 60.659, 52.13, 51.248])

#modeled
pfos_cstar_nacl_model = conc_star(conc_model, 0.1, 0.1)**2
pfos_nacl_sft, pfos_nacl_kaw = langmuir(73.08, 2.64E-5, 0.2, pfos_cstar_nacl_model, conc_model*1000)

#%%-PFHxS
pfhxs_mw = 438200 #mg/mol

#measured
pfhxs_conc = np.multiply(conc, 1/pfhxs_mw)
pfhxs_meas = np.array([72.717, 72.543, 72.858, 72.497, 72.548, 71.916, 71.248])

#modeled
pfhxs_cstar_model = conc_star(conc_model, conc_model, 0)
pfhxs_sft, pfhxs_kaw = langmuir(72.829, 5.94*10**-7, 6.378, pfhxs_cstar_model, conc_model*1000)

#PFHxS nacl===================================================================================================
pfhxs_nacl_conc = np.flip(np.multiply([90, 45, 9, 4.5, 0.9, 0.45, 0.09], 1/pfhxs_mw))
pfhxs_nacl_meas = np.array([72.984, 72.55, 72.408, 71.014, 69.92, 65.711, 62.779])

#modeled
pfhxs_cstar_nacl_model = conc_star(conc_model, 0.1, 0.1)**2
pfhxs_nacl_sft, pfhxs_nacl_kaw = langmuir(73.27, 1.5*10**-6, 1.399, pfhxs_cstar_nacl_model, conc_model*1000)

#%%-PFDA
pfda_mw = 514080 #mg/mol

#measured
pfda_conc = np.multiply(conc, 1/pfda_mw)
pfda_meas = np.array([72.575, 72.366, 72.284, 71.591, 71.292, 57.084, 45.958])

#modeled
pfda_cstar_model = conc_star(conc_model, 0, 0)
pfda_sft, pfda_kaw = langmuir(72.55, 3.77*10**-5, 1.77, pfda_cstar_model, conc_model*1000)

#PFOS nacl
pfda_nacl_conc = np.flip(np.multiply([90, 45, 9, 4.5, 0.9, 0.45, 0.09], 1/pfda_mw))
pfda_nacl_meas = np.array([72.405, 71.769, 70.791, 57.155, 50.982, 30.946, 29.472])

#modeled
pfda_cstar_nacl_model = conc_star(conc_model, 0.1, 0.1)**2
pfda_nacl_sft, pfda_nacl_kaw = langmuir(73.28, 1.5*10**-5, 0.7, pfda_cstar_nacl_model, conc_model*1000)

#%%--PLOTTING
pfoac = plt.cm.Reds(np.linspace(0,1,5))
pfosc = plt.cm.Blues(np.linspace(0,1,5))
pfhxsc = plt.cm.Greens(np.linspace(0,1,5))
pfdac = plt.cm.Purples(np.linspace(0,1,5))

mi = 'o' #marker icon
mi2 = 's'
ms = 100 #marker size
lw = 5 #linewidth
ls = '--'
ft = 24 #fontsize
#%%
fig0, ([ax, ax1], [ax2, ax3]) =  plt.subplots(2, 2, figsize=(14, 12), dpi=200)

# fig1, (ax, ax1) =  plt.subplots(1, 2, figsize=(14, 6), dpi=200)

#PFAS measured data in ultrapure water========================================
ax.scatter(pfoa_conc*1000, pfoa_meas, color=pfoac[2], s = ms, marker = mi, edgecolor = 'b', zorder = 10, 
           label = 'PFOA')
ax.scatter(pfos_conc*1000, pfos_meas, color=pfosc[2], s = ms, marker = mi, edgecolor = 'b', zorder = 10,
           label = 'PFOS')
ax.scatter(pfhxs_conc*1000, pfhxs_meas, color=pfhxsc[2], s = ms, marker = mi, edgecolor = 'b', zorder = 10,
           label = 'PFHxS')
ax.scatter(pfda_conc*1000, pfda_meas, color=pfdac[2], s = ms, marker = mi, edgecolor = 'b', zorder = 10,
           label = 'PFDA')

#PFAS langmuir fits in ultra pure=============================================
ax.plot(conc_model*1000, pfoa_sft, color=pfoac[2], lw = lw, ls = ls)
ax.plot(conc_model*1000, pfos_sft, color=pfosc[2], lw = lw, ls = ls)
ax.plot(conc_model*1000, pfhxs_sft, color=pfhxsc[2], lw = lw, ls = ls)
ax.plot(conc_model*1000, pfda_sft, color=pfdac[2], lw = lw, ls = ls)

ax.set_title('Ultra-pure Water', fontsize = ft, pad = 15)
# ax.set_xlabel('')
ax.set_ylabel('Surface Tension [mN/m]', fontsize = ft, labelpad = 15)
ax.set_xscale('log')
ax.set_ylim(25, 80)
ax.set_xlim(10**-5, 10**0)
ax.grid()
ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=16)
ax.legend(loc =3, prop={'size': 22})

ax1.scatter(pfoa_nacl_conc*1000, pfoa_nacl_meas, color=pfoac[2], s = ms, marker = mi2, edgecolor = 'b', zorder = 10,
            label = 'PFOA$_{salt}$')
ax1.scatter(pfoa_bruss_conc*1000, pfoa_bruss_meas, color=pfoac[3], s = ms, marker = 'D', edgecolor = 'b', zorder = 10)
ax1.scatter(pfos_nacl_conc*1000, pfos_nacl_meas, color=pfosc[2], s = ms, marker = mi2, edgecolor = 'b', zorder = 10,
            label = 'PFOS$_{salt}$')
ax1.scatter(pfhxs_nacl_conc*1000, pfhxs_nacl_meas, color=pfhxsc[2], s = ms, marker = mi2, edgecolor = 'b', zorder = 10,
            label = 'PFHxS$_{salt}$')
ax1.scatter(pfda_nacl_conc*1000, pfda_nacl_meas, color=pfdac[2], s = ms, marker = mi2, edgecolor = 'b', zorder = 10,
            label = 'PFDA$_{salt}$')

ax1.plot(conc_model*1000, pfoa_nacl_sft, color=pfoac[2], lw = lw, ls = ls)
ax1.plot(conc_model*1000, pfos_nacl_sft, color=pfosc[2], lw = lw, ls = ls)
ax1.plot(conc_model*1000, pfhxs_nacl_sft, color=pfhxsc[2], lw = lw, ls = ls)
ax1.plot(conc_model*1000, pfda_nacl_sft, color=pfdac[2], lw = lw, ls = ls)

ax1.set_title('0.1 M NaCl', fontsize = ft, pad = 15)
ax1.set_xscale('log')
ax1.set_ylim(25, 80)
ax1.set_xlim(10**-5, 10**0)
ax1.grid()
ax1.tick_params(axis='y', labelsize=18)
ax1.tick_params(axis='x', labelsize=16)
ax1.legend(loc =3, prop={'size': 20})

# - kaw
# fig2, (ax2, ax3) =  plt.subplots(1, 2, figsize=(14, 6), dpi=200)

#ultra-pure
ax2.plot(conc_model*1000, pfoa_kaw, color=pfoac[2], lw = lw)
ax2.plot(conc_model*1000, pfos_kaw, color=pfosc[2], lw = lw)
ax2.plot(conc_model*1000, pfhxs_kaw, color=pfhxsc[2], lw = lw)
ax2.plot(conc_model*1000, pfda_kaw, color=pfdac[2], lw = lw)

ax2.set_xlabel('Concentration [mol/m$^3$]', fontsize = ft)
ax2.set_ylabel('Kaw [m]', fontsize = ft)
ax2.tick_params(axis='y', labelsize=18)
ax2.tick_params(axis='x', labelsize=16)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(10**-7, 10**-3)
ax2.grid()

#salt
ax3.plot(conc_model*1000, pfoa_nacl_kaw, color=pfoac[2], lw = lw)
ax3.plot(conc_model*1000, pfos_nacl_kaw, color=pfosc[2], lw = lw)
ax3.plot(conc_model*1000, pfhxs_nacl_kaw, color=pfhxsc[2], lw = lw)
ax3.plot(conc_model*1000, pfda_nacl_kaw, color=pfdac[2], lw = lw)

ax3.set_xlabel('Concentration [mol/m$^3$]', fontsize = ft)
ax3.tick_params(axis='y', labelsize=18)
ax3.tick_params(axis='x', labelsize=16)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_ylim(10**-7, 10**-3)
ax3.grid()