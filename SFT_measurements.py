# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:37:09 2022
Surface Tension Only
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

#import functions
from vadose_zone_adsorption_functions import *
#from ADRs import *

#%matplotlib inline
#%matplotlib qt
#%% - all data - concentration range
conc_all = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0])
conc = np.array([0.1, 0.5, 1, 5, 10, 50, 100]) #mg/L
conc_salt = np.array([0.09, 0.45, 0.9, 4.5, 9, 45, 90])

modhigh = 1 #model max limits
modlow = 0.0001 #model max limits
iterations = 500
anion = 100 #Cl- concentration
cation = 100 #Na+ concentration
#%% Pfda - 0.1 NaCl
pfda_mw = 514080 #mg/mol
pfda_conc_nacl = np.multiply(conc_salt, 1000/pfda_mw) #mol/m3
pfda_conc_nacl_model = np.linspace(min(pfda_conc_nacl), 1, num = iterations)
pfda_conc_nacl_star = (conc_star(pfda_conc_nacl_model, anion, cation))**2 #note C* is mean ionic activity squared

#PFda measured SFT
pfda_nacl_wtr = 73.283
SFT_pfda_nacl = ([72.023, 72.54, 72.652, 71.324, 71.777, 72.205, 70.551, 70.391, 71.431, 55.224, 57.725, 58.517, 49.965, 51.385, 51.597, 30.35, 30.95, 31.539, 29.249, 29.537, 29.629])
SFT_pfda_nacl_ave = trip_averages(SFT_pfda_nacl, 3)

#freundlich sft predictions
pfda_nacl_k = 7.44*10**-6#5.33*10**-6
pfda_nacl_n = 0.917#0.685
pfda_sft_nacl_f = freundlich_sft(pfda_nacl_wtr, 293.22, pfda_conc_nacl_star, pfda_nacl_k, pfda_nacl_n)

#langmuir sft predictions
pfda_nacl_T_max = 7.84*10**-6
pfda_nacl_a = 2.41
pfda_sft_nacl_l = langmuir_sft(pfda_nacl_wtr, 293.22, pfda_conc_nacl_star, pfda_nacl_T_max, pfda_nacl_a)

#surf ex and kaw (freundllch vs langmuir)
pfda_se_nacl_f, pfda_kaw_nacl_f = freundlich_kaw(pfda_nacl_k, pfda_nacl_n, pfda_conc_nacl_star, pfda_conc_nacl_model)
pfda_se_nacl_l, pfda_kaw_nacl_l = langmuir_kaw(pfda_nacl_T_max, pfda_nacl_a, pfda_conc_nacl_star, pfda_conc_nacl_model)
#%% Pfos - 0.1 NaCl
pfos_mw = 538220 #mg/mol
pfos_conc_nacl = np.multiply(conc_salt, 1000/pfos_mw) #mol/m3
pfos_conc_nacl_model = np.linspace(min(pfos_conc_nacl), 1, num = iterations)
pfos_conc_nacl_star = (conc_star(pfos_conc_nacl_model+pfos_conc_nacl_model, anion, cation))**2 #output in mol/m3

#PFOA measured SFT
pfos_nacl_wtr = 73.0826
SFT_pfos_nacl = ([72.146, 72.748, 72.792, 71.622, 72.038, 72.312, 67.945, 70.794, 71.28, 64.243, 66.092, 66.577, 59.858, 60.985, 61.133, 51.734, 52.275, 52.381, 51.078, 51.305, 51.362])
SFT_pfos_nacl_ave = trip_averages(SFT_pfos_nacl, 3)

#freundlich sft predictions
pfos_nacl_k = 2.49*10**-6#5.33*10**-6
pfos_nacl_n = 0.757#0.685
pfos_sft_nacl_f = freundlich_sft(pfos_nacl_wtr, 293.16, pfos_conc_nacl_star, pfos_nacl_k, pfos_nacl_n)

#langmuir sft predictions
pfos_nacl_T_max = 7.41*10**-6
pfos_nacl_a = 0.522
pfos_sft_nacl_l = langmuir_sft(pfos_nacl_wtr, 293.16, pfos_conc_nacl_star, pfos_nacl_T_max, pfos_nacl_a)

#surf ex and kaw
pfos_se_nacl_f, pfos_kaw_nacl_f = freundlich_kaw(pfos_nacl_k, pfos_nacl_n, pfos_conc_nacl_star, pfos_conc_nacl_model)
pfos_se_nacl_l, pfos_kaw_nacl_l = langmuir_kaw(pfos_nacl_T_max, pfos_nacl_a, pfos_conc_nacl_star, pfos_conc_nacl_model)
#%% PfHxS - 0.1 NaCl
pfhxs_mw = 438200 #mg/mol
pfhxs_conc_nacl = np.multiply(conc_salt, 1000/pfhxs_mw) #mol/m3
pfhxs_conc_nacl_model = np.linspace(min(pfhxs_conc_nacl), 1, num = iterations)
pfhxs_conc_nacl_star = (conc_star(pfhxs_conc_nacl_model+pfhxs_conc_nacl_model, anion, cation))**2 #output in mol/m3

#PFOA measured SFT
pfhxs_nacl_wtr = 73.276
SFT_pfhxs_nacl = ([73.063, 73.001, 72.887, 72.175, 72.675, 72.799, 72.317, 72.589, 72.317, 70.599, 71.217, 71.225, 69.703, 70.023, 70.034, 65.703, 65.707, 65.722, 62.483, 62.867, 62.987])
SFT_pfhxs_nacl_ave = trip_averages(SFT_pfhxs_nacl, 3)

#freundlich sft predictions
pfhxs_nacl_k = 8.04*10**-7#5.33*10**-6
pfhxs_nacl_n = 0.8#0.685
pfhxs_sft_nacl_f = freundlich_sft(pfhxs_nacl_wtr, 293.16, pfhxs_conc_nacl_star, pfhxs_nacl_k, pfhxs_nacl_n)

#langmuir sft predictions
pfhxs_nacl_T_max = 1.75*10**-6
pfhxs_nacl_a = 1.307
pfhxs_sft_nacl_l = langmuir_sft(pfhxs_nacl_wtr, 293.16, pfhxs_conc_nacl_star, pfhxs_nacl_T_max, pfhxs_nacl_a)

#surf ex and kaw
pfhxs_se_nacl_f, pfhxs_kaw_nacl_f = freundlich_kaw(pfhxs_nacl_k, pfhxs_nacl_n, pfhxs_conc_nacl_star, pfhxs_conc_nacl_model)
pfhxs_se_nacl_l, pfhxs_kaw_nacl_l = langmuir_kaw(pfhxs_nacl_T_max, pfhxs_nacl_a, pfhxs_conc_nacl_star, pfhxs_conc_nacl_model)
#%% PFOA
#PFOA conc range mol/m3 -- activity corrections
pfoa_mw = 438200 #mg/mol
pfoa_conc = np.multiply(conc, 1000/pfoa_mw) #mol/m3
pfoa_conc_model = np.linspace(0.0001, modhigh, num = iterations)
pfoa_conc_star = conc_star(pfoa_conc_model, 0, 0)

#PFOA measured SFT
pfoa_wtr = 72.899
SFT_pfoa = ([72.158, 72.797, 72.841, 72.093, 72.734, 72.83, 72.631, 72.637, 72.749, 72.352, 72.402, 72.692, 70.46, 72.338, 72.373, 70.615, 70.49, 70.417, 66.977, 67.061, 66.463])
SFT_pfoa_ave = trip_averages(SFT_pfoa, 3)

#freundlich sft predictions
pfoa_k = 9.5*10**-6#5.33*10**-6
pfoa_n = 0.962#0.685
pfoa_sft_f = freundlich_sft(pfoa_wtr, 293.73, pfoa_conc_star, pfoa_k, pfoa_n)

#langmuir sft predictions
pfoa_T_max = 6.92*10**-6
pfoa_a = 1.79
pfoa_sft_l = langmuir_sft(pfoa_wtr, 293.73, pfoa_conc_star, pfoa_T_max, pfoa_a)

#surf ex and kaw
pfoa_se_f, pfoa_kaw_f = freundlich_kaw(pfoa_k, pfoa_n, pfoa_conc_star, pfoa_conc_model)
pfoa_se_l, pfoa_kaw_l = langmuir_kaw(pfoa_T_max, pfoa_a, pfoa_conc_star, pfoa_conc_model)
#%% PFDA
#PFDA conc range mol/m3 -- activity corrections
pfda_mw = 514080 #mg/mol
pfda_conc = np.multiply(conc, 1000/pfda_mw) #mol/m3
pfda_conc_model = np.linspace(0.0001, modhigh, num = iterations)
pfda_conc_star = conc_star(pfda_conc_model, 0, 0)

#PFDA measured SFT
pfda_wtr = 72.7945
SFT_pfda = ([72.533, 72.56, 72.632, 72.328, 72.334, 72.437, 72.214, 72.208, 72.429, 71.631, 71.743, 71.399, 71.137, 71.349, 71.391, 57.13, 57.172, 56.95, 45.886, 45.932, 46.057])
SFT_pfda_ave = trip_averages(SFT_pfda, 3)

#freundlich sft predictions
pfda_k = 5.35*10**-5
pfda_n = 0.969
pfda_sft_f = freundlich_sft(pfda_wtr, 293.514, pfda_conc_star, pfda_k, pfda_n)

#langmuir sft predictions
pfda_T_max = 3.77*10**-5
pfda_a = 1.77
pfda_sft_l = langmuir_sft(pfda_wtr, 293.514, pfda_conc_star, pfda_T_max, pfda_a)

#surf ex and kaw
pfda_se_f, pfda_kaw_f = freundlich_kaw(pfda_k, pfda_n, pfda_conc_star, pfda_conc_model)
pfda_se_l, pfda_kaw_l = langmuir_kaw(pfda_T_max, pfda_a, pfda_conc_star, pfda_conc_model)
#%% PFOS
#PFOS conc range mol/m3 -- activity corrections
pfos_mw = 538220 #mg/mol
pfos_conc = np.multiply(conc, 1000/pfos_mw) #mol/m3
pfos_conc_model = np.linspace(0.0001, modhigh, num = iterations)
pfos_conc_star = conc_star(pfos_conc_model, pfos_conc_model, 0)

#PFOS measured SFT
pfos_wtr = 72.7984
SFT_pfos = ([72.603, 72.71, 72.722, 72.6, 72.76, 72.685, 72.5, 72.525, 72.614, 71.981, 72.087, 72.111, 70.847, 71.629, 71.702, 68.179, 69.168, 69.365, 65.108, 65.571, 66.003])
SFT_pfos_ave = trip_averages(SFT_pfos, 3)

#freundlich sft predictions
pfos_k = 8.28*10**-6
pfos_n = 0.85
pfos_sft_f = freundlich_sft(pfos_wtr, 293.751, pfos_conc_star, pfos_k, pfos_n)

#langmuir sft predictions
pfos_T_max = 3.08*10**-6
pfos_a = 6.36
pfos_sft_l = langmuir_sft(pfos_wtr, 293.751, pfos_conc_star, pfos_T_max, pfos_a)

#surf ex and kaw
pfos_se_f, pfos_kaw_f = freundlich_kaw(pfos_k, pfos_n, pfos_conc_star, pfos_conc_model)
pfos_se_l, pfos_kaw_l = langmuir_kaw(pfos_T_max, pfos_a, pfos_conc_star, pfos_conc_model)
#%% PFHxS
#PFHxS conc range mol/m3 -- activity corrections
pfhxs_mw = 438200 #mg/mol
pfhxs_conc = np.multiply(conc, 1000/pfhxs_mw) #mol/m3
pfhxs_conc_model = np.linspace(0.0001, modhigh, num = iterations)
pfhxs_conc_star = conc_star(pfhxs_conc_model, pfhxs_conc_model, 0)

#PFHxS measured SFT
pfhxs_wtr = 72.829
SFT_pfhxs = ([72.689, 72.746, 72.716, 72.43, 72.6, 72.599, 72.261, 72.684, 73.628, 72.437, 72.518, 72.537, 72.479, 72.55, 72.616, 71.963, 71.88, 71.905, 71.2, 71.236, 71.308])
SFT_pfhxs_ave = trip_averages(SFT_pfhxs, 3)

#freundlich sft predictions
pfhxs_k = 1.6*10**-6
pfhxs_n = 0.866
pfhxs_sft_f = freundlich_sft(pfhxs_wtr, 294.027, pfhxs_conc_star, pfhxs_k, pfhxs_n)

#langmuir sft predictions
pfhxs_T_max = 5.94*10**-7
pfhxs_a = 6.378
pfhxs_sft_l = langmuir_sft(pfhxs_wtr, 294.027, pfhxs_conc_star, pfhxs_T_max, pfhxs_a)

#surf ex and kaw
pfhxs_se_f, pfhxs_kaw_f = freundlich_kaw(pfhxs_k, pfhxs_n, pfhxs_conc_star, pfhxs_conc_model)
pfhxs_se_l, pfhxs_kaw_l = langmuir_kaw(pfhxs_T_max, pfhxs_a, pfhxs_conc_star, pfhxs_conc_model)
#%% plotting
pfoacolors = plt.cm.Reds(np.linspace(0,1,5))
pfoscolors = plt.cm.Blues(np.linspace(0,1,5))
pfhxscolors = plt.cm.Greens(np.linspace(0,1,5))
pfdacolors = plt.cm.Purples(np.linspace(0,1,5))
mi = '^' #marker icon
ms = 150 #marker size
lw = 4 #linewidth
ft = 24 #fontsize


#for large comparison plt betwee Freundlichs and Langmuirs in milliQ
# fig1, ([ax01, ax02, ax03],
#        [ax04, ax05, ax06]) =  plt.subplots(2, 3, figsize=(24,14), dpi=200) #figsize=(22,17)

fig1, (ax01, ax02) =  plt.subplots(1, 2, figsize=(16,8), dpi=200)
#fig1, ax01 =  plt.subplots(1, 1, figsize=(8,6), dpi=200)
#----------------------------------------------------------------MilliQ----------------------------------------------------------------
#PFOA
# ax01.scatter(pfoa_conc, SFT_pfoa_ave, marker = mi, s=ms, color=pfoacolors[2], zorder = 10)
# ax01.plot(pfoa_conc_model, pfoa_sft_f, '--', color=pfoacolors[1], label = 'PFOA', linewidth = lw)

ax01.scatter(pfoa_conc, SFT_pfoa_ave, marker = mi, s=ms, color=pfoacolors[2], zorder = 10)
ax01.plot(pfoa_conc_model, pfoa_sft_l, ':', color=pfoacolors[1], label = 'PFOA', linewidth = lw)
#PFDA
# ax01.scatter(pfda_conc, SFT_pfda_ave, marker = mi, s=ms, color=pfdacolors[2], zorder = 8)
# ax01.plot(pfda_conc_model, pfda_sft_f, '--', color=pfdacolors[1], label = 'PFDA', linewidth = lw)

ax01.scatter(pfda_conc, SFT_pfda_ave, marker = mi, s=ms, color=pfdacolors[2], zorder = 8)
ax01.plot(pfda_conc_model, pfda_sft_l, ':', color=pfdacolors[1], label = 'PFDA', linewidth = lw)
#PFOA
# ax01.scatter(pfos_conc, SFT_pfos_ave, marker = mi, s=ms, color=pfoscolors[2], zorder = 9)
# ax01.plot(pfos_conc_model, pfos_sft_f, '--', color=pfoscolors[1], label = 'K-PFOS', linewidth = lw)

ax01.scatter(pfos_conc, SFT_pfos_ave, marker = mi, s=ms, color=pfoscolors[2], zorder = 9)
ax01.plot(pfos_conc_model, pfos_sft_l, ':', color=pfoscolors[1], label = 'K-PFOS', linewidth = lw)
#PFHxS
# ax01.scatter(pfhxs_conc, SFT_pfhxs_ave, marker = mi, s=ms, color=pfhxscolors[2], zorder = 7)
# ax01.plot(pfhxs_conc_model, pfhxs_sft_f, '--', color=pfhxscolors[1], label = 'K-PFHxS', linewidth = lw)

ax01.scatter(pfhxs_conc, SFT_pfhxs_ave, marker = mi, s=ms, color=pfhxscolors[2], zorder = 7)
ax01.plot(pfhxs_conc_model, pfhxs_sft_l, ':', color=pfhxscolors[1], label = 'K-PFHxS', linewidth = lw)
#plot
ax01.set_title('Surface Tension ({})'.format(r'$\sigma$'), pad= 15, fontsize = ft)
ax01.set_ylabel('Surface Tension [mN/m]', fontsize = 20)
ax01.set_ylim([45,74])
ax01.set_xlim([10**-4, 10**0])
ax01.set_xlabel('Concentration [mol/m$^3$]', fontsize = 20)
ax01.set_xscale('log')
ax01.legend()
ax01.grid()

#ax04.set_title('Surface Tension ({})'.format(r'$\sigma$'), pad = 15, fontsize = 16)
# ax04.set_ylabel('Freundlich Model \n Surface Tension [mN/m]', fontsize = ft)
# ax04.set_ylim([45,74])
# ax04.set_xlim([10**-4, 10**0])
# ax04.set_xlabel('Concentration [mol/m$^3$]', fontsize = ft)
# ax04.set_xscale('log')
# ax04.legend()
# ax04.grid()

#surf_ex
# ax02.plot(pfoa_conc_model, pfoa_se_f, color=pfoacolors[2], label = 'PFOA', linewidth = 3)
# ax05.plot(pfoa_conc_model, pfoa_se_l, color=pfoacolors[2], label = 'PFOA', linewidth = 3)

# ax02.plot(pfda_conc_model, pfda_se_f, color=pfdacolors[2], label = 'PFDA', linewidth = 3)
# ax05.plot(pfda_conc_model, pfda_se_l, color=pfdacolors[2], label = 'PFDA', linewidth = 3)

# ax02.plot(pfos_conc_model, pfos_se_f, color=pfoscolors[2], label = 'K-PFOS', linewidth = 3)
# ax05.plot(pfos_conc_model, pfos_se_l, color=pfoscolors[2], label = 'K-PFOS', linewidth = 3)

# ax02.plot(pfhxs_conc_model, pfhxs_se_f, color=pfhxscolors[2], label = 'K-PFHxS', linewidth = 3)
# ax05.plot(pfhxs_conc_model, pfhxs_se_l, color=pfhxscolors[2], label = 'K-PFHxS', linewidth = 3)

# ax02.set_title('Surface Excess ({})'.format(r'$\Gamma$'), pad = 15, fontsize = ft)
# #ax02.set_xlabel('Concentration [mol/m$^3$]', fontsize = 16)
# ax02.set_ylabel('Surface excess [mol/m$^2$]', fontsize = ft)
# ax02.set_xscale('log')
# ax02.set_yscale('log')
# ax02.legend()
# ax02.grid()

#ax05.set_title('Surface Excess ({})'.format(r'$\Gamma$'), pad = 15, fontsize = 16)
# ax05.set_xlabel('Concentration [mol/m$^3$]', fontsize = ft)
# ax05.set_ylabel('Surface excess [mol/m$^2$]', fontsize = ft)
# ax05.set_xscale('log')
# ax05.set_yscale('log')
# ax05.legend()
# ax05.grid()

#kaw
# ax03.plot(pfoa_conc_model, pfoa_kaw_f, color=pfoacolors[2], label = 'PFOA', linewidth = 3)
ax02.plot(pfoa_conc_model, pfoa_kaw_l, color=pfoacolors[2], label = 'PFOA', linewidth = 3)

# ax03.plot(pfda_conc_model, pfda_kaw_f, color=pfdacolors[2], label = 'PFDA', linewidth = 3)
ax02.plot(pfda_conc_model, pfda_kaw_l, color=pfdacolors[2], label = 'PFDA', linewidth = 3)

# ax03.plot(pfos_conc_model, pfos_kaw_f, color=pfoscolors[2], label = 'K-PFOS', linewidth = 3)
ax02.plot(pfos_conc_model, pfos_kaw_l, color=pfoscolors[2], label = 'K-PFOS', linewidth = 3)

# ax03.plot(pfhxs_conc_model, pfhxs_kaw_f, color=pfhxscolors[2], label = 'K-PFHxS', linewidth = 3)
ax02.plot(pfhxs_conc_model, pfhxs_kaw_l, color=pfhxscolors[2], label = 'K-PFHxS', linewidth = 3)

ax02.set_title('Air-water adsorption ({})'.format(r'$K_{aw}$'), pad = 15, fontsize = ft)
ax02.set_xlabel('Concentration [mol/m$^3$]', fontsize = 20)
ax02.set_ylabel('Kaw [m]', fontsize = 20)
ax02.set_xscale('log')
ax02.set_yscale('log')
ax02.set_ylim([10**-7, 10**-4])
ax02.set_xlim([10**-4, 0])
ax02.legend()
ax02.grid()

#ax06.set_title('Air-water partitioning ({})'.format(r'$K_{aw}$'), pad = 15, fontsize = 16)
# ax06.set_xlabel('Concentration [mol/m$^3$]', fontsize = ft)
# ax06.set_ylabel('Kaw [m]', fontsize = ft)
# ax06.set_xscale('log')
# ax06.set_yscale('log')
# ax06.set_ylim([10**-6, 10**-4])
# ax06.legend()
# ax06.grid()


#----------------------------------------------------------------SALT----------------------------------------------------------------
# fig2, ([ax07, ax08]) =  plt.subplots(2, 1, figsize=(8,10), dpi=200)
# #PFDA-NaCl 0.1M
# ax07.scatter(pfda_conc_nacl, SFT_pfda_nacl_ave, marker = 's', s=ms, color=pfdacolors[2], zorder = 10)
# ax07.plot(pfda_conc_nacl_model, pfda_sft_nacl_f, '--', color=pfdacolors[1], label = 'PFDA - 0.1 M NaCl', linewidth = lw)

# ax08.scatter(pfda_conc_nacl, SFT_pfda_nacl_ave, marker = 's', s=ms, color=pfdacolors[2], zorder = 10)
# ax08.plot(pfda_conc_nacl_model, pfda_sft_nacl_l, ':', color=pfdacolors[1], label = 'PFDA - 0.1 M NaCl', linewidth = lw)
# #PFOS-NaCl 0.1M
# ax07.scatter(pfos_conc_nacl, SFT_pfos_nacl_ave, marker = 's', s=ms, color=pfoscolors[2], zorder = 10)
# ax07.plot(pfos_conc_nacl_model, pfos_sft_nacl_f, '--', color=pfoscolors[1], label = 'PFOS - 0.1 M NaCl', linewidth = lw)

# ax08.scatter(pfos_conc_nacl, SFT_pfos_nacl_ave, marker = 's', s=ms, color=pfoscolors[2], zorder = 10)
# ax08.plot(pfos_conc_nacl_model, pfos_sft_nacl_l, ':', color=pfoscolors[1], label = 'PFOS - 0.1 M NaCl', linewidth = lw)
# #PFHxS-NaCl 0.1M
# ax07.scatter(pfhxs_conc_nacl, SFT_pfhxs_nacl_ave, marker = 's', s=ms, color=pfhxscolors[2], zorder = 10)
# ax07.plot(pfhxs_conc_nacl_model, pfhxs_sft_nacl_f, '--', color=pfhxscolors[1], label = 'PFHxS - 0.1 M NaCl', linewidth = lw)

# ax08.scatter(pfhxs_conc_nacl, SFT_pfhxs_nacl_ave, marker = 's', s=ms, color=pfhxscolors[2], zorder = 10)
# ax08.plot(pfhxs_conc_nacl_model, pfhxs_sft_nacl_l, ':', color=pfhxscolors[1], label = 'PFHxS - 0.1 M NaCl', linewidth = lw)


# ax07.set_title('Surface Tension ({})'.format(r'$\sigma$'), fontsize = 16)
# ax07.set_ylabel('Freundlich Model \n Surface Tension [mN/m]', fontsize = 16)
# ax07.set_ylim([25,75])
# ax07.set_xlim([10**-4, 10**0])
# #ax01.set_xlabel('Concentration [mol/m$^3$]', fontsize = 16)
# ax07.set_xscale('log')
# ax07.legend()
# ax07.grid()

# #ax04.set_title('Surface Tension ({})'.format(r'$\sigma$'), pad = 15, fontsize = 16)
# ax08.set_ylabel('Langmuir Model \n Surface Tension [mN/m]', fontsize = 16)
# ax08.set_ylim([25,75])
# ax08.set_xlim([10**-4, 10**0])
# ax08.set_xlabel('Concentration [mol/m$^3$]', fontsize = 16)
# ax08.set_xscale('log')
# ax08.legend()
# ax08.grid()

#Figure 3 - PFAS in 0.1 M NaCl solution surface tension comparisions to Kaw
fig3, ([ax09, ax010]) =  plt.subplots(1, 2, figsize=(16,8), dpi=200)

# ax03.plot(pfoa_conc_model, pfoa_kaw_f, color=pfoacolors[2], label = 'PFOA', linewidth = 3)

ax09.scatter(pfda_conc_nacl, SFT_pfda_nacl_ave, marker = 's', s=ms, color=pfdacolors[2], zorder = 10)
ax09.plot(pfda_conc_nacl_model, pfda_sft_nacl_l, ':', color=pfdacolors[1], label = 'PFDA - 0.1 M NaCl', linewidth = lw)
ax010.plot(pfda_conc_model, pfda_kaw_nacl_l, color=pfdacolors[2], label = 'PFDA - 0.1 M NaCl', linewidth = 3)

ax09.scatter(pfos_conc_nacl, SFT_pfos_nacl_ave, marker = 's', s=ms, color=pfoscolors[2], zorder = 10)
ax09.plot(pfos_conc_nacl_model, pfos_sft_nacl_l, ':', color=pfoscolors[1], label = 'K-PFOS - 0.1 M NaCl', linewidth = lw)
ax010.plot(pfos_conc_model, pfos_kaw_nacl_l, color=pfoscolors[2], label = 'K-PFOS - 0.1 M NaCl', linewidth = 3)

ax09.scatter(pfhxs_conc_nacl, SFT_pfhxs_nacl_ave, marker = 's', s=ms, color=pfhxscolors[2], zorder = 10)
ax09.plot(pfhxs_conc_nacl_model, pfhxs_sft_nacl_l, ':', color=pfhxscolors[1], label = 'K-PFHxS - 0.1 M NaCl', linewidth = lw)
ax010.plot(pfhxs_conc_model, pfhxs_kaw_nacl_l, color=pfhxscolors[2], label = 'K-PFHxS - 0.1 M NaCl', linewidth = 3)

ax09.set_title('Surface Tension ({})'.format(r'$\sigma$'), pad = 15, fontsize = ft)
ax09.set_ylabel('Surface Tension [mN/m]', fontsize = 20)
ax09.set_ylim([25,75])
ax09.set_xlim([10**-4, 10**0])
ax09.set_xlabel('Concentration [mol/m$^3$]', fontsize = 20)
ax09.set_xscale('log')
ax09.legend()
ax09.grid()

ax010.set_title('Air-water adsorption ({})'.format(r'$K_{aw}$'), pad = 15, fontsize = ft)
ax010.set_xlabel('Concentration [mol/m$^3$]', fontsize = 20)
ax010.set_ylabel('Kaw [m]', fontsize = 20)
ax010.set_xscale('log')
ax010.set_yscale('log')
ax010.set_xlim(10**-4, 10**0)
ax010.legend()
ax010.grid()
#%% Figure 4 plotted together
fig4, ([axA, axB]) =  plt.subplots(1, 2, figsize=(16,8), dpi=200)

#MilliQ data
#Surface Tension
#PFOA
axA.scatter(pfoa_conc, SFT_pfoa_ave, marker = mi, s=ms, color=pfoacolors[2], zorder = 10)
axA.plot(pfoa_conc_model, pfoa_sft_l, ':', color=pfoacolors[1], label = 'PFOA', linewidth = lw)
#PFDA
axA.scatter(pfda_conc, SFT_pfda_ave, marker = mi, s=ms, color=pfdacolors[2], zorder = 8)
axA.plot(pfda_conc_model, pfda_sft_l, ':', color=pfdacolors[1], label = 'PFDA', linewidth = lw)
#PFOS
axA.scatter(pfos_conc, SFT_pfos_ave, marker = mi, s=ms, color=pfoscolors[2], zorder = 9)
axA.plot(pfos_conc_model, pfos_sft_l, ':', color=pfoscolors[1], label = 'K-PFOS', linewidth = lw)
#PFHxS
axA.scatter(pfhxs_conc, SFT_pfhxs_ave, marker = mi, s=ms, color=pfhxscolors[2], zorder = 7)
axA.plot(pfhxs_conc_model, pfhxs_sft_l, ':', color=pfhxscolors[1], label = 'K-PFHxS', linewidth = lw)
#With SALF
#PFDA
axA.scatter(pfda_conc_nacl, SFT_pfda_nacl_ave, marker = 's', s=ms, color=pfdacolors[2], zorder = 10)
axA.plot(pfda_conc_nacl_model, pfda_sft_nacl_l, ':', color=pfdacolors[1], label = 'PFDA - 0.1 M NaCl', linewidth = lw)
#PFOS
axA.scatter(pfos_conc_nacl, SFT_pfos_nacl_ave, marker = 's', s=ms, color=pfoscolors[2], zorder = 10)
axA.plot(pfos_conc_nacl_model, pfos_sft_nacl_l, ':', color=pfoscolors[1], label = 'K-PFOS - 0.1 M NaCl', linewidth = lw)
#PFHxS
axA.scatter(pfhxs_conc_nacl, SFT_pfhxs_nacl_ave, marker = 's', s=ms, color=pfhxscolors[2], zorder = 10)
axA.plot(pfhxs_conc_nacl_model, pfhxs_sft_nacl_l, ':', color=pfhxscolors[1], label = 'K-PFHxS - 0.1 M NaCl', linewidth = lw)
#plot stuff
axA.set_title('Surface Tension ({})'.format(r'$\sigma$'), fontsize = 16)
axA.set_ylabel('Surface Tension [mN/m]', fontsize = 16)
axA.set_ylim([25,75])
axA.set_xlim([10**-4, 10**0])
axA.set_xlabel('Concentration [mol/m$^3$]', fontsize = 20)
axA.set_xscale('log')
axA.legend()
axA.grid()

#KAW
#PFOA
axB.plot(pfoa_conc_model, pfoa_kaw_l, color=pfoacolors[2], label = 'PFOA', linewidth = 3)
#PFDA
axB.plot(pfda_conc_model, pfda_kaw_l, color=pfdacolors[2], label = 'PFDA', linewidth = 3)
#PFOS
axB.plot(pfos_conc_model, pfos_kaw_l, color=pfoscolors[2], label = 'K-PFOS', linewidth = 3)
#PFHxS
axB.plot(pfhxs_conc_model, pfhxs_kaw_l, color=pfhxscolors[2], label = 'K-PFHxS', linewidth = 3)

axB.plot(pfda_conc_model, pfda_kaw_nacl_l, color=pfdacolors[2], label = 'PFDA - 0.1 M NaCl', linewidth = 3)

axB.plot(pfos_conc_model, pfos_kaw_nacl_l, color=pfoscolors[2], label = 'K-PFOS - 0.1 M NaCl', linewidth = 3)

axB.plot(pfhxs_conc_model, pfhxs_kaw_nacl_l, color=pfhxscolors[2], label = 'K-PFHxS - 0.1 M NaCl', linewidth = 3)

axB.set_title('Air-water adsorption ({})'.format(r'$K_{aw}$'), pad = 15, fontsize = ft)
axB.set_xlabel('Concentration [mol/m$^3$]', fontsize = 20)
axB.set_ylabel('Kaw [m]', fontsize = 20)
axB.set_xscale('log')
axB.set_yscale('log')
axB.legend()
axB.grid()
