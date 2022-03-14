# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:56:31 2022

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
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#import other scripts
from vadose_zone_adsorption_functions import *
from SFT_measurements import *
from ADRs import *

#%matplotlib inline
#%matplotlib qt
#%% 2D ORA Awi and Retardation plots
Pc = np.linspace(0, 300,100) #Actual HAW
Pc_kpa = np.multiply(Pc, 1/10.194)
haw = (Pc_kpa / 9.81)*100
a_val = 0.4 #entry pressure of 4 kPa
n_val = 4.0
RH_U = 5.08
phi = 0.42
rhob = 1.8

fig_RH, (ax01, ax02, ax03) =  plt.subplots(1, 3, figsize=(18, 8), dpi=200)

Sw = normSw_func(a_val, n_val, Pc_kpa, 1, 0) #no residual

##Aia vs Sw (ax02)
Aia = Aia_Sw_func2(1, RH_U, Sw, 0.0, 0)
ax01.plot(Sw, Aia, linewidth = 3, label = 'O.R.A.')
ax01.set_title('Air Water Interfacial Area \n water saturation relationship', pad=15)
ax01.set_xlim(-0.05,1.05)
ax01.set_ylabel('Relative $A_{wi}$ [-]')
ax01.set_xlabel('Water saturation [-]')
ax01.grid()

#Aia vs Pc (ax02)
Aia = Aia_Sw_func2(1, RH_U, Sw, 0, 0)
#Aia_hayhook = Aia_Sw_func2(1, 16, Sw_hayhook, 0, 0)
ax02.plot(Pc_kpa, Aia, linewidth = 3, label = 'O.R.A.')
#ax00.plot(Sw_hayhook, Aia_hayhook, linewidth = 3, label = 'O.R.A.')
ax02.set_title('Air Water Interfacial Area \n capillary pressure relationship', pad=15)
ax02.set_ylim(-0.05,1.05)
ax02.set_xlabel('Capillary Pressure [kPa]')
ax02.set_ylabel('Relative $A_{wi}$ [-]')
ax02.grid()

#R Pc (ax003)
C_dat_loop = [0.00001, 0.0000796, 0.0001, 0.001, 0.01, 0.1] #mg/L
C_labels = ["10 ppt", '79.6 ppt', "100 ppt", '1000 ppt', '10000 ppt', '100000 ppt']

ppt_pfoa = np.multiply(C_dat_loop,(1000/pfoa_mw)) #in units of mol/m3
ppt_star_pfoa = conc_star(ppt_pfoa, 0, 0)
#ppt_star_pfoa_nacl = conc_star(ppt_pfoa, 0, 0)

ppt_pfda = np.multiply(C_dat_loop,(1000/pfda_mw))
ppt_star_pfda = conc_star(ppt_pfda, 0, 0)
ppt_star_pfda_nacl = (conc_star(ppt_pfoa, 100, 100))**2

ppt_pfos = np.multiply(C_dat_loop,(1000/pfos_mw))
ppt_star_pfos = conc_star(ppt_pfos, ppt_pfos, 0)
ppt_star_pfos_nacl = (conc_star(ppt_pfoa, 100, 0))**2

ppt_pfhxs = np.multiply(C_dat_loop,(1000/pfhxs_mw))
ppt_star_pfhxs = conc_star(ppt_pfhxs, ppt_pfhxs, 0)
ppt_star_pfhxs_nacl = (conc_star(ppt_pfhxs, 100, 0))**2

#concentration too use from C_dat_loop
C = 1

#calulating KAW
#pfoa_se_f, pfoa_kaw_f = freundlich_kaw(pfoa_k, pfoa_n, ppt_star_pfoa[C], ppt_pfoa[C])
pfoa_se_l, pfoa_kaw_l = langmuir_kaw(pfoa_T_max, pfoa_a, ppt_star_pfoa[C], ppt_pfoa[C])

#pfda_se_f, pfda_kaw_f = freundlich_kaw(pfda_k, pfda_n, ppt_star_pfda[C], ppt_pfda[C])
pfda_se_l, pfda_kaw_l = langmuir_kaw(pfda_T_max, pfda_a, ppt_star_pfda[C], ppt_pfda[C])
pfda_se_nacl_l, pfda_kaw_nacl_l = langmuir_kaw(pfda_T_max, pfda_nacl_a, ppt_star_pfda_nacl[C], ppt_pfda[C])
pfda_kaw_nacl_l = 1*10**-3
#pfos_se_f, pfos_kaw_f = freundlich_kaw(pfos_k, pfos_n, ppt_star_pfos[C], ppt_pfos[C])
pfos_se_l, pfos_kaw_l = langmuir_kaw(pfos_T_max, pfos_a, ppt_star_pfos[C], ppt_pfos[C])
pfos_se_nacl_l, pfos_kaw_nacl_l = langmuir_kaw(pfos_T_max, pfos_nacl_a, ppt_star_pfos_nacl[C], ppt_pfos[C])
pfos_kaw_nacl_l = 5*10**-4
#pfhxs_se_f, pfhxs_kaw_f = freundlich_kaw(pfhxs_k, pfhxs_n, ppt_star_pfhxs[C], ppt_pfhxs[C])
pfhxs_se_l, pfhxs_kaw_l = langmuir_kaw(pfhxs_T_max, pfhxs_a, ppt_star_pfhxs[C], ppt_pfhxs[C])
pfhxs_kaw_nacl_l = 1*10**-4
#specific surface area of ORA sediments
s = 90000

#calulatin retardation
# R_PFOA_f = 1 + (pfoa_kaw_f*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
# R_PFDA_f = 1 + (pfda_kaw_f*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
# R_PFOS_f = 1 + (pfos_kaw_f*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
# R_PFHxS_f = 1 + (pfhxs_kaw_f*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
# freundlich_r = np.asarray([R_PFOA_f, R_PFDA_f, R_PFOS_f, R_PFHxS_f])
#solid phase
OC_dis = (1*10**-6)*np.exp(0.3621*(Pc_kpa))

Koc_array = np.array([96, 710])
pfas_names = np.array(['PFOA', 'PFOS'])

Kd_pfoa = Koc_array[0] * OC_dis + 0.3
Kd_pfos =  Koc_array[1] * OC_dis

R_sp_pfoa = Kd_pfoa*rhob*(Sw*phi)
R_sp_pfos = Kd_pfos*rhob*(Sw*phi)

#air-water
R_pfoa_l = (pfoa_kaw_l*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
R_pfda_l = 1+(pfda_kaw_l*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
R_pfda_nacl_l = 1+(pfda_kaw_nacl_l*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
R_PFOS_l = (pfos_kaw_l*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
R_PFOS_nacl_l = 1+(pfos_kaw_nacl_l*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
R_PFHxS_l = 1+(pfhxs_kaw_l*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)
R_PFHxS_nacl_l = 1+(pfhxs_kaw_nacl_l*100) * Aia_Sw_func2(s, 5.08, Sw, 0, 0)

R_tot_PFOA = 1 + R_sp_pfoa + R_pfoa_l
R_tot_PFOS = 1 +  R_sp_pfos + R_PFOS_l

#ax03.plot(Pc_kpa, R_PFHxS_f, color=pfhxscolors[2], label = 'K-PFHxS', linewidth = 3)
#ax03.plot(Pc_kpa, R_PFOS_f, color=pfoscolors[2], label = 'K-PFOS', linewidth = 3)
#ax03.plot(Pc_kpa, R_PFOA_f, color=pfoacolors[2], label = 'PFOA', linewidth = 3)
#ax03.plot(Pc_kpa, R_PFDA_f, color=pfdacolors[2], label = 'PFDA', linewidth = 3)

ax03.plot(Pc_kpa, R_PFHxS_l, color=pfhxscolors[2], label = 'K-PFHxS', linewidth = 3)
ax03.plot(Pc_kpa, R_PFHxS_nacl_l, color=pfhxscolors[1], label = 'K-PFHxS', linewidth = 3)
ax03.plot(Pc_kpa, R_tot_PFOS, color=pfoscolors[2], label = 'K-PFOS', linewidth = 3)
ax03.plot(Pc_kpa, R_PFOS_nacl_l, color=pfoscolors[1], label = 'K-PFOS', linewidth = 3)
ax03.plot(Pc_kpa, R_tot_PFOA, color=pfoacolors[2], label = 'PFOA', linewidth = 3)
ax03.plot(Pc_kpa, R_pfda_l, color=pfdacolors[2], label = 'PFDA', linewidth = 3)
ax03.plot(Pc_kpa, R_pfda_nacl_l, color=pfdacolors[1], label = 'PFDA', linewidth = 3)
    
ax03.set_title('Air-water Retardation \n @ $C_0$ = {} mg/L'.format(C_labels[C]), pad=15)
ax03.set_ylabel('Retardation Factor [-]')
ax03.set_xlabel('Capillary Pressure [kPa]')
ax03.set_ylim(1, 700)
ax03.grid()
xvals = [20,20,20,20,20,20,20]
labelLines(ax03.get_lines(),fontsize=16, align=False, xvals=xvals)
fig_RH.tight_layout()

# #testing axis break
# fig_test, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=200)
# ax1.plot(Pc_kpa, R_PFHxS_l, color=pfhxscolors[2], label = 'K-PFHxS', linewidth = 3)
# ax1.plot(Pc_kpa, R_PFHxS_nacl_l, color=pfhxscolors[1], label = 'K-PFHxS', linewidth = 3)
# ax1.plot(Pc_kpa, R_PFOS_l, color=pfoscolors[2], label = 'K-PFOS', linewidth = 3)
# ax1.plot(Pc_kpa, R_PFOS_nacl_l, color=pfoscolors[1], label = 'K-PFOS', linewidth = 3)
# ax1.plot(Pc_kpa, R_PFOA_l, color=pfoacolors[2], label = 'PFOA', linewidth = 3)
# ax1.plot(Pc_kpa, R_PFDA_l, color=pfdacolors[2], label = 'PFDA', linewidth = 3)
# ax1.plot(Pc_kpa, R_PFDA_nacl_l, color=pfdacolors[1], label = 'PFDA', linewidth = 3)

# ax2.plot(Pc_kpa, R_PFHxS_l, color=pfhxscolors[2], label = 'K-PFHxS', linewidth = 3)
# ax2.plot(Pc_kpa, R_PFHxS_nacl_l, color=pfhxscolors[1], label = 'K-PFHxS', linewidth = 3)
# ax2.plot(Pc_kpa, R_PFOS_l, color=pfoscolors[2], label = 'K-PFOS', linewidth = 3)
# ax2.plot(Pc_kpa, R_PFOS_nacl_l, color=pfoscolors[1], label = 'K-PFOS', linewidth = 3)
# ax2.plot(Pc_kpa, R_PFOA_l, color=pfoacolors[2], label = 'PFOA', linewidth = 3)
# ax2.plot(Pc_kpa, R_PFDA_l, color=pfdacolors[2], label = 'PFDA', linewidth = 3)
# ax2.plot(Pc_kpa, R_PFDA_nacl_l, color=pfdacolors[1], label = 'PFDA', linewidth = 3)

# ax1.set_ylim(4000, 9000)
# ax2.set_ylim(1, 1000)

#%% Vadose Zone surface plot
#Pc
Pc = np.linspace(0, 300, 200) #Actual HAW
Pc_kpa = np.multiply(Pc, 1/10.194)
haw = (Pc_kpa / 9.81)*100
phi = 0.42
rhob = 1.8

#Entry pressure
a = np.linspace(0, 1, 200)

#Saturation surface
a_surf, Pc_surf = np.meshgrid(a, Pc_kpa, sparse=False, indexing='xy') 
Sw_surf = Sw_func(a_surf, 4, Pc_surf, 0.05)
#Sw_surf = np.where(Sw_surf <= 0.125, 0.125, Sw_surf)

Sw_vol_surf =np.multiply(Sw_surf, phi)
Sw_vol_surf =np.where(Sw_vol_surf <= 0.05*phi, 0.05*phi, Sw_vol_surf)


#Air-water Interfacial area associated retardation
Aia_surf = Aia_Sw_func2(s, RH_U, Sw_surf, 0.0, 0.0)
#Kaw  associated with PFA concentration
PFOS_Kaw_cm = pfos_kaw_l*100
R_aw = PFOS_Kaw_cm * Aia_surf/Sw_vol_surf

#Organic Carbon and total associated Retardation
#OC distribution in 1D
OC_dis = (1*10**-6)*np.exp(0.03621*(Pc_surf))
logKoc = 2.8 #literature values for PFAS
Kd_OC = 10**logKoc * OC_dis
Kd_min = 0.3
R_oc = Kd_OC*rhob/(Sw_vol_surf)
R_sp = Kd_min*rhob/(Sw_vol_surf)
R_sp_tot = R_oc + R_sp

#total R
R = 1 + R_aw + R_oc + R_sp

for i in a:
    for j in Pc_kpa:
        Sw = Sw_func(a, 4, Pc_kpa, 0.0)
        
# for i in a:
#     for j in Pc_kpa:
#         Sw1 = Sw_func(a, 4, Pc_kpa, 0.12)

#fig_R, (ax, ax1, ax2) =  plt.subplots(1, 3, figsize=(27, 8), dpi=200)
fig_R, (ax, ax1) =  plt.subplots(1, 2, figsize=(14, 6), dpi=200)
fig_R, ax2 =  plt.subplots(1, 1, figsize=(7, 6), dpi=200)

Lev1 = [1, 800 ,1600, 2400, 3200, 4000, round(np.amax(1+R_aw))]
Lev2 = [1, 5, 10, 15, 20, 25, round(np.amax(1+R_sp_tot))]
Lev3 = [1, 800 ,1600, 2400, 3200, 4000, round(np.amax(R))]
im = ax.contourf(Sw, haw, 1+R_aw, Lev1, cmap = 'Reds') #norm = colors.BoundaryNorm(boundaries=Lev1, ncolors=256, extend='both'), cmap = 'Reds') #Air-water Retardation
im2 = ax1.contourf(Sw, haw, 1+R_sp_tot, Lev2, cmap = 'Blues') #norm = colors.BoundaryNorm(boundaries=Lev2, ncolors=256, extend='both'), cmap = 'Blues') #Solid-phase retardation
im3 = ax2.contourf(Sw, haw, R, Lev3, cmap = 'Purples') #norm = colors.BoundaryNorm(boundaries=Lev3, ncolors=256, extend='both'), cmap = 'Purples') #combined retardation

cbar = fig_R.colorbar(im, ax=ax, label='Retardation Factor [-]')
cbar2 = fig_R.colorbar(im2, ax=ax1, extend='max', label='Retardation Factor [-]')
cbar3 = fig_R.colorbar(im3, ax=ax2, label='Retardation Factor [-]')
#im2.set_clim(1, 10)
ax.set_title('PFOS Retardation@ $C_0$ = {} mg/L \n Air-water Adsorption'.format(C_labels[C]), pad =15)
ax1.set_title('PFOS Retardation@ $C_0$ = {} mg/L \n Solid-phase (OC+Min) Adsorption'.format(C_labels[C]), pad =15)
ax2.set_title('PFOS Retardation@ $C_0$ = {} mg/L \n Combined Adsorption'.format(C_labels[C]), pad =15)
ax.set_ylabel('Height above water table [cm]')
ax.set_xlabel('Sw [-]')
ax1.set_xlabel('Sw [-]')
ax2.set_xlabel('Sw [-]')
ax2.set_ylabel('Height above water table [cm]')

ax2.plot(Sw_func(0.4, 4, Pc_kpa, 0.125), haw, "--", linewidth = 2, color = 'black', label = "Sand SWCC")
ax2.plot(Sw_func(0.2, 4, Pc_kpa, 0.39), haw, "--", linewidth = 2, color = 'grey', label = "Silty-sand SWCC")
#ax2.plot(Sw_func(0.8, 4, Pc_kpa, 0.035), haw, "--", color = 'grey', label = "ORA Gravely-sand SWCC")
labelLines(ax2.get_lines(),fontsize=14, align=True, xvals=[0.5, 0.7])
#labelLines(ax2.get_lines(), fontsize=14, shrink_factor=1)
fig_R.tight_layout()
#%%
#Looking at PFOA retardation across all Kaw values
figR, ([ax, ax1], [ax0, ax2]) = plt.subplots(2, 2, figsize=(18, 16), dpi=200)
figR, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)


Pc = np.linspace(0,300,100)
Pc_kpa = np.linspace(0,30,100)
Sw = normSw_func(a_val, n_val, Pc_kpa, 1, 0)
#C_loop = np.logspace(-4, 0, 7)
C_dat_loop2 = [0.000001, 0.001] #mg/L
C_labels2 = ["1 ppt", '1000 ppt']

ppt_pfoa = np.multiply(C_dat_loop2,(1000/pfoa_mw))
ppt_star_pfoa = conc_star(ppt_pfoa, 0, 0)

ppt_pfda = np.multiply(C_dat_loop2,(1000/pfda_mw))
ppt_star_pfda = conc_star(ppt_pfda, 0, 0)

ppt_pfos = np.multiply(C_dat_loop2,(1000/pfos_mw))
ppt_star_pfos = conc_star(ppt_pfos, 0, 0)
pfos_M = np.multiply(C_dat_loop,(pfos_mw)) #mol/L
n_sp = 1.184
k_sp = 4.93
kd = k_sp*pfos_M[C]**(n_sp-1) #L/kg -- solid phase
phi = 0.42 #porosity

ppt_pfhxs = np.multiply(C_dat_loop2,(1000/pfhxs_mw))
ppt_star_pfhxs = conc_star(ppt_pfhxs, 0, 0)


for i in range(0, len(C_dat_loop2), 1):
    pfoa_se_f, pfoa_kaw_f = freundlich_kaw(pfoa_k, pfoa_n, ppt_star_pfoa[i], ppt_pfoa[i])
    pfoa_se_l, pfoa_kaw_l = langmuir_kaw(pfoa_T_max, pfoa_a, pfoa_conc_star[i], pfoa_conc_model[i])
    R_PFOA_f = 1 + (pfoa_kaw_f*100) * Aia_Sw_func2(90000, 5.08, Sw, 0, 0)
    R_PFOA_l = 1 + (pfoa_kaw_l*100) * Aia_Sw_func2(90000, 5.08, Sw, 0, 0)
    ax.plot(R_PFOA_f, Pc, color=pfoacolors[1], label = C_labels2[i], linewidth = 3)
    ax.plot(R_PFOA_l, Pc, color=pfoacolors[2], label = C_labels2[i], linewidth = 3)
  
for i in range(0,len(C_dat_loop2),1):
    pfda_se_f, pfda_kaw_f = freundlich_kaw(pfda_k, pfda_n, ppt_star_pfda[i], ppt_pfda[i])
    pfda_se_l, pfda_kaw_l = langmuir_kaw(pfda_T_max, pfda_a, pfda_conc_star[i], pfda_conc_model[i])
    R_PFDA_f = 1 + (pfda_kaw_f*100) * Aia_Sw_func2(90000, 5.08, Sw, 0, 0)
    R_PFDA_l = 1 + (pfda_kaw_l*100) * Aia_Sw_func2(90000, 5.08, Sw, 0, 0)
    ax.plot(R_PFDA_f, Pc, color=pfdacolors[1], label = C_labels2[i], linewidth = 3)
    ax.plot(R_PFDA_l, Pc, color=pfdacolors[2], label = C_labels2[i], linewidth = 3)
  
for i in range(0,len(C_dat_loop2),1):
    #kd = k_sp*np.multiply(ppt_pfos[i], 1/1000)**(n_sp-1)
    pfos_se_f, pfos_kaw_f = freundlich_kaw(pfos_k, pfos_n, ppt_star_pfos[i], ppt_pfos[i])
    pfos_se_l, pfos_kaw_l = langmuir_kaw(pfos_T_max, pfos_a, pfos_conc_star[i], pfos_conc_model[i])
    R_PFOS_f = 1 + ((pfos_kaw_f*100) * Aia_Sw_func2(90000, 5.08, Sw*phi, 0, 0)) #+ (kd*1.8*(Sw/phi))
    R_PFOS_l = 1 + ((pfos_kaw_l*100) * Aia_Sw_func2(90000, 5.08, Sw*phi, 0, 0)) #+ (kd*1.8*(Sw/phi))
    ax.plot(R_PFOS_f, Pc, color=pfoscolors[1], label = C_labels2[i], linewidth = 3) 
    ax.plot(R_PFOS_l, Pc, color=pfoscolors[2], label = C_labels2[i], linewidth = 3) 
  
for i in range(0,len(C_dat_loop2),1):
    pfhxs_se_f, pfhxs_kaw_f = freundlich_kaw(pfhxs_k, pfhxs_n, ppt_star_pfhxs[i], ppt_pfhxs[i])
    pfhxs_se_l, pfhxs_kaw_l = langmuir_kaw(pfhxs_T_max, pfhxs_a, pfhxs_conc_star[i], pfhxs_conc_model[i])
    R_PFHxS_f = 1 + (pfhxs_kaw_f*100) * Aia_Sw_func2(90000, 5.08, Sw, 0, 0)
    R_PFHxS_l = 1 + (pfhxs_kaw_l*100) * Aia_Sw_func2(90000, 5.08, Sw, 0, 0)
    ax.plot(R_PFHxS_f, Pc, color=pfhxscolors[1], label = C_labels2[i], linewidth = 3)
    ax.plot(R_PFHxS_l, Pc, color=pfhxscolors[2], label = C_labels2[i], linewidth = 3)

ax.set_title('PFOA Retardation at different $C_0$')
ax.set_xlabel('Retardation Factor [-]')
ax.set_ylabel('Heigth above water table [cm]')
ax.set_ylim(0, 300)
#ax.set_xlim(0, 700)
ax.grid()
#labelLines(ax.get_lines(), shrink_factor= 1, fontsize=20) 

# ax0.set_title('PFDA Retardation at different $C_0$')
# ax0.set_xlabel('Retardation Factor [-]')
# ax0.set_ylabel('Heigth above water table [cm]')
# ax0.set_ylim(0, 300)
# ax0.grid()
# labelLines(ax0.get_lines(), shrink_factor= 1, fontsize=20) 

# ax1.set_title('PFOS Retardation at different $C_0$')
# ax1.set_xlabel('Retardation Factor [-]')
# ax1.set_ylabel('Heigth above water table [cm]')
# ax1.set_ylim(0, 300)
# ax1.grid()
# labelLines(ax1.get_lines(), shrink_factor= 1, fontsize=20) 

# ax2.set_title('PFHxS Retardation at different $C_0$')
# ax2.set_xlabel('Retardation Factor [-]')
# ax2.set_ylabel('Heigth above water table [cm]')
# ax2.set_ylim(0, 300)
# ax2.grid()
# labelLines(ax2.get_lines(), shrink_factor= 1, fontsize=20)
#%% - PFOS follow Sw on van genuchten NOTe implies homogeneous
Sw = np.array([0.16, 0.165, 0.3]) #degree of saturation
rhob = 1.8 #kg/L - bulk density

C_dat_loop = [0.00001, 0.0000796, 0.0001, 0.001, 0.01, 0.1] #mg/L
C_labels = ["10 ppt", '79.6 ppt', "100 ppt", '1000 ppt', '10000 ppt', '100000 ppt']
pfos_M = np.multiply(C_dat_loop,(1/pfos_mw)) #mol/L
n_sp = 1.27#1.184
k_sp = 21.17#4.93
kd = k_sp*pfos_M**(n_sp-1) #L/kg -- solid phase
phi = 0.42 #porosity
i = 1
Rsp1 = kd[i]*(rhob/(Sw[2]*phi))
Rsp2 = kd[i]*(rhob/(Sw[1]*phi))
Rsp3 = kd[i]*(rhob/(Sw[2]*phi))

ppt_pfos = np.multiply(C_dat_loop,(1000/pfos_mw))
ppt_star_pfos = conc_star(ppt_pfos, ppt_pfos, 0)

pfos_se_l, pfos_kaw_l = langmuir_kaw(pfos_T_max, pfos_a, ppt_star_pfos[i], ppt_pfos[i])
Raw = 1 + (pfoa_kaw_l*100) * Aia_Sw_func2(90000, 5.08, Sw[2]*phi, 0, 0)
Raw = 150

R1 = 1 + Rsp1 + Raw #solid phase component
# R2 = 1 + Rsp2 + Raw[1]
# R3 = 1 + Rsp3 + Raw[2]

D = 0.01387584 #m2/yr --- for PFOA from 10.1061/(asce)ee.1943-7870.0001585
avg_rain = 0.81 # m/yr
phi = 0.399
swm = 0.5
# calculate pore water velocity
v = avg_rain/(phi*swm) #m/yr

t = np.linspace(0, 80, 200)
C0 = C_dat_loop[i]

#superposition with different values of R across the distance??

#how to work in heterogeneity --> integrate the retardation and Sw across the capillary pressure curve
#get saturations plug into R and model in ADR thru some x -- I guess I just defined what makes the vadose zone some complicated...
#R_vert = np.linspace(1, 150, 100) 
# C_wR1 = ADEwReactions_type1_fun(1, t, v, D, R, 0, 0.0001, C0, 10, 0)
# C_nR1 = ADEwReactions_type1_fun(1, t, v, D, 1, 0, 0.0001, C0, 10, 0)
R1, R2, R3 = [4000, 120, 35]

fig, ax =  plt.subplots(1, 1, figsize=(6, 6), dpi=200)
C_wR1 = ADEwReactions_type3_fun(0.5, t, v, D, R1, 0, 0.0001, C0, 5, 0) #5 year pulse
C_nR3 = ADEwReactions_type3_fun(0.5, t, v, D, 1, 0, 0.0001, C0, 5, 0) #5 year pulse
C_wR2 = ADEwReactions_type3_fun(0.5, t, v, D, R2, 0, 0.0001, C0, 5, 0)
C_wR3 = ADEwReactions_type3_fun(0.5, t, v, D, R3, 0, 0.0001, C0, 5, 0)
#ax.plot(t, C_nR3/C0)
ax.plot(t, C_wR1/C0, linewidth = 2)
ax.plot(t, C_wR2/C0, linewidth = 2)
ax.plot(t, C_wR3/C0, linewidth = 2)
ax.set_title('PFOS BTC, C0 = {}'.format(C_labels[i]))
ax.set_xlabel('Time [yrs]')
ax.set_ylabel('C/C0 [-]')
#ax.grid(axis='y')
#ax.set_xlim(0, 80)

# fig1, ax =  plt.subplots(1, 1, figsize=(6, 6), dpi=200)
# C_wR3 = ADEwReactions_type3_fun(1, t, v, D, R2, 0, 0.0001, C0, 5, 0) #5 year pulse
# ax.plot(t, C_wR3/C0)
# # ax.set_title('PFOS BTC @ R=60, C0 = {}'.format(C_labels[i]))
# # ax.set_xlabel('Time [yrs]')
# # ax.set_ylabel('C/C0 [-]')
# # ax.grid(axis='y')

# fig2, ax =  plt.subplots(1, 1, figsize=(6, 6), dpi=200)
# C_wR3 = ADEwReactions_type3_fun(1, t, v, D, R3, 0, 0.0001, C0, 5, 0) #5 year pulse
# ax.plot(t, C_wR3/C0)
# # ax.set_title('PFOS BTC @ R=10, C0 = {}'.format(C_labels[i]))
# # ax.set_xlabel('Time [yrs]')
# # ax.set_ylabel('C/C0 [-]')
# # ax.grid(axis='y')