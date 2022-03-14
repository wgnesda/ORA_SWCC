# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:48:56 2022
Valocchi - adsorption from moments
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
# from vadose_zone_adsorption_functions import *
# from SFT_measurements import *
# from ADRs import *

#%matplotlib inline
#%matplotlib qt
#%%
def Sw_func(a, n, Pc, Sr):
    m = 1 - (1/n)
    Sw = Sr+(1-Sr)*(1+(a*Pc)**n)**-m
    return Sw

def j_leverett_func(Sw, n, a, Sr, k, e, sft, p):
    S_eff = (Sw- Sr)/(1-Sr) #ignoring non-wetting
    m = 1 - (1/n)
    Pc = ((p* 9.81)/a)*((S_eff)**(-1/m)-1)**(1/n)
    Js = (Pc/sft)*(k/e)**(1/2)
    return Js

def Aia_Sw_func2(s, U, S, Sm, C):
    a = 14.3*np.log(U)+3.72
    if np.any(U < 3.5):
        m = 0.098*U+1.53
    else:
        m = 1.2
    n = 1/(2-m)
    Aia = s*(1+(a*(S-Sm))**n)**-m - C
    return Aia

#for 1D plots
def vadoseRetaration_profile_func(av, nv, Sr, Pc, s, U, phi, rhob, kaw, koc, foc, km):
    #units
    #Pc in kPa
    #a in 1/kPa
    #s in 1/cm
    #rhob in kg/L or g/cm3
    #kaw in cm
    #koc in L/kg
    #km in L/kg
    
    #saturation from Van Genucten equation
    mv = 1 - (1/nv)
    Sw = Sr+(1-Sr)*(1+(av*Pc)**nv)**-mv #change with thermo appraoch 
    
    #Awi calulations
    a = 14.3*np.log(U)+3.72
    m = 1.2 # currently only valid for U>3.5
    n = 1/(2-m)
    Aia = s*(1+(a*(Sr+(1-Sr)*(Sw))**n))**-m  
    R_aw = kaw * Aia/(Sw*phi)
    R_aw_tot = 1 + R_aw
    #solid-phase
    kd_oc = koc*foc
    R_oc = (kd_oc *rhob/(Sw*phi)) #OC component
    R_m = (km *rhob/(Sw*phi)) #mineral component
    R_sp = R_m + R_oc
    R_sp_tot = 1+ R_sp
    R_total = 1 + R_aw + R_sp #combined retardation
    
    return R_total, R_aw_tot, R_sp_tot, Sw, Aia

#For 2D plots - calculate Saturation surface from av and Pc range --> 
#use to get volumetric water content surface and Awi surface
def vadoseRetaration_2Dprofile_func(av, nv, Sr, Pc, s, U, phi, rhob, kaw, koc, foc, km):
    #saturation profile across values of av
    Sw = np.zeros([len(Pc_kpa), len(av)])
    for i in range(0, len(Pc_kpa)):
       for j in range(0, len(av)):
            mv = 1 - (1/nv)
            Sw[i,j] = Sr+(1-Sr)*(1+(av[j]*Pc[i])**nv)**-mv #change with thermo appraoch?
    #Awi parameters
    a = 14.3*np.log(U)+3.72
    m = 1.2# currently only valid for U>3.5
    n = 1/(2-m)
    Aia = s*(1+(a*(Sr+(1-Sr)*(Sw))**n))**-m
    
    #Note: KAW and KD are derived from isotherms = specifc for a concentration (build in that part?)
    R_aw = kaw * Aia/(Sw*phi)
    R_aw_tot = 1 + R_aw #total retarration associated with air-water interface
    kd_oc = koc*foc
    R_sp = (km *rhob/(Sw*phi)) + (kd_oc *rhob/(Sw*phi))
    R_sp_tot = 1 + R_sp #total retardation associated with solid-phase sorption
    R_total = 1 + R_aw + R_sp #combine retardation effect.
    
    return R_total, Sw, R_aw_tot, R_sp_tot

#%%
Sr = 0.125 #residual
Pc_kpa = np.linspace(0, 30, 100) #approimate capillary pressure range from measured SWCC
density = 1 #desity of water g/cm3
haw = (Pc_kpa / 9.81*density)*100 
a_val = 0.32 #entry pressure of 4 kPa
n_val = 4.0 #pore-size distribution
RH_U = 5.08 #uniformity coefficent
BET = 5 #spefici surface area in m2/g
phi = 0.42 #porosity
rhob = 1.53 #averaged dry bulk density?? <--- use this
#sg = 2.62 #from specifc gravity tests?
kaw = 0.00196 #PFOS kaw at 70ppt
#kaw = 0.0196 #PFOS in 0.1 M nacl
s = (BET*100**2)*rhob #specific surface area in 1/cm

OC_dis = (1*10**-6)*np.exp(0.3621*(Pc_kpa)) #distribution of organic carbon
koc = (10**2.8) #literature values for PFOS
km = 0.03 #calculated Kd from LCMS - assumed completely associated with mineral phase adsorption

R_total, R_aw_tot, R_sp_tot, Sw_ORA, Aia_ORA = vadoseRetaration_profile_func(a_val, n_val, Sr, Pc_kpa, s, RH_U, 
                                                                     phi, rhob, kaw, koc, OC_dis, km)
    
#fig_R, (ax1, ax) =  plt.subplots(1, 2, figsize=(15, 6), dpi=200)
#fig_R, (ax2, ax1, ax) = plt.subplots(1, 3, figsize=(12,6), gridspec_kw={'width_ratios': [1, 1, 3]}, dpi=200)
fig_R, (ax) =  plt.subplots(1, 1, figsize=(8, 6), dpi=200)

#calculatiing depth averaged R from Valocchi paper
R_bar = 1/haw[-1] * np.trapz(R_total, haw) 
R_bar_sum = 1/haw[-1] * np.cumsum(np.trapz(R_total, haw))

R_bar_sp = 1/haw[-1] * np.trapz(R_sp_tot, haw) 
R_bar_aw = 1/haw[-1] * np.trapz(R_aw_tot, haw) 

#1D profile
ax.plot(R_aw_tot, haw, label = '$R_{air-water}$', linewidth = 3, c='indianred')
ax.plot(R_sp_tot, haw, label = '$R_{solid-phase}$', linewidth = 3, c='cornflowerblue')
ax.plot(R_total, haw, label= '$R_{combined}$', linewidth = 3, c = 'mediumpurple')

#depth average values
ax.axvline(R_bar, ls ='--', c = 'mediumpurple', label = '$depth_{ave}$')
ax.axvline(R_bar_sp, ls ='--', c = 'cornflowerblue')
ax.axvline(R_bar_aw, ls ='--', c = 'indianred')

ax.set_title('ORA site-specific retardation\n $S_r = {}$'.format(Sr), pad = 15)
ax.set_xlabel('Retardation Factor [-]')
ax.set_ylabel('Heigth above water table [cm]')
ax.legend()
ax.grid()
#ax.set_yticklabels([])
#ax.yaxis.tick_right()
#ax.yaxis.set_label_position("right")
#%%- comparing against different residuals
fig_R_test, ax1 =  plt.subplots(1, 1, figsize=(6, 5), dpi=200)
Sr_lin = [0.125, 0.25, 0.50, 1]
for i in Sr_lin:
    R_total, R_aw, R_sp, Sw_ORA, Aia_ORA = vadoseRetaration_profile_func(a_val, n_val, i, Pc_kpa, s, RH_U, phi, rhob, kaw, koc, OC_dis, km)
    plt.plot(R_total, haw, label = i, linewidth = 2)

ax1.set_title('With increasing water saturation', pad = 15)
ax1.set_xlabel('Retardation Factor [-]')
ax1.set_ylabel('Heigth above water table [cm]')
ax1.grid()    
labelLines(ax1.get_lines(),fontsize=12, align=True)
#%% - Surface plot

#ranging across values of av and pc
av = np.array(np.linspace(0, 1, 100))

#call surface retardation function
R_out, Sw_out, R_aw_out, R_sp_out = vadoseRetaration_2Dprofile_func(av, n_val, 0.125, Pc_kpa, s, RH_U, phi, rhob, kaw, koc, OC_dis, km)

#1D range of Sw values 
for i in av:
    for j in Pc_kpa:
        Sw = Sw_func(av, 4, Pc_kpa, 0.0)
  
fig, (ax, ax2) =  plt.subplots(1, 2, figsize=(14, 6), dpi=200)
#Air-water specifc
im = ax.contourf(Sw, haw, R_aw_out, 5, cmap = 'Reds')
cbar = fig.colorbar(im, ax=ax, label='Retardation Factor [-]')
#figuring out how to do the solid-phase
lognorm = colors.LogNorm(vmin=R_sp_out.min(), vmax=R_sp_out.max())
lev = [1, 10, 100, 200, 300, 500, 700, 800, 1000]
im2 = ax2.contourf(Sw, haw, R_sp_out, 5, cmap = 'Blues')
cbar = fig.colorbar(im2, ax=ax2, label='Retardation Factor [-]')

#Separate plot for total retardation
fig1, ax3 =  plt.subplots(1, 1, figsize=(8, 6), dpi=200)
im3 = ax3.contourf(Sw, haw, R_out, 5, cmap = 'Purples')
cbar = fig1.colorbar(im3, ax=ax3, label='Retardation Factor [-]')

#%% WORKING -- Eustis and Vinton soil comparisions to Brusseau work...

Sr = 0.125
Pc_kpa = np.linspace(0, 20, 100)
density = 1 #desity of water
haw = (Pc_kpa / 9.81*density)*100 
a_val = 0.4 #entry pressure of 4 kPa
n_val = 4.0
V_U = 2.4
s = 51330
phi = 0.42
rhob = 1.8
kaw = 0.00196 #PFOS kaw at 70ppt

