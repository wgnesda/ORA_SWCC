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
    Raw_tot = 1 + R_aw
    #solid-phase
    kd_oc = koc*foc
    R_oc = (kd_oc *rhob/(Sw*phi)) #OC component
    R_m = (km *rhob/(Sw*phi)) #mineral component
    R_sp = R_m + R_oc
    Rsp_tot = 1+ R_sp
    R_total = 1 + R_aw + R_sp #combined retardation
    
    return R_total, Raw_tot, Rsp_tot

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
    
    return R_total

#%% Kaw predictions
def conc_star(conc_pfas, anion, cation): 
    I = 0.5*(conc_pfas/1000 + anion/1000 + cation/1000) # - Ionic strength calculations are in molarity
    logy = (0.507*np.sqrt(I))/(1+np.sqrt(I))+0.1*I #activity
    y = 10**(-logy)
    conc_star = np.sqrt((y*(conc_pfas+cation))*(y*conc_pfas)) #C*
    return conc_star

def langmuir_kaw(T_max, a, conc_star, conc):
    surf_ex = (T_max*a*conc_star)/(1 + a*conc_star)
    kaw = surf_ex/conc
    return surf_ex, kaw
#%% - Kaw estimations

conc_range = np.array([0.00001, 0.0001, 0.001, 0.01]) #mg/L -- 10ppt, 100ppt, 1ppb, 10 ppb
conc_text = np.array(['10 ppt', '100 ppt', '1ppb', '10 ppb'])
C = 1 #index concentration of interest
#PFOS==========================================================================================
pfos_mw = 538220 #mg/mol
pfos_conc = np.multiply(conc_range, 1000/pfos_mw) #mol/m3
pfos_star = conc_star(pfos_conc, 0, pfos_conc)
#langmuir Parameters
pfos_T_max = 3.08*10**-6
pfos_a = 6.36
#kaw estimate
pfos_se, pfos_kaw = langmuir_kaw(pfos_T_max, pfos_a, pfos_star[C], pfos_conc[C])
pfos_kaw = 100*pfos_kaw #cm
#==============================================================================================

#PHFxS=========================================================================================
pfhxs_mw = 438200 #mg/mol
pfhxs_conc = np.multiply(conc_range, 1000/pfhxs_mw) #mol/m3
pfhxs_star = conc_star(pfhxs_conc, 0, pfhxs_conc)
#langmuir Parameters
pfhxs_T_max = 5.94*10**-7
pfhxs_a = 6.378
#kaw estimate
pfhxs_se, pfhxs_kaw = langmuir_kaw(pfhxs_T_max, pfhxs_a, pfhxs_star[C], pfhxs_conc[C])
pfhxs_kaw = 100*pfhxs_kaw #cm
#============================================================================================

#PFOA========================================================================================
pfoa_mw = 438200 #mg/mol
pfoa_conc = np.multiply(conc_range, 1000/pfoa_mw) #mol/m3
pfoa_star = conc_star(pfoa_conc, 0, 0)
#langmuir Parameters
pfoa_T_max = 6.92*10**-6
pfoa_a = 1.79
#kaw estimate
pfoa_se, pfoa_kaw = langmuir_kaw(pfoa_T_max, pfoa_a, pfoa_star[C], pfoa_conc[C])
pfoa_kaw = 100*pfoa_kaw #cm
#==========================================================================================

#PFDA===================================================================================
pfda_mw = 514080 #mg/mol
pfda_conc = np.multiply(conc_range, 1000/pfda_mw) #mol/m3
pfda_star = conc_star(pfda_conc, 0, 0)

pfda_T_max = 3.77*10**-5
pfda_a = 1.77

pfda_se, pfda_kaw = langmuir_kaw(pfda_T_max, pfda_a, pfda_star[C], pfda_conc[C])
pfda_kaw = 100*pfda_kaw #cm
#============================================================================================
#Now with salt






#%%-- 1D Estimated Retardatin in vadose zone

#Site Specific Parameters=================================================================================
Sr_ORA = 0.125 #residual
Pc_kpa = np.linspace(0, 25, 100) #approimate capillary pressure range from measured SWCC
density = 1 #desity of water g/cm3
haw = (Pc_kpa / 9.81*density)*100 #height above water table [cm]
a_val = 0.32 #entry pressure of 4 kPa
n_val = 4.0 #pore-size distribution
RH_U = 5.08 #uniformity coefficent
BET = 5 #spefici surface area in m2/g
phi = 0.42 #porosity
rhob = 1.53 #averaged dry bulk density?? <--- use this
s = (BET*100**2)*rhob #specific surface area in 1/cm
OC_dis = (1*10**-6)*np.exp(0.3621*(Pc_kpa)) #distribution of organic carbon

#PFAS solid-phase sorption 
#PFOS=================================================================================================
pfos_koc = (10**3.41) #averaged lit values for     alt literature values for PFOS (2.8)
pfos_km = 0.03 #calculated Kd from LCMS - assumed completely associated with mineral phase adsorption

#PFHxS==============================================================================================
pfhxs_koc = (10**2.71) #averaged lit values for

#PFOA=============================================================================================
pfoa_koc = (10**2.62) #averaged lit values for

#PFDA=============================================================================================
pfda_koc = (10**3.65) #averaged lit values for

#================================================================================================
#%%- 1d profile

#PFOS=========================================================================================================
R_pfos, Raw_pfos, Rsp_pfos = vadoseRetaration_profile_func(a_val, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfos_kaw, pfos_koc, OC_dis, pfos_km)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_pfos = 1/haw[-1] * np.trapz(R_pfos, haw) 
R_bar_sp_pfos = 1/haw[-1] * np.trapz(Rsp_pfos, haw) 
R_bar_aw_pfos = 1/haw[-1] * np.trapz(Raw_pfos, haw) 

#PFHxS========================================================================================================
R_pfhxs, Raw_pfhxs, Rsp_pfhxs = vadoseRetaration_profile_func(a_val, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfhxs_kaw, pfhxs_koc, OC_dis, 0)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_pfhxs = 1/haw[-1] * np.trapz(R_pfhxs, haw) 
R_bar_sp_pfhxs = 1/haw[-1] * np.trapz(Rsp_pfhxs, haw) 
R_bar_aw_pfhxs = 1/haw[-1] * np.trapz(Raw_pfhxs, haw) 

#PFOA========================================================================================================
R_pfoa, Raw_pfoa, Rsp_pfoa = vadoseRetaration_profile_func(a_val, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfoa_kaw, pfoa_koc, OC_dis, 0)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_pfoa = 1/haw[-1] * np.trapz(R_pfoa, haw) 
R_bar_sp_pfoa = 1/haw[-1] * np.trapz(Rsp_pfoa, haw) 
R_bar_aw_pfoa = 1/haw[-1] * np.trapz(Raw_pfoa, haw) 



#PFDA========================================================================================================
R_pfda, Raw_pfda, Rsp_pfda = vadoseRetaration_profile_func(a_val, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfda_kaw, pfda_koc, OC_dis, 0)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_pfda = 1/haw[-1] * np.trapz(R_pfda, haw) 
R_bar_sp_pfda = 1/haw[-1] * np.trapz(Rsp_pfda, haw) 
R_bar_aw_pfda = 1/haw[-1] * np.trapz(Raw_pfda, haw) 


#Plotting====================================================================================================
pfoacolors = plt.cm.Reds(np.linspace(0,1,5))
pfoscolors = plt.cm.Blues(np.linspace(0,1,5))
pfhxscolors = plt.cm.Greens(np.linspace(0,1,5))
pfdacolors = plt.cm.Purples(np.linspace(0,1,5))

fig_R, (ax) =  plt.subplots(1, 1, figsize=(8, 6), dpi=200)

rpfos, = ax.plot(R_pfos, haw, label = '$R_{pfos}$', linewidth = 3, c=pfoscolors[2])
rpfhxs, = ax.plot(R_pfhxs, haw, label = '$R_{pfhxs}$', linewidth = 3, c=pfhxscolors[2])
rpfoa, = ax.plot(R_pfoa, haw, label = '$R_{pfoa}$', linewidth = 3, c=pfoacolors[2])
rpfda, = ax.plot(R_pfda, haw, label = '$R_{pfda}$', linewidth = 3, c=pfdacolors[2])


#depth average values
ax.axvline(R_bar_pfos, ls ='--', c=pfoscolors[2], label = round(R_bar_pfos))
ax.axvline(R_bar_pfhxs, ls ='--', c=pfhxscolors[2], label = round(R_bar_pfhxs))
ax.axvline(R_bar_pfoa, ls ='--', c=pfoacolors[2], label = round(R_bar_pfoa))
ax.axvline(R_bar_pfda, ls ='--', c=pfdacolors[2], label = round(R_bar_pfda))

ax.set_title('ORA site-specific retardation\n $S_r = {}$'.format(Sr_ORA), pad = 15)
ax.set_xlabel('Retardation Factor [-]')
ax.set_ylabel('Heigth above water table [cm]')
ax.legend(handles=[rpfos, rpfhxs, rpfoa, rpfda])
ax.grid()
#ax.set_yticklabels([])
#ax.yaxis.tick_right()
#ax.yaxis.set_label_position("right")
#%%- comparing against different residuals
fig_R_test, ax1 =  plt.subplots(1, 1, figsize=(6, 5), dpi=200)
Sr_lin = [0.125, 0.25, 0.50, 1]
for i in Sr_lin:
    R_tot_pfos, R_aw_pfos, R_sp_pfos = vadoseRetaration_profile_func(a_val, n_val, i, Pc_kpa, s, RH_U, phi, rhob, pfos_kaw, pfos_koc, OC_dis, pfos_km)
    plt.plot(R_tot_pfos, haw, label = i, linewidth = 2)

ax1.set_title('With increasing water saturation', pad = 15)
ax1.set_xlabel('Retardation Factor [-]')
ax1.set_ylabel('Heigth above water table [cm]')
ax1.grid()    
labelLines(ax1.get_lines(),fontsize=12, align=True)
#%% - Surface plots

#ranging across values of av and pc
av = np.array(np.linspace(0, 1, 100))
Sr = 0.125 #causes some issues...
#call surface retardation function
#PFOS=============================================================================================================
R_out_pfos = vadoseRetaration_2Dprofile_func(av, n_val, Sr, Pc_kpa, s, RH_U, phi, rhob, 
                                                                    pfos_kaw, pfos_koc, OC_dis, pfos_km)

#PFHxS=============================================================================================================
R_out_pfhxs = vadoseRetaration_2Dprofile_func(av, n_val, Sr, Pc_kpa, s, RH_U, phi, rhob, 
                                                                    pfhxs_kaw, pfhxs_koc, OC_dis, 0)

#PFOA=============================================================================================================
R_out_pfoa = vadoseRetaration_2Dprofile_func(av, n_val, Sr, Pc_kpa, s, RH_U, phi, rhob, 
                                                                    pfoa_kaw, pfoa_koc, OC_dis, 0)

#PFOA=============================================================================================================
R_out_pfda = vadoseRetaration_2Dprofile_func(av, n_val, Sr, Pc_kpa, s, RH_U, phi, rhob, 
                                                                    pfda_kaw, pfda_koc, OC_dis, 0)

#1D range of Sw values 
for i in av:
    for j in Pc_kpa:
        Sw = Sw_func(av, 4, Pc_kpa, Sr)
  
# fig, (ax, ax2) =  plt.subplots(1, 2, figsize=(14, 6), dpi=200)

#for looking at the relative contribution of each retardation component, note must add to output of 2D plot func
# #Air-water specifc
# im = ax.contourf(Sw, haw, R_aw_out, 5, cmap = 'Reds')
# cbar = fig.colorbar(im, ax=ax, label='Retardation Factor [-]')
# #figuring out how to do the solid-phase
# lognorm = colors.LogNorm(vmin=R_sp_out.min(), vmax=R_sp_out.max())
# lev = [1, 10, 100, 200, 300, 500, 700, 800, 1000]
# im2 = ax2.contourf(Sw, haw, R_sp_out, 5, cmap = 'Blues')
# cbar = fig.colorbar(im2, ax=ax2, label='Retardation Factor [-]')


#Separate plot for total retardation
fig_2D, ([ax, ax1],
       [ax2, ax3]) =  plt.subplots(2, 2, figsize=(18, 16), dpi=200)

ft = 18
pds = 15

im = ax.contourf(Sw, haw, np.log10(R_out_pfos), 5, cmap = 'Blues')
cbar = fig_2D.colorbar(im, ax=ax, label='$Log_{10}$Retardation Factor [-]')
ax.set_title('PFOS retardation, $C_0 = {}$'.format(conc_text[C]), fontsize = ft, pad=pds)
ax.set_ylabel('Height above water table [cm]', fontsize = ft)

im1 = ax1.contourf(Sw, haw, np.log10(R_out_pfhxs), 5, cmap = 'Greens')
cbar = fig_2D.colorbar(im1, ax=ax1, label='$Log_{10}$Retardation Factor [-]')
ax1.set_title('PFHxS retardation, $C_0 = {}$'.format(conc_text[C]), fontsize = ft, pad=pds)


im2 = ax2.contourf(Sw, haw, np.log10(R_out_pfoa), 5, cmap = 'Reds')
cbar = fig_2D.colorbar(im2, ax=ax2, label='$Log_{10}$Retardation Factor [-]')
ax2.set_title('PFOA retardation, $C_0 = {}$'.format(conc_text[C]), fontsize = ft, pad=pds)
ax2.set_ylabel('Height above water table [cm]', fontsize = ft)
ax2.set_xlabel('water saturation [-]', fontsize = ft)

im3 = ax3.contourf(Sw, haw, np.log10(R_out_pfda), 5, cmap = 'Purples')
cbar = fig_2D.colorbar(im3, ax=ax3, label='$Log_{10}$Retardation Factor [-]')
ax3.set_title('PFDA retardation, $C_0 = {}$'.format(conc_text[C]), fontsize = ft, pad=pds)
ax3.set_xlabel('water saturation [-]', fontsize = ft)

axes = [ax, ax1, ax2, ax3]
for i in axes:
    i.plot(Sw_func(a_val, n_val, Pc_kpa, 0.125), haw, "--", linewidth = 3, color = 'black', label = "Measured SWCC")
    i.plot(Sw_func(0.2, n_val, Pc_kpa, 0.39), haw, "--", linewidth = 3, color = 'grey', label = "Silty-sand SWCC")

fig_2D.tight_layout()

#%% WORKING -- Eustis and Vinton soil comparisions to Brusseau work...

#********if you run this line must run from beginning -- variables will overprint!!!!!!!!

# #for 1D plots
def vadoseRetaration_func(av, nv, Sr, Pc, s, U, phi, rhob, kaw, kd):
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
    Sw = Sr+(1-Sr)*(1+(av*Pc)**nv)**-mv #change with thermo appraoch ?
    
    #Awi calulations
    a = 14.3*np.log(U)+3.72
    m = -0.098*U + 1.53 # currently only valid for U<=3.5
    n = 1/(2-m)
    Aia = s*(1+(a*(Sr+(1-Sr)*(Sw))**n))**-m  
    R_aw = kaw * Aia/(Sw*phi)
    #solid-phase
    R_sp = (kd *rhob/(Sw*phi)) #mineral component
    R_total = 1 + R_aw + R_sp #combined retardation
    
    return R_total

#vinton soil
Sr_v = 0.07
Pc_cmH2O = np.linspace(0, 200, 100)

#values from 10.1061/(ASCE)HE.1943-5584.0000515.AIR-WATER
a_val_v = 0.025 #entry pressure of 4 kPa
n_val_v = 3.43
V_U = 2.4 #need to use correction for m -- all good
ssa_v = 3.54 #m2/g
phi_v = 0.46
rhob_v = 1.46
s_v = (ssa_v*100**2)*rhob

#values from 
kd_v = 0.5 #PFOS 10.1021/acs.est.9b02343
kaw_v = 0.0027 #PFOS 10.1021/acs.est.9b02343

vinton_R = vadoseRetaration_func(a_val_v, n_val_v, Sr_v, Pc_cmH2O, s_v, V_U, phi_v, rhob_v, kaw_v, kd_v)
vinton_R_bar = 1/Pc_cmH2O[-1] * np.trapz(vinton_R, Pc_cmH2O)

fig_vinton, ax =  plt.subplots(1, 1, figsize=(8, 6), dpi=200)
#our data
ax.plot(R_pfos, haw, label= '$R_{ora}$', linewidth = 3, c = pfoscolors[2] )
ax.axvline(R_bar_pfos, ls ='--', label = '$\overline{R}_{ora}$', c = pfoscolors[2] )
#vinton
ax.plot(vinton_R, Pc_cmH2O, lw=2, c = 'purple', label = '$R_{vinton}$')
ax.axvline(vinton_R_bar, ls ='--', c = 'purple', label = '$\overline{R}_{vinton}$')

ax.set_xlabel('PFOS: Retardation Factor [-]')
ax.set_ylabel('Pc [cm H2O]')
ax.set_title('Vinton vs ORA soil retardation profile')
ax.legend()
#variation due to residual...