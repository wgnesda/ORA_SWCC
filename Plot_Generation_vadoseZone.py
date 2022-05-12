# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:14:17 2022
All figures in paper
@author: willg
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for analytical solutions
import scipy
from scipy import optimize
from scipy.special import erfc as erfc
import math
from math import pi
from matplotlib import rc
import matplotlib.ticker as mtick
from labellines import labelLine, labelLines
rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams['font.size'] = 16
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#%% - All functions
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

def Sw_func(a, n, Pc, Sr):
    m = 1 - (1/n)
    Sw = Sr+(1-Sr)*(1+(a*Pc)**n)**-m
    return Sw

def Pc_func(a, m, Sw, Sr):
    Seff = (Sw - Sr)/(1 - Sr)
    #n = 1 / (1-m)
    Pc = (1/a)*(Seff**(-1/m)-1)**(1-m)
    return Pc

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
    #av in 1/kPa
    #s in 1/cm
    #rhob in kg/L or g/cm3
    #kaw in cm
    #koc in L/kg
    #km in L/kg
    
    #saturation from Van Genucten (1980)
    #from 10.2136/sssaj1980.03615995004400050002x
    mv = 1 - (1/nv)   
    Sw = Sr+(1-Sr)*(1+(av*Pc)**nv)**-mv #change with thermo appraoch 
    
    #Awi calulations
    a = 14.3*np.log(U)+3.72
    m = 1.2 # currently only valid for U>3.5
    # m = -0.098*U+1.53 #for U<3.5
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
def vadoseRetaration_2Dprofile_func(av, nv, Sr, Pc_kpa, s, U, phi, rhob, kaw, koc, foc, km):
    #saturation profile across values of av
    Sw = np.zeros([len(Pc_kpa), len(av)])
    for i in range(0, len(Pc_kpa)):
       for j in range(0, len(av)):
            mv = 1 - (1/nv)
            Sw[i,j] = Sr+(1-Sr)*(1+(av[j]*Pc_kpa[i])**nv)**-mv #change with thermo appraoch?
    #Awi parameters
    a = 14.3*np.log(U)+3.72
    m = 1.2# currently only valid for U>3.5
    # m = -0.098*U+1.53 #for U<3.5
    n = 1/(2-m)
    Aia = s*(1+(a*(Sr+(1-Sr)*(Sw))**n))**-m
    
    #Note: KAW and KD are derived from isotherms = specifc for a concentration (build in that part?)
    R_aw = kaw * Aia/(Sw*phi)
    R_aw_tot = 1 + R_aw #total retarration associated with air-water interface
    kd_oc = koc*foc
    R_sp = (km *rhob/(Sw*phi)) + (kd_oc *rhob/(Sw*phi))
    R_sp_tot = 1 + R_sp #total retardation associated with solid-phase sorption
    R_total = 1 + R_aw + R_sp #combine retardation effect.
    
    return R_total, R_aw_tot, R_sp_tot

def vadoseRetaration_SW_func(Sw, Sr, Pc, s, U, phi, rhob, kaw, koc, foc, km):
   
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
#PFOS in 0.1M NaCl==========================================================================
pfos_nacl_star = conc_star(pfos_conc, 100, 100)**2 #note, square of mean ionic activiy

pfos_nacl_T_max = 7.41*10**-6
pfos_nacl_a = 0.522

pfos_nacl_se, pfos_nacl_kaw = langmuir_kaw(pfos_nacl_T_max, pfos_nacl_a, pfos_nacl_star[C], pfos_conc[C])
pfos_nacl_kaw = 100*pfos_nacl_kaw #cm
#============================================================================================

#PFHxS in 0.1M NaCl==========================================================================
pfhxs_nacl_star = conc_star(pfhxs_conc, 100, 100)**2 #note, square of mean ionic activiy

pfhxs_nacl_T_max = 1.75*10**-6
pfhxs_nacl_a = 1.307

pfhxs_nacl_se, pfhxs_nacl_kaw = langmuir_kaw(pfhxs_nacl_T_max, pfhxs_nacl_a, pfhxs_nacl_star[C], pfhxs_conc[C])
pfhxs_nacl_kaw = 100*pfhxs_nacl_kaw #cm
#============================================================================================

#PFOA in 0.1M NaCl=============Using Brusseau================================================
pfoa_nacl_star = conc_star(pfoa_conc, 100, 100)**2 #note, square of mean ionic activiy

pfoa_nacl_T_max = 3.08*10**-6
pfoa_nacl_a = 0.32

pfoa_nacl_se, pfoa_nacl_kaw = langmuir_kaw(pfoa_nacl_T_max, pfoa_nacl_a, pfoa_nacl_star[C], pfoa_conc[C])
pfoa_nacl_kaw = 100*pfoa_nacl_kaw #cm

#============================================================================================

#PFDA in 0.1M NaCl==========================================================================
pfda_nacl_star = conc_star(pfda_conc, 100, 100)**2 #note, square of mean ionic activiy

pfda_nacl_T_max = 7.84*10**-6
pfda_nacl_a = 2.41

pfda_nacl_se, pfda_nacl_kaw = langmuir_kaw(pfda_nacl_T_max, pfda_nacl_a, pfda_nacl_star[C], pfda_conc[C])
pfda_nacl_kaw = 100*pfda_nacl_kaw #cm
#============================================================================================

#-- 1D Estimated Retardatin in vadose zone

#Site Specific Parameters=================================================================================
Sr_ORA = 0.05 #residual
Pc_kpa = np.linspace(0, 30, 100) #approimate capillary pressure range from measured SWCC
density = 1 #desity of water g/cm3
haw = (Pc_kpa / 9.81*density)*100 #height above water table [cm]

n_val = 4.0 #pore-size distribution
RH_U = 4.99#5.08 #uniformity coefficent
BET = 5 #spefici surface area in m2/g
phi = 0.42 #porosity
rhob = 1.53 #averaged dry bulk density?? <--- use this
s = (BET*100**2)*rhob #specific surface area in 1/cm
OC_dis = (1*10**-6)*np.exp(0.3621*(Pc_kpa)) #distribution of organic carbon

#PFAS solid-phase sorption 
#PFOS=================================================================================================
pfos_koc = (10**3.41) #averaged lit values for     alt literature values for PFOS (2.8)
pfos_km = 0.1 #calculated Kd from LCMS - assumed completely associated with mineral phase adsorption
pfos_km_nacl = 0.34
#PFHxS==============================================================================================
pfhxs_koc = (10**2.71) #averaged lit values for

#PFOA=============================================================================================
pfoa_koc = (10**2.62) #averaged lit values for

#PFDA=============================================================================================
pfda_koc = (10**3.65) #averaged lit values for

#================================================================================================

a_sand = 0.25 #entry pressure of 4 kPa -GRAVELY SAND #formerly a_sand
a_fsand = 0.08 #sactually the fine-sand
a_silt = 0.055 #actually the silty-sand

#%%- 1d profiles
#PFOS=========================================================================================================
#gravely-sand
R_pfos, Raw_pfos, Rsp_pfos = vadoseRetaration_profile_func(a_sand, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfos_kaw, pfos_koc, OC_dis, pfos_km)
#fine-sand
R_pfos_fine = vadoseRetaration_profile_func(a_fsand, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfos_kaw, pfos_koc, OC_dis, pfos_km)
#silty sand
R_pfos_silt = vadoseRetaration_profile_func(a_silt, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfos_kaw, pfos_koc, OC_dis, pfos_km)
    

#calculatiing depth averaged R from Valocchi paper
R_bar_pfos      = 1/haw[-1] * np.trapz(R_pfos, haw) 
R_bar_pfos_fine = 1/haw[-1] * np.trapz(R_pfos_fine[0], haw) 
R_bar_pfos_silt = 1/haw[-1] * np.trapz(R_pfos_silt[0], haw) 

R_bar_sp_pfos = 1/haw[-1] * np.trapz(Rsp_pfos, haw) 
R_bar_pfos_sp_silt = 1/haw[-1] * np.trapz(R_pfos_silt[0], haw) 

R_bar_aw_pfos = 1/haw[-1] * np.trapz(Raw_pfos, haw) 
R_bar_pfos_aw_silt = 1/haw[-1] * np.trapz(R_pfos_silt[1], haw) 

#PFHxS========================================================================================================
R_pfhxs, Raw_pfhxs, Rsp_pfhxs = vadoseRetaration_profile_func(a_sand, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfhxs_kaw, pfhxs_koc, OC_dis, 0)

R_pfhxs_silt = vadoseRetaration_profile_func(a_silt, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfhxs_kaw, pfhxs_koc, OC_dis, 0)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_pfhxs = 1/haw[-1] * np.trapz(R_pfhxs, haw) 
R_bar_pfhxs_silt = 1/haw[-1] * np.trapz(R_pfhxs_silt[0], haw) 

R_bar_sp_pfhxs = 1/haw[-1] * np.trapz(Rsp_pfhxs, haw) 
R_bar_pfhxs_sp_silt = 1/haw[-1] * np.trapz(R_pfhxs_silt[2], haw) 

R_bar_aw_pfhxs = 1/haw[-1] * np.trapz(Raw_pfhxs, haw) 
R_bar_pfhxs_aw_silt = 1/haw[-1] * np.trapz(R_pfhxs_silt[1], haw) 

#PFOA========================================================================================================
R_pfoa, Raw_pfoa, Rsp_pfoa = vadoseRetaration_profile_func(a_sand, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfoa_kaw, pfoa_koc, OC_dis, 0)

R_pfoa_silt = vadoseRetaration_profile_func(a_silt, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfoa_kaw, pfoa_koc, OC_dis, 0)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_pfoa = 1/haw[-1] * np.trapz(R_pfoa, haw) 
R_bar_pfoa_silt = 1/haw[-1] * np.trapz(R_pfoa_silt[0], haw) 

R_bar_sp_pfoa = 1/haw[-1] * np.trapz(Rsp_pfoa, haw) 
R_bar_pfoa_sp_silt = 1/haw[-1] * np.trapz(R_pfoa_silt[2], haw) 

R_bar_aw_pfoa = 1/haw[-1] * np.trapz(Raw_pfoa, haw) 
R_bar_pfoa_aw_silt = 1/haw[-1] * np.trapz(R_pfoa_silt[1], haw) 


#PFDA========================================================================================================
R_pfda, Raw_pfda, Rsp_pfda = vadoseRetaration_profile_func(a_sand, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfda_kaw, pfda_koc, OC_dis, 0)

R_pfda_silt = vadoseRetaration_profile_func(a_silt, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfda_kaw, pfda_koc, OC_dis, 0)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_pfda = 1/haw[-1] * np.trapz(R_pfda, haw) 
R_bar_pfda_silt = 1/haw[-1] * np.trapz(R_pfda_silt[0], haw) 

R_bar_sp_pfda = 1/haw[-1] * np.trapz(Rsp_pfda, haw) 
R_bar_pfda_sp_silt = 1/haw[-1] * np.trapz(R_pfda_silt[2], haw) 

R_bar_aw_pfda = 1/haw[-1] * np.trapz(Raw_pfda, haw)
R_bar_pfda_aw_silt = 1/haw[-1] * np.trapz(R_pfda_silt[1], haw) 
 
#============================================================================================================

#%%Salty PFAS
#PFOS in 0.1 M NaCl
R_nacl_pfos, Raw_nacl_pfos, Rsp_nacl_pfos = vadoseRetaration_profile_func(a_sand, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfos_nacl_kaw, pfos_koc, OC_dis, pfos_km_nacl)

R_nacl_pfos_silt = vadoseRetaration_profile_func(a_silt, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfos_nacl_kaw, pfos_koc, OC_dis, pfos_km_nacl)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_nacl_pfos = 1/haw[-1] * np.trapz(R_nacl_pfos, haw) 
R_bar_nacl_pfos_silt = 1/haw[-1] * np.trapz(R_nacl_pfos_silt[0], haw)

R_bar_sp_nacl_pfos = 1/haw[-1] * np.trapz(Rsp_nacl_pfos, haw) 
R_bar_sp_nacl_pfos_silt = 1/haw[-1] * np.trapz(R_nacl_pfos_silt[2], haw)

R_bar_aw_nacl_pfos = 1/haw[-1] * np.trapz(Raw_nacl_pfos, haw)
R_bar_aw_nacl_pfos_silt = 1/haw[-1] * np.trapz(R_nacl_pfos_silt[1], haw)

#PFHxS========================================================================================================
R_nacl_pfhxs, Raw_nacl_pfhxs, Rsp_nacl_pfhxs = vadoseRetaration_profile_func(a_sand, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfhxs_nacl_kaw, pfhxs_koc, OC_dis, 0)

R_nacl_pfhxs_silt = vadoseRetaration_profile_func(a_silt, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfhxs_nacl_kaw, pfhxs_koc, OC_dis, 0)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_nacl_pfhxs = 1/haw[-1] * np.trapz(R_nacl_pfhxs, haw) 
R_bar_nacl_pfhxs_silt = 1/haw[-1] * np.trapz(R_nacl_pfhxs_silt[0], haw)

R_bar_sp_nacl_pfhxs = 1/haw[-1] * np.trapz(Rsp_nacl_pfhxs, haw) 
R_bar_sp_nacl_pfhxs_silt = 1/haw[-1] * np.trapz(R_nacl_pfhxs_silt[2], haw)

R_bar_aw_nacl_pfhxs = 1/haw[-1] * np.trapz(Raw_nacl_pfhxs, haw) 
R_bar_aw_nacl_pfhxs_silt = 1/haw[-1] * np.trapz(R_nacl_pfhxs_silt[1], haw)

#PFOA========================================================================================================
R_nacl_pfoa, Raw_nacl_pfoa, Rsp_nacl_pfoa = vadoseRetaration_profile_func(a_sand, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfoa_nacl_kaw, pfoa_koc, OC_dis, 0)

R_nacl_pfoa_silt = vadoseRetaration_profile_func(a_silt, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfoa_nacl_kaw, pfoa_koc, OC_dis, 0)

R_bar_nacl_pfoa = 1/haw[-1] * np.trapz(R_nacl_pfoa, haw)
R_bar_nacl_pfoa_silt = 1/haw[-1] * np.trapz(R_nacl_pfoa_silt[0], haw)
 
R_bar_sp_nacl_pfoa = 1/haw[-1] * np.trapz(Rsp_nacl_pfoa, haw) 
R_bar_sp_nacl_pfoa_silt = 1/haw[-1] * np.trapz(R_nacl_pfoa_silt[2], haw)

R_bar_aw_nacl_pfoa = 1/haw[-1] * np.trapz(Raw_nacl_pfoa, haw) 
R_bar_aw_nacl_pfoa_silt = 1/haw[-1] * np.trapz(R_nacl_pfoa_silt[1], haw)

#PFDA========================================================================================================
R_nacl_pfda, Raw_nacl_pfda, Rsp_nacl_pfda = vadoseRetaration_profile_func(a_sand, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfda_nacl_kaw, pfda_koc, OC_dis, 0)

R_nacl_pfda_silt = vadoseRetaration_profile_func(a_silt, n_val, Sr_ORA, Pc_kpa, s, RH_U, 
                                                           phi, rhob, pfda_nacl_kaw, pfda_koc, OC_dis, 0)
    
#calculatiing depth averaged R from Valocchi paper
R_bar_nacl_pfda = 1/haw[-1] * np.trapz(R_nacl_pfda, haw) 
R_bar_nacl_pfda_silt = 1/haw[-1] * np.trapz(R_nacl_pfda_silt[0], haw)

R_bar_sp_nacl_pfda = 1/haw[-1] * np.trapz(Rsp_nacl_pfda, haw) 
R_bar_sp_nacl_pfda_silt = 1/haw[-1] * np.trapz(R_nacl_pfda_silt[2], haw)

R_bar_aw_nacl_pfda = 1/haw[-1] * np.trapz(Raw_nacl_pfda, haw) 
R_bar_aw_nacl_pfda_silt = 1/haw[-1] * np.trapz(R_nacl_pfda_silt[1], haw)

#==============================================================================================================

Sw_B004 = np.array([0.213208020050125, 0.214745762711865, 0.368404737534727, 0.370619047619048, 
                    0.284150537634409, 0.283969568892646, 0.243361344537815, 0.233676875114616, 
                    0.208460686600221, 0.208887998397115, 0.189209699711019, 0.191699507389162, 
                    0.152599531615924, 0.165651365651366, 0.136269527573875, 0.148271728271729, 
                    0.139743492063492, 0.134092332833992, 0.21468416988417, 0.227485021967781, 
                    0.370490405117271, 0.36953120665742, 0.614019392372334, 0.628207094918504, 
                    0.723143645682744, 0.693631610942249, 0.807926887926888, 0.776729884412524, 
                    0.958964729764019, 0.886779987944545, 0.934317578332449, 1])

haw_B004 = np.array([207, 207, 192, 192, 176, 176, 167, 167, 159, 159, 136, 136, 115, 115, 96, 
                     96, 76, 76, 56, 56, 46, 46, 35, 35, 26, 26, 16, 16, 4, 4, 0, 0])

Pc_kpa_B004 = np.array([20.3067, 20.3067, 18.8352, 18.8352, 17.2656, 17.2656, 16.3827, 16.3827, 15.5979, 
                        15.5979, 13.3416, 13.3416, 11.2815, 11.2815, 9.4176, 9.4176, 7.4556, 7.4556, 5.4936, 
                        5.4936, 4.5126, 4.5126, 3.4335, 3.4335, 2.5506, 2.5506, 1.5696, 1.5696, 0.3924, 0.3924, 
                        0, 0])

Sw_B006 = np.array([0.458333333333333, 0.43137107980709, 0.33882001912235, 0.393006802721089, 
                    0.1178182589433, 0.124046097982484, 0.174709952474141, 0.167703175944934, 
                    0.155297735532569, 0.159672319632078, 0.203614586042993, 0.214513901891032, 
                    0.178871222047035, 0.186246355685131, 0.216292964554242, 0.197292571647132, 
                    0.211381706755321, 0.206659226190476, 0.270260927457558, 0.282242357523922, 
                    0.453766939846805, 0.40691117849477, 0.589559334026135, 0.53099459924485, 
                    0.571717247784723, 0.558749569866857, 0.782300990571667, 0.802127454755487, 
                    0.851893358499469, 0.852209711470796, 0.985170009784737, 1])

haw_B006 = np.array([269, 269, 258, 258, 233, 233, 218, 218, 198, 198, 174, 174, 149, 149, 124, 
                     124, 103, 103, 79, 79, 69, 69, 58, 58, 47, 47, 29, 29, 15, 15, 0, 0])

Pc_kpa_B006 = np.array([26.3889, 26.3889, 25.3098, 25.3098, 22.8573, 22.8573, 21.3858, 21.3858, 
                        19.4238, 19.4238, 17.0694, 17.0694, 14.6169, 14.6169, 12.1644, 12.1644, 
                        10.1043, 10.1043, 7.7499, 7.7499, 6.7689, 6.7689, 5.6898, 5.6898, 4.6107,
                        4.6107, 2.8449, 2.8449, 1.4715, 1.4715, 0, 0])

#BEGIN PLOTTING FIGURES
#%% -- Barplots
pfoacolors = plt.cm.Reds(np.linspace(0,1,5))
pfoscolors = plt.cm.Blues(np.linspace(0,1,5))
pfhxscolors = plt.cm.Greens(np.linspace(0,1,5))
pfdacolors = plt.cm.Purples(np.linspace(0,1,5))

# fig, ax = plt.subplots(1, 1, figsize=(20,4), dpi=200)
# fig, ax = plt.subplots(1, 1, figsize=(4,20), dpi=200)

labels = np.array(['PFOS', 'PFOA', 'PFHxS', 'PFDA'])
labels_salt = np.array(['PFOS$_{salt}$', 'PFOA$_{salt}$', 'PFHxS$_{salt}$', 'PFDA$_{salt}$'])
#y_ax = np.arange(len(labels))
# pfas_hbar = np.array([R_bar_pfos - R_bar_pfos_silt, R_bar_pfoa - R_bar_pfoa_silt, R_bar_pfhxs - R_bar_pfhxs_silt, R_bar_pfda - R_bar_pfda_silt])


lims = np.array([R_bar_pfos_silt, R_bar_pfoa_silt, R_bar_pfhxs_silt, R_bar_pfda_silt])
lims_nacl = np.array([R_bar_nacl_pfos_silt, R_bar_nacl_pfoa_silt, R_bar_nacl_pfhxs_silt, R_bar_nacl_pfda_silt])

# w = 0.7
# ax.bar(labels[2], pfas_hbar[2], width=w, bottom = lims[0], color = pfhxscolors[2], edgecolor = 'black')
# ax.bar(labels[1], pfas_hbar[1], width=w, bottom = lims[1], color = pfoacolors[2], edgecolor = 'black')
# ax.bar(labels[0], pfas_hbar[0], width=w, bottom = lims[2], color = pfoscolors[2], edgecolor = 'black')
# ax.bar(labels[3], pfas_hbar[3], width=w, bottom = lims[2], color = pfdacolors[2], edgecolor = 'black')

# ax.bar(labels[2], pfas_nacl_hbar[2], width=w, bottom = lims_nacl[0], color = pfhxscolors[2], edgecolor = 'black')
# ax.bar(labels[1], pfas_nacl_hbar[1], width=w, bottom = lims_nacl[1], color = pfoacolors[2], edgecolor = 'black')
# ax.bar(labels[0], pfas_nacl_hbar[0], width=w, bottom = lims_nacl[2], color = pfoscolors[2], edgecolor = 'black')
# ax.bar(labels[3], pfas_nacl_hbar[3], width=w, bottom = lims_nacl[2], color = pfdacolors[2], edgecolor = 'black')



# # ax.set_xscale('log')
# # ax.set_xlabel('Retardation Factor [-]', fontsize = 26, labelpad = 10)
# ax.set_ylabel('Retardation Factor [-]', fontsize = 26, labelpad = 10)
# ax.tick_params(axis='x', labelsize=24, rotation=-45)
# ax.tick_params(axis='y', labelsize=24)
# # ax.set_xlim(-100, 6500)
# ax.set_ylim(-100, 6000)
# ax.grid()

fig, ax = plt.subplots(1, 1, figsize=(12,4), dpi=200)

pfas_hbar = np.array([R_bar_pfos, R_bar_pfoa, R_bar_pfhxs, R_bar_pfda])
pfas_nacl_hbar = np.array([R_bar_nacl_pfos, R_bar_nacl_pfoa, R_bar_nacl_pfhxs, R_bar_nacl_pfda])
#pfhxs
ax.barh(labels[2], pfas_hbar[2], left = lims[2], edgecolor = 'black', color = pfhxscolors[2])
ax.barh(labels[2], R_bar_aw_pfhxs, left = R_bar_pfhxs_aw_silt, edgecolor = 'black', color = pfhxscolors[2], hatch = '/')
ax.barh(labels[2], R_bar_sp_pfhxs, left = R_bar_pfhxs_sp_silt, edgecolor = 'black', color = pfhxscolors[2], hatch = '..')

#pfoa
ax.barh(labels[1], pfas_hbar[1], left = lims[1], edgecolor = 'black', color = pfoacolors[2])
ax.barh(labels[1], R_bar_aw_pfoa, left = R_bar_pfoa_aw_silt, edgecolor = 'black', color = pfoacolors[2], hatch = '/')
ax.barh(labels[1], R_bar_sp_pfoa, left = R_bar_pfoa_sp_silt, edgecolor = 'black', color = pfoacolors[2], hatch = '..')

#pfos
ax.barh(labels[0], pfas_hbar[0], left = lims[0], edgecolor = 'black', color = pfoscolors[2])
ax.barh(labels[0], R_bar_aw_pfos, left = R_bar_pfos_aw_silt, edgecolor = 'black', color = pfoscolors[2], hatch = '/')
ax.barh(labels[0], R_bar_sp_pfos, left = R_bar_pfos_sp_silt, edgecolor = 'black', color = pfoscolors[2], hatch = '..')

#pfda
ax.barh(labels[3], pfas_hbar[3], left = lims[3], edgecolor = 'black', color = pfdacolors[2])
ax.barh(labels[3], R_bar_aw_pfda, left = R_bar_pfda_aw_silt, edgecolor = 'black', color = pfdacolors[2], hatch = '/')
ax.barh(labels[3], R_bar_sp_pfda, left = R_bar_pfda_sp_silt, edgecolor = 'black', color = pfdacolors[2], hatch = '..')

ax.set_xscale('log')
ax.set_xlabel('Retardation Factor [-]', fontsize = 26, labelpad = 10)
# ax.set_ylabel('Retardation Factor [-]', fontsize = 26, labelpad = 10)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
# ax.set_xlim(0, 10**5)
# ax.set_xlim(0, 6500)
ax.grid()

# fig, ax = plt.subplots(1, 1, figsize=(12,4), dpi=200)
#pfhxs salt
ax.barh(labels_salt[2], pfas_nacl_hbar[2], left = lims_nacl[2], edgecolor = 'black', color = pfhxscolors[2])
ax.barh(labels_salt[2], R_bar_aw_nacl_pfhxs, left = R_bar_aw_nacl_pfhxs_silt, edgecolor = 'black', color = pfhxscolors[2], hatch = '/')
ax.barh(labels_salt[2], R_bar_sp_nacl_pfhxs, left = R_bar_sp_nacl_pfhxs_silt, edgecolor = 'black', color = pfhxscolors[2], hatch = '..')

#pfoa salt
ax.barh(labels_salt[1], pfas_nacl_hbar[1], left = lims_nacl[1], edgecolor = 'black', color = pfoacolors[2])
ax.barh(labels_salt[1], R_bar_aw_nacl_pfoa, left = R_bar_aw_nacl_pfoa_silt, edgecolor = 'black', color = pfoacolors[2], hatch = '/')
ax.barh(labels_salt[1], R_bar_sp_nacl_pfoa, left = R_bar_sp_nacl_pfoa_silt, edgecolor = 'black', color = pfoacolors[2], hatch = '..')

#pfos salt
ax.barh(labels_salt[0], pfas_nacl_hbar[0], left = lims_nacl[0], edgecolor = 'black', color = pfoscolors[2])
ax.barh(labels_salt[0], R_bar_aw_nacl_pfos, left = R_bar_aw_nacl_pfos_silt, edgecolor = 'black', color = pfoscolors[2], hatch = '/')
ax.barh(labels_salt[0], R_bar_sp_nacl_pfos, left = R_bar_sp_nacl_pfos_silt, edgecolor = 'black', color = pfoscolors[2], hatch = '..')

ax.barh(labels[3], R_bar_sp_pfda, left = R_bar_pfda_sp_silt, edgecolor = 'black', color = pfdacolors[2], hatch = '..')
#pfda salt
ax.barh(labels_salt[3], pfas_nacl_hbar[3], left = lims_nacl[3], edgecolor = 'black', color = pfdacolors[2])
ax.barh(labels_salt[3], R_bar_aw_nacl_pfda, left = R_bar_aw_nacl_pfda_silt, edgecolor = 'black', color = pfdacolors[2], hatch = '/')
ax.barh(labels_salt[3], R_bar_sp_nacl_pfda, left = R_bar_sp_nacl_pfda_silt, edgecolor = 'black', color = pfdacolors[2], hatch = '..')

# ax.set_xscale('log')
# ax.set_xlabel('Retardation Factor [-]', fontsize = 26, labelpad = 10)
# # ax.set_ylabel('Retardation Factor [-]', fontsize = 26, labelpad = 10)
# ax.tick_params(axis='x', labelsize=24)
# ax.tick_params(axis='y', labelsize=24)
# ax.set_xlim(0, 10**5)
# # ax.set_ylim(-100, 6000)
# ax.grid()

#%% - Plots Capillary pressure curve
wrk_dir = "C:/Users/willg/OneDrive/Documents/2_School/GraduateSchool/Fall2021/!Research/PFAS_adsorption_paper_202109/SWCC_rhinelander"
os.chdir(wrk_dir)
#data -- #note the depth
Lab1 = pd.read_csv("B009004.csv") 
Lab2 = pd.read_csv("B0012004.csv")
Lab3 = pd.read_csv("B0012004DUP.csv")
Lab4 = pd.read_csv("B004002_Dense.csv")
Lab5 = pd.read_csv("B004002_Loose.csv")
Lab6 = pd.read_csv("B004007_Dense.csv")
Lab7 = pd.read_csv("B004007_Loose.csv")
Lab8 = pd.read_csv("B006009_Dense.csv")
Lab9 = pd.read_csv("B006009_Loose.csv")
Lab10 = pd.read_csv("B004013_Dense.csv")
Lab11 = pd.read_csv("B004013_Loose.csv")
Lab12 = pd.read_csv("B005003_Dense.csv")
Lab13 = pd.read_csv("B005003_Loose.csv")

Field1 = pd.read_csv("B004.csv")
Field2 = pd.read_csv("B006.csv")

#max_kpa = max()
h_corr = 100/(9.81*density)

#fig, ax =  plt.subplots(1, 1, figsize=(6, 5), dpi=200)
fig, (ax, ax2, ax1) = plt.subplots(1, 3, figsize=(14,8), sharey=True, gridspec_kw={'width_ratios': [5, 1.5, 1.5]}, dpi=200)

#testing new layout -- single central plot with organic carbon on right and pore size distribution under x axis
# fig = plt.figure(figsize=(15,16), dpi = 200)
# gs1 = fig.add_gridspec(nrows=4, ncols=4, wspace=0.5, hspace=0.7)
# ax  = fig.add_subplot(gs1[:3,:3])
# ax1 = fig.add_subplot(gs1[:-1,3])
# ax2 = fig.add_subplot(gs1[3,:-1])

lab_dat = 'dimgrey'


#coloring by depth
Bdepths = 'royalblue' #plt.cm.Blues(np.linspace(0,1,5))
Rdepths = 'indianred' #plt.cm.Reds(np.linspace(0,1,5))
Gdepths = 'forestgreen' #plt.cm.Greens(np.linspace(0,1,5))

field_dat = 'orangered'
#data points
# ax.scatter(Lab1['S (measured)'], Lab1['Applied Suction (kPa)']*h_corr, label = 'Lab measurements', c = lab_dat, s=70) #'B009004'
# ax.scatter(Lab2['S (measured)'], Lab2['Applied Suction (kPa)']*h_corr, label = 'Lab measurements', c = lab_dat, s=70) #'B0012004'
# ax.scatter(Lab3['S (measured)'], Lab3['Applied Suction (kPa)']*h_corr, label = 'Lab measurements', c = lab_dat, s=70) #'B0012004DUP'
#Loose measurements
ax.scatter(Lab5['S (measured)'], Lab5['Applied Suction (kPa)']*h_corr, label = 'B004002_Loose', c = Bdepths, s=70) 
ax.scatter(Lab7['S (measured)'], Lab7['Applied Suction (kPa)']*h_corr, label = 'B004007_Loose', c = Bdepths, s=70) 
ax.scatter(Lab9['S (measured)'], Lab9['Applied Suction (kPa)']*h_corr, label = 'B006009_Loose', c = Rdepths, s=70) 
ax.scatter(Lab11['S (measured)'], Lab11['Applied Suction (kPa)']*h_corr, label = 'B004013_Loose', c = Bdepths, s=70) 
B005 = ax.scatter(Lab13['S (measured)'], Lab13['Applied Suction (kPa)']*h_corr, label = 'B005', c = Gdepths, s=70) 
#Dense measurements
ax.scatter(Lab4['S (measured)'], Lab4['Applied Suction (kPa)']*h_corr, label = 'B004002_Dense', c = Bdepths, s=70) 
ax.scatter(Lab6['S (measured)'], Lab6['Applied Suction (kPa)']*h_corr, label = 'B004007_Dense', c = Bdepths, s=70) 
ax.scatter(Lab8['S (measured)'], Lab8['Applied Suction (kPa)']*h_corr, label = 'B006009_Dense', c = Rdepths, s=70) 
ax.scatter(Lab10['S (measured)'], Lab10['Applied Suction (kPa)']*h_corr, label = 'B004013_Dense', c = Bdepths, s=70) 
ax.scatter(Lab12['S (measured)'], Lab12['Applied Suction (kPa)']*h_corr, label = 'B005', c = Gdepths, s=70) #'B005003_Dense'

B004 = ax.scatter(Field1['S'], Field1['Pc kpa']*h_corr, label = 'B004', c = 'royalblue', s=70) #'B004'
ax.plot(Field1['S'], Field1['Pc kpa']*h_corr, ls = '--', lw = 2, c = 'royalblue')
B006 = ax.scatter(Field2['S'], Field2['Pc kpa']*h_corr, label = 'B006', c = 'indianred', s=70) #'B006'
ax.plot(Field2['S'], Field2['Pc kpa']*h_corr, ls = '--', lw = 2, c = 'indianred')


#some neat zoom in stuff
# axins = zoomed_inset_axes(ax, 1.5, loc=3)
# axins.scatter(Lab4['S (measured)'], Lab4['Applied Suction (kPa)']*h_corr, label = 'B004002_Dense', c = Bdepths, s=70)
# axins.set_xlim(0.8, 1.0)
# axins.set_ylim(0, 50)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

#Capillary Pressure Curves
fine = ax.plot(Sw_func(0.055, 4, Pc_kpa, Sr_ORA), haw, '--', c = 'black', linewidth = 3, label = 'silty-sand')
silt = ax.plot(Sw_func(0.08, 4, Pc_kpa, Sr_ORA), haw, '--', c = 'black', linewidth = 3, label = 'fine-sand')
sand = ax.plot(Sw_func(0.25, 4, Pc_kpa, Sr_ORA), haw, '--', c = 'black', linewidth = 3, label = 'gravelly-sand')

OC_content = np.array([0.0189829145728643, 0.0264462336322742, 0.0122093114671614, 0.0104005392872223, 0.00877704930458179, 
                       0.0132095912176801, 0.01699798553961, 0.0170841381931032, 0.018554129984075, 0.0323392365273025, 
                       0.019139905501722, 0.00930831601440175, 0.0354845879231549, 0.0309317832108053, 0.00768186640902385, 
                       0.00653082549634263, 0.00507228175932141])
np.average(OC_content)
haw_range = np.array([280, 260, 245, 229, 220, 212, 280, 270, 260, 252, 244, 233, 280, 270, 259, 234, 219])



ft = 30
ax.set_title('Capillary Pressure Profile', fontsize = ft, pad = 35)
ax.set_xlabel('Water saturation [-]', fontsize = 26, labelpad = 10)

ax.set_ylabel('Height Above Water Table [cm]', fontsize = 26, labelpad = 15)
ax.tick_params(axis='y', labelsize=24)
ax.set_ylim(0,max(haw))

ax.set_xlim(0.02,1.02)
ax.tick_params(axis='x', labelsize=24)

ax.grid()
ax.legend(handles=[B004, B005, B006], prop={'size': 22})


#Organic Carbon distribution
ax1.set_title("Organic\nCarbon", fontsize = ft, pad = 15)

ax1.scatter(OC_content, haw_range, zorder=10, s=70)
ax1.plot((2*10**-6)*np.exp(0.3621*(Pc_kpa)), Pc_kpa*h_corr, '--', c= 'orange', linewidth=3)

ax1.set_xlim([0, 0.05])
ax1.set_xticklabels([0, 0.025, 0.05], fontsize = 20)
ax1.set_xlabel('f$_{oc}$', fontsize = 24)
ax1.tick_params(axis='x', rotation=-15)

ax1.grid(True)


#gsd data % fines
depths = np.array([1.2192, 0.9144, 1.2192, 0.9144, 0.2, 0.35, 0.51, 0.6, 0.91, 1.12, 1.51, 1.81, 2.01, 
                   1.88, 1.78, 1.46, 1, 0.67, 0.47, 0.28, 0.1, 2.64, 2.32, 2.1, 1.76, 1.3, 0.81, 0.46])
depths_cm = np.multiply(depths, 100)
haw_gsd = np.subtract(max(haw), depths_cm)
sample = np.array(['B012', 'B012', 'B009', 'B009', 'B004', 'B004', 'B004', 'B004', 'B004', 'B004', 'B004', 
                   'B004', 'B004', 'B005', 'B005', 'B005', 'B005', 'B005', 'B005', 'B005', 'B005', 'B006', 
                   'B006', 'B006', 'B006', 'B006', 'B006', 'B006'])

fines_perc = np.array([1.51, 1.54, 4.49, 2.1, 8.75, 11.98, 10.61, 8.43, 7.11, 6.77, 2.41, 1.77, 7.78, 
                       1.88, 6.36, 5.73, 4.35, 9.35, 14.58, 13.11, 10.4, 1.45, 1.08, 1.85, 1.02, 0.71, 4.3, 4.26])

for s in range(0, len(sample), 1):
    if sample[s] == 'B004':
        ax2.scatter(fines_perc[s], haw_gsd[s], label = sample[s], c= 'royalblue', s = 70)
    elif sample[s] == 'B005':
        ax2.scatter(fines_perc[s], haw_gsd[s], label = sample[s], c= 'forestgreen', s = 70)
    elif sample[s] == 'B006':
        ax2.scatter(fines_perc[s], haw_gsd[s], label = sample[s], c= 'indianred', s = 70)

#plotting curves
ax2.plot(fines_perc[4:13], haw_gsd[4:13], lw = 3, ls = '--', c = 'royalblue')
ax2.plot(fines_perc[13:21], haw_gsd[13:21], lw = 3, ls = '--', c = 'forestgreen')
ax2.plot(fines_perc[21:], haw_gsd[21:], lw = 3, ls = '--', c = 'indianred')


ax2.set_title('Grain Size\nDistribution', fontsize=ft, pad = 15)

ax2.set_xlabel('Fines [%]', fontsize = 24, labelpad = 15)
ax2.set_xticklabels([0, 5, 10, 15], fontsize = 20)
ax2.tick_params(axis='x', rotation=-15)

ax2.grid()

#plt.xticks(sieve_sizes)

# IFT = 72.8 #water @20C
# theta_aw = 45*(pi/180)
# Sw_test = np.linspace(0, 1, 50)
# Pc_fine = Pc_func(0.05, 3/4, Sw_test, 0.05)
# Pc_silt = Pc_func(0.071, 3/4, Sw_test, 0.05)
# Pc_sand = Pc_func(0.3, 3/4, Sw_test, 0.05)

# pore_thrt_fine = 2*IFT*np.cos(theta_aw)/Pc_func(0.05, 3/4, Sw_test, 0.05)
# pore_thrt_silt = 2*IFT*np.cos(theta_aw)/Pc_func(0.071, 3/4, Sw_test, 0.05)
# pore_thrt_sand = 2*IFT*np.cos(theta_aw)/Pc_func(0.3, 3/4, Sw_test, 0.05)


# ax2.plot(Sw_test, pore_thrt_fine)
# ax2.plot(Sw_test, pore_thrt_silt)
# ax2.plot(Sw_test, pore_thrt_sand)

#ax2.set_ylim(0,300)
#ax2.set_yticklabels([])
#ax2.set_xticklabels([])
#ax2.set_title('Grain Size Distribution', fontsize = ft, pad = 15)

# fig_test, (ax, ax1) =  plt.subplots(1, 2, figsize=(8, 8), dpi=200)

# ax.plot(Sw_test, pore_thrt_fine)
# ax.plot(Sw_test, pore_thrt_silt)
# ax.plot(Sw_test, pore_thrt_sand)

# ax1.plot(Sw_test, Pc_fine, label = 'fine')
# ax1.plot(Sw_test, Pc_silt, label = 'silt')
# ax1.plot(Sw_test, Pc_sand, label = 'sand')
# ax1.legend()

#%% - 1D retaration profile for gravely-sand profile
#Plotting====================================================================================================


fig_R, (ax) =  plt.subplots(1, 1, figsize=(8, 8), dpi=200)

rpfos_aw, = ax.plot(R_pfos[:72], haw[:72], label = 'PFOS', linewidth = 3, c=pfoscolors[2])
rpfos_sp, = ax.plot(R_pfos[73:], haw[73:], linewidth = 3, c=pfoscolors[2])

rpfhxs_aw, = ax.plot(R_pfhxs[:72], haw[:72], label = 'PFHxS', linewidth = 3, c=pfhxscolors[2])
rpfhxs_sp, = ax.plot(R_pfhxs[73:], haw[73:], linewidth = 3, c=pfhxscolors[2])

rpfoa_aw, = ax.plot(R_pfoa[:72], haw[:72],  label = 'PFOA', linewidth = 3, c=pfoacolors[2])
rpfoa_sp, = ax.plot(R_pfoa[73:], haw[73:], linewidth = 3, c=pfoacolors[2])

rpfda_aw, = ax.plot(R_pfda[:72], haw[:72],  label = 'PFDA', linewidth = 3, c=pfdacolors[2])
rpfda_sp, = ax.plot(R_pfda[73:], haw[73:], linewidth = 3, c=pfdacolors[2])

rpfos_nacl_aw, = ax.plot(R_nacl_pfos[:72], haw[:72], label = '$PFOA_{NaCl}$', linewidth = 3, c=pfoscolors[2], ls = '--')
rpfos_nacl_sp, = ax.plot(R_nacl_pfos[73:], haw[73:], linewidth = 3, c=pfoscolors[2], ls = '--')

rpfhxs_nacl_aw, = ax.plot(R_nacl_pfhxs[:72], haw[:72], label = '$PFHxS_{NaCl}$', linewidth = 3, c=pfhxscolors[2], ls = '--')
rpfhxs_nacl_sp, = ax.plot(R_nacl_pfhxs[73:], haw[73:], linewidth = 3, c=pfhxscolors[2], ls = '--')

# rpfoa_nacl_aw, = ax.plot(R_nacl_pfoa[:72], haw[:72], label = '$R_{pfoa}$', linewidth = 3, c=pfoacolors[2])
# rpfoa_nacl_sp, = ax.plot(R_nacl_pfoa[73:], haw[73:], linewidth = 3, c=pfoacolors[2])

rpfda_nacl_aw, = ax.plot(R_nacl_pfda[:72], haw[:72], label = '$PFDA_{NaCl}$', linewidth = 3, c=pfdacolors[2], ls = '--')
rpfda_nacl_sp, = ax.plot(R_nacl_pfda[73:], haw[73:], linewidth = 3, c=pfdacolors[2], ls = '--')

#sp dominant line
sp_depth = haw[72]
ax.axhline(sp_depth, ls = ':', c = 'black')
#depth average values
# ax.axvline(R_bar_nacl_pfos, ls ='--', c=pfoscolors[2], label = round(R_bar_pfos))
# ax.axvline(R_bar_nacl_pfhxs, ls ='--', c=pfhxscolors[2], label = round(R_bar_pfhxs))
# #ax.axvline(R_bar_nacl_pfoa, ls ='--', c=pfoacolors[2], label = round(R_bar_pfoa))
# ax.axvline(R_bar_nacl_pfda, ls ='--', c=pfdacolors[2], label = round(R_bar_pfda))
ft = 26
ax.set_title('ORA steady-state homogeneous\nretardation profile ($C_0$ = {})'.format(conc_text[C]), pad = 20, fontsize = ft)
ax.set_xlabel('Retardation Factor [-]', fontsize = 24, labelpad = 15)
ax.set_ylabel('Heigth Above Water Table [cm]', fontsize = 24, labelpad = 15)
#ax.legend(handles=[rpfos_nacl, rpfhxs_nacl, rpfda_nacl])
#ax.set_xscale('log')
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=24)


#ax.set_xlim(0, 10**6)
ax.set_xscale('log')
ax.set_ylim(0, 280)
ax.grid('True')
ax.legend(loc = 2, prop={'size': 20})

# xvals = [520, 100, 250, 1250, 2100, 1400, 2500]
#labelLines(ax.get_lines(),fontsize=14, align=True)#, xvals=xvals)
fig_R.tight_layout()

#PFOS example=================================================================================================================
fig_PFOS_1D, ax =  plt.subplots(1, 1, figsize=(7, 6), dpi=200)
ax.plot(R_pfos, haw, label = 'gravely-sand', linewidth = 5, c=pfoscolors[2])
#ax.plot(R_pfos_silt[0], haw, label = 'fine-sand', linewidth = 4, c=pfoscolors[2])
ax.plot(R_pfos_fine[0], haw, label = 'silty-sand', linewidth = 5, c=pfoscolors[2])

sp_depth = haw[72]
#ax.axhline(sp_depth, ls = ':', c = 'black', lw = 3)
ax.axvline(R_bar_pfos, ls = '--', c = pfoscolors[3], lw = 3)
#ax.axvline(R_bar_pfos_silt, ls = '--', c = pfoscolors[3], lw = 3)
ax.axvline(R_bar_pfos_fine, ls = '--', c = pfoscolors[3], lw = 3)

ft = 26
ax.set_title('ORA steady-state homogeneous\nretardation profile ($C_0$ = {})'.format(conc_text[C]), pad = 20, fontsize = ft)
ax.set_xlabel('Retardation Factor [-]', fontsize = 24, labelpad = 15)
ax.set_ylabel('Heigth Above Water Table [cm]', fontsize = 24, labelpad = 15)
#ax.legend(handles=[rpfos_nacl, rpfhxs_nacl, rpfda_nacl])

ax.tick_params(axis='y', labelsize=22)
ax.tick_params(axis='x', labelsize=22)

ax.set_xscale('log')
ax.set_ylim(0, 280)
ax.grid('True')
#ax.legend(loc = 4, prop={'size': 20})
labelLines(ax.get_lines(), fontsize=24, align=True, xvals=[100,80,200])
#%%
#PFOS example 2
fig_PFOS_comp, (ax, ax1) = plt.subplots(2, 1, figsize=(6, 14), dpi=200)
linwid = 3
#ultrapure comparision
ax.plot(R_pfos, haw, label = 'PFOS$_{aw+sp}$', linewidth = linwid, c=pfoscolors[2])
ax.plot(Raw_pfos, haw, label = 'PFOS$_{aw}$', linewidth = linwid, c='indianred')
ax.plot(Rsp_pfos, haw, label = 'PFOS$_{sp}$', linewidth = linwid, c='mediumpurple')

ax.axvline(R_bar_pfos, ls = '--', c = pfoscolors[3], lw = 3)
ax.axvline(R_bar_aw_pfos, ls = '--', c = 'indianred', lw = 3)
ax.axvline(R_bar_sp_pfos, ls = '--', c = 'mediumpurple', lw = 3)

#saline
ax1.plot(R_nacl_pfos, haw, label = 'PFOS$_{aw+sp}$', linewidth = linwid, c=pfoscolors[2])
ax1.plot(Raw_nacl_pfos, haw, label = 'PFOS$_{aw}$', linewidth = linwid, c='indianred')
ax1.plot(Rsp_nacl_pfos, haw, label = 'PFOS$_{sp}$', linewidth = linwid, c='mediumpurple')

ax1.axvline(R_bar_nacl_pfos, ls = '--', c = pfoscolors[3], lw = 3)
ax1.axvline(R_bar_aw_nacl_pfos, ls = '--', c = 'indianred', lw = 3)
ax1.axvline(R_bar_sp_nacl_pfos, ls = '--', c = 'mediumpurple', lw = 3)

ax.set_title('Ultrapure vs Saline', fontsize = 24, pad =15)
#ax1.set_title('PFOS Retardation: Saline', fontsize = 24, pad =15)

#ax.set_xlabel('Retardation Factor [-]', fontsize = 22, labelpad=15)
ax.set_ylabel('Height Above Water Table [-]', fontsize = 22, labelpad=15)
ax1.set_ylabel('Height Above Water Table [-]', fontsize = 22, labelpad=15)
ax1.set_xlabel('Retardation Factor [-]', fontsize = 22, labelpad=15)

ax.set_ylim(0, max(haw))
ax1.set_ylim(0, max(haw))
# ax1.set_xlim()

ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax1.tick_params(axis='x', labelsize=20)

ax.legend(loc = 4, prop={'size': 18})
ax1.legend(loc = 4, prop={'size': 16})

ax.grid()
ax1.grid()



#%% - Surface plots -- note something above changes the surface map so run this before making the other plots

#ranging across values of av and pc
av = np.array(np.linspace(0.01, 1, 100))

#call surface retardation function
#PFOS=============================================================================================================
R_out_pfos = vadoseRetaration_2Dprofile_func(av, n_val, Sr_ORA, Pc_kpa, s, RH_U, phi, rhob, 
                                                                    pfos_kaw, pfos_koc, OC_dis, pfos_km)

#PFHxS=============================================================================================================
R_out_pfhxs = vadoseRetaration_2Dprofile_func(av, n_val, Sr_ORA, Pc_kpa, s, RH_U, phi, rhob, 
                                                                    pfhxs_kaw, pfhxs_koc, OC_dis, 0)

#PFOA=============================================================================================================
R_out_pfoa = vadoseRetaration_2Dprofile_func(av, n_val, Sr_ORA, Pc_kpa, s, RH_U, phi, rhob, 
                                                                    pfoa_kaw, pfoa_koc, OC_dis, 0)

#PFOA=============================================================================================================
R_out_pfda = vadoseRetaration_2Dprofile_func(av, n_val, Sr_ORA, Pc_kpa, s, RH_U, phi, rhob, 
                                                                    pfda_kaw, pfda_koc, OC_dis, 0)

#1D range of Sw values 
for i in av:
    for j in Pc_kpa:
        Sw = Sw_func(av, 4, Pc_kpa, Sr_ORA)
  
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
       [ax2, ax3]) =  plt.subplots(2, 2, figsize=(20, 16), dpi=200)

#fig_2D, (ax1, ax2, ax, ax3) =  plt.subplots(1, 4, figsize=(28, 8), dpi=200)

ft = 30
pds = 20
level_test = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0]

im = ax.contourf(Sw, haw, np.log10(R_out_pfos[0]), level_test, cmap = 'Blues')
cbar = fig_2D.colorbar(im, ax=ax)
cbar.set_label(label='$Log_{10}$Retardation Factor [-]', size = ft, labelpad = pds)
cbar.ax.tick_params(labelsize=ft) 
for label in cbar.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)


#ax.set_title('PFOS retardation, $C_0 = {}$'.format(conc_text[C]), fontsize = ft, pad=pds)
#ax.set_ylabel('Height above water table [cm]', fontsize = ft)
ax.tick_params(axis='y', labelsize=ft)
ax.tick_params(axis='x', labelsize=ft)

im1 = ax1.contourf(Sw, haw, np.log10(R_out_pfhxs[0]), level_test, cmap = 'Greens')
cbar = fig_2D.colorbar(im1, ax=ax1)
cbar.set_label(label='$Log_{10}$Retardation Factor [-]', size = ft, labelpad = 20)
cbar.ax.tick_params(labelsize=ft) 
for label in cbar.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

#ax1.set_title('PFHxS retardation, $C_0 = {}$'.format(conc_text[C]), fontsize = ft, pad=pds)
ax1.tick_params(axis='y', labelsize=ft)
ax1.tick_params(axis='x', labelsize=ft)

im2 = ax2.contourf(Sw, haw, np.log10(R_out_pfoa[0]), level_test, cmap = 'Reds')
cbar = fig_2D.colorbar(im2, ax=ax2)
cbar.set_label(label='$Log_{10}$Retardation Factor [-]', size = ft, labelpad = 20)
cbar.ax.tick_params(labelsize=ft) 
for label in cbar.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

#ax2.set_title('PFOA retardation, $C_0 = {}$'.format(conc_text[C]), fontsize = ft, pad=pds)
#ax2.set_ylabel('Height above water table [cm]', fontsize = ft)
#ax2.set_xlabel('water saturation [-]', fontsize = ft, labelpad = pds)
ax2.tick_params(axis='y', labelsize=ft)
ax2.tick_params(axis='x', labelsize=ft)

im3 = ax3.contourf(Sw, haw, np.log10(R_out_pfda[0]), level_test, cmap = 'Purples')
cbar = fig_2D.colorbar(im3, ax=ax3)
cbar.set_label(label='$Log_{10}$Retardation Factor [-]', size = ft, labelpad = 20)
cbar.ax.tick_params(labelsize=ft) 
for label in cbar.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

#ax3.set_title('PFDA retardation, $C_0 = {}$'.format(conc_text[C]), fontsize = ft, pad=pds)
#ax3.set_xlabel('water saturation [-]', fontsize = ft, labelpad = pds)
ax3.tick_params(axis='y', labelsize=ft)
ax3.tick_params(axis='x', labelsize=ft)

#actual B006 profile
Sw_B004 = np.array([0.213208020050125, 0.214745762711865, 0.368404737534727, 0.370619047619048, 
                    0.284150537634409, 0.283969568892646, 0.243361344537815, 0.233676875114616, 
                    0.208460686600221, 0.208887998397115, 0.189209699711019, 0.191699507389162, 
                    0.152599531615924, 0.165651365651366, 0.136269527573875, 0.148271728271729, 
                    0.139743492063492, 0.134092332833992, 0.21468416988417, 0.227485021967781, 
                    0.370490405117271, 0.36953120665742, 0.614019392372334, 0.628207094918504, 
                    0.723143645682744, 0.693631610942249, 0.807926887926888, 0.776729884412524, 
                    0.958964729764019, 0.886779987944545, 0.934317578332449, 1
])

haw_B004 = np.array([207, 207, 192, 192, 176, 176, 167, 167, 159, 159, 136, 136, 115, 115, 96, 
                     96, 76, 76, 56, 56, 46, 46, 35, 35, 26, 26, 16, 16, 4, 4, 0, 0
])

axes = [ax, ax1, ax2, ax3]
for i in axes:
    i.plot(Sw_func(0.3, n_val, Pc_kpa, Sr_ORA), haw, "--", c = 'black', linewidth = 3, label = "gravely-sand")
    i.plot(Sw_func(0.05, n_val, Pc_kpa, Sr_ORA), haw, '--', c = 'black', linewidth = 3, label = 'silty-sand')
    #i.plot(Sw_func(0.071, n_val, Pc_kpa, Sr_ORA), haw, '--', c = 'black', linewidth = 3, label = 'fine-sand')
    i.scatter(Sw_B006, haw_B006, c = 'white', linewidth = 3, s = 200, zorder = 10, alpha = 0.7, edgecolors='black')
    i.plot(Sw_B006, haw_B006, "--", c = 'black', linewidth = 3, alpha = 0.5)
    labelLines(i.get_lines(),fontsize=30, align=True, xvals=[0.3, 0.6, 0.6])
    #i.scatter(Sw_B004, haw_B004, color = 'black', label = "B004")
    
at = AnchoredText(
    "A", prop=dict(size=30), frameon=True, loc='upper right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)
at1 = AnchoredText(
    "B", prop=dict(size=30), frameon=True, loc='upper right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
at2 = AnchoredText(
    "C", prop=dict(size=30), frameon=True, loc='upper right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at2)
at3 = AnchoredText(
    "D", prop=dict(size=30), frameon=True, loc='upper right')
at3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at3)

fig_2D.tight_layout()
#%%
#PFOS example
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

ft =20
fig_PFOS_surf, ax =  plt.subplots(1, 1, figsize=(6, 9), dpi=200)


im = ax.contourf(Sw, haw, np.log10(R_out_pfos[0]), 5, cmap = 'Blues')
cbar = fig_PFOS_surf.colorbar(im, ax = ax, orientation ='horizontal', pad = 0.15)
cbar.set_label(label='$Log_{10}$Retardation Factor [-]', size = ft, labelpad = 5)
cbar.ax.tick_params(labelsize=ft) 

ax.plot(Sw_func(0.3, n_val, Pc_kpa, Sr_ORA), haw, "--", c = 'black', linewidth = 3, label = "gravely-sand")
ax.plot(Sw_func(0.05, n_val, Pc_kpa, Sr_ORA), haw, '--', c = 'black', linewidth = 3, label = 'silty-sand')
#ax.plot(Sw_func(0.071, n_val, Pc_kpa, Sr_ORA), haw, '--', c = 'black', linewidth = 3, label = 'fine-sand')
labelLines(ax.get_lines(),fontsize=20, align=True, xvals=[0.6, 0.6, 0.6])

axins = zoomed_inset_axes(ax, 1.5, loc=3)
axins.contourf(Sw, haw,np.log10(R_out_pfos[0]), cmap="Blues")
axins.set_xlim(0.4, 0.6)
axins.set_ylim(100, 150)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

ax.set_title('PFOS retardation, $C_0 = {}$'.format(conc_text[C]), fontsize = ft, pad=pds)
ax.set_ylabel('Height above water table [cm]', fontsize = ft, labelpad = 10)
ax.set_xlabel('Water saturation [-]', fontsize = ft)
ax.tick_params(axis='y', labelsize=ft)
ax.tick_params(axis='x', labelsize=ft)



