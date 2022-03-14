# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:41:11 2022
Vadose zone adsorption functins
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

#import ADRs
#from ADRs import *

#%matplotlib inline
#%matplotlib qt
#%% - functions
def trip_averages(data, n):
    data_ave = [sum(data[i:i+n])/n for i in range(0,len(data),n)]
    return data_ave

def conc_star(conc_pfas, anion, cation): 
    I = 0.5*(conc_pfas/1000 + anion/1000 + cation/1000) # - Ionic strength calculations are in molarity
    logy = (0.507*np.sqrt(I))/(1+np.sqrt(I))+0.1*I #activity
    y = 10**(-logy)
    conc_star = np.sqrt((y*(conc_pfas+cation))*(y*conc_pfas)) #C*
    return conc_star

def conc_star_salt(conc_pfas, anion, cation): 
    I = 0.5*(conc_pfas/1000 + anion/1000 + cation/1000) # - Ionic strength calculations are in molarity
    logy = (0.507*np.sqrt(I))/(1+np.sqrt(I))+0.1*I #activity
    y = 10**(-logy)
    conc_star = (np.sqrt((y*(conc_pfas+cation))*(y*conc_pfas)))**2 #C*
    return conc_star
    
#surface tension estimates
def freundlich_sft(wtr, T, conc_star, k, n):
    sft = wtr - (8310*T)*k*((conc_star**n)/n)
    return sft

def langmuir_sft(wtr, T, conc_star, T_max, a):
    sft = wtr - (8310*T)*T_max*np.log(1 + a*conc_star)
    return sft
    
#Kaw estimamtes
def freundlich_kaw(k, n, conc_star, conc):
    surf_ex = k*conc_star**n
    kaw = surf_ex/conc
    return surf_ex, kaw

def langmuir_kaw(T_max, a, conc_star, conc):
    surf_ex = (T_max*a*conc_star)/(1+ a*conc_star)
    kaw = surf_ex/conc
    return surf_ex, kaw

#Retardation Modeling
def Aia_Sw_func2(s, U, S, Sr, Sm):
    a = 14.3*np.log(U)+3.72
    if np.any(U < 3.5):
        m = 0.098*U+1.53
    else:
        m = 1.2
    n = 1/(2-m)
    Aia = s*(1+(a*(Sr+(1-Sr)*(S-Sm))**n))**-m
    return Aia

def dimlessSw_func(a, n, Pc, S, Sr):
    S_eff = (S - Sr)/(1 - Sr)
    m = 1 - (1/n)
    Sw = S_eff*(1+(a*Pc)**n)**-m
    return Sw

def normSw_func(a, n, Pc, S, Sr):
    S_eff = (S - Sr)/(1 - Sr)
    m = 1 - (1/n)
    Sw = S_eff*(1+(a*Pc)**n)**-m
    return Sw

def Sw_func(a, n, Pc, Sr):
    m = 1 - (1/n)
    Sw = Sr+(1-Sr)*(1+(a*Pc)**n)**-m
    return Sw
#%%