# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:29:09 2022
SWCC
@author: willg
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# for analytical solutions
import scipy
from scipy import optimize
from scipy.special import erfc as erfc
from math import pi
from matplotlib import rc
import matplotlib.ticker as mtick
from labellines import labelLine, labelLines
rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams['font.size'] = 16
#%matplotlib inline
#%%
def Sw_func(a, n, Pc, Sr):
    m = 1 - (1/n)
    Sw = Sr+(1-Sr)*(1+(a*Pc)**n)**-m
    return Sw

def normSw_func(a, n, Pc, S, Sr):
    S_eff = (S - Sr)/(1 - Sr)
    m = 1 - (1/n)
    Sw = S_eff*(1+(a*Pc)**n)**-m
    return Sw

def van_g_pc(Sw, Pc_entry, n):
    # Now calculate the effective saturation (think of this as normalized saturation (ranges from 0-1))
    Pc = Pc_entry*(Sw**(-1/n)-1)**(1/n) 
    return Pc

#%%
wrk_dir = "C:/Users/willg/OneDrive/Documents/2_School/GraduateSchool/Fall2021/!Research/PFAS_adsorption_paper_202109/SWCC_rhinelander"
os.chdir(wrk_dir)
#data
Lab1 = pd.read_csv("B009004.csv")
Lab2 = pd.read_csv("B0012004.csv")
Lab3 = pd.read_csv("B0012004DUP.csv")
Field1 = pd.read_csv("B004.csv")
Field2 = pd.read_csv("B006.csv")
Pc = np.linspace(0, 30, 50)

#fig, ax =  plt.subplots(1, 1, figsize=(6, 5), dpi=200)
fig, (ax, ax2, ax1) = plt.subplots(1, 3, figsize=(10,6), gridspec_kw={'width_ratios': [5, 1, 1]}, dpi=200)
h_corr = 10.194
lab_dat = 'royalblue'
field_dat = 'indianred'
#data points
ax.scatter(Lab1['S (measured)'], Lab1['Applied Suction (kPa)']*h_corr, label = 'Lab measurements', c = lab_dat) #'B009004'
ax.scatter(Lab2['S (measured)'], Lab2['Applied Suction (kPa)']*h_corr, label = 'Lab measurements', c = lab_dat) #'B0012004'
ax.scatter(Lab3['S (measured)'], Lab3['Applied Suction (kPa)']*h_corr, label = 'Lab measurements', c = lab_dat) #'B0012004DUP'

ax.scatter(Field1['S'], Field1['Pc kpa']*h_corr, label = 'Field measurements', c = field_dat) #'B004'
ax.scatter(Field2['S'], Field2['Pc kpa']*h_corr, label = 'Field measurements', c = field_dat) #'B006'

silt = ax.plot(Sw_func(0.25, 4, Pc, 0.39), Pc*h_corr, '--', c = 'grey', linewidth = 2)
gravel = ax.plot(Sw_func(0.67, 4, Pc, 0.035), Pc*h_corr, '--', c = 'grey')
measured = ax.plot(Sw_func(0.3, 4, Pc, 0.125), Pc*h_corr, '--', c = 'black', linewidth = 2)

OC_content = np.array([0.0189829145728643, 0.0264462336322742, 0.0122093114671614, 0.0104005392872223, 0.00877704930458179, 
                       0.0132095912176801, 0.01699798553961, 0.0170841381931032, 0.018554129984075, 0.0323392365273025, 
                       0.019139905501722, 0.00930831601440175, 0.0354845879231549, 0.0309317832108053, 0.00768186640902385, 
                       0.00653082549634263, 0.00507228175932141])
haw_range = np.array([280, 260, 245, 229, 220, 212, 280, 270, 260, 252, 244, 233, 280, 270, 259, 234, 219])
#lines = [silt, gravel]
#labelLines(ax.get_lines(), fontsize = 10)

#ax.scatter(np.multiply(OC_content, 1), haw_range)

ax.set_title('ORA Heterogeneity', pad=15)
ax.set_xlabel('Water saturation [-]')
#ax.set_ylabel('Height above water table [kPa]')
ax.set_ylabel('Heigth above water table [cm]')
ax.set_xlim(0,1.05)
ax.set_ylim(-5,300)
ax.grid()
#ax.legend(handles=[silt, gravel])

#fig, ax1 = plt.subplots(1, 1, figsize=(1, 5), dpi=200)
ax1.set_yticklabels([])
ax1.scatter(OC_content, haw_range, zorder=10)
ax1.plot((2*10**-6)*np.exp(0.3621*(Pc)), Pc*h_corr, '--', c= 'orange', linewidth=3)
#ax1.yaxis.tick_right()
ax1.set_ylim(-5, 300)
ax1.set_xlim(0, 0.05)
ax1.set_title("Organic Carbon", fontsize = 12)
ax1.set_xlabel('f$_{oc}$')

ax1.grid(True)
plt.xticks(rotation=-45)

ax2.set_ylim(-5,300)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_title('Lithology', fontsize = 12)

#%%
#van g fits normalized
fig, ax =  plt.subplots(1, 1, figsize=(8, 6), dpi=200)
haw = (Pc / 9.81)*100 #kpa -> cm

ax.plot(normSw_func(1/4.47, 4.509, Pc, 1, 0.049), haw, label = 'B009004 - fit')
ax.plot(normSw_func(1/4.32, 5.45, Pc, 1, 0.057), haw, label = 'B0012004 - fit')
ax.plot(normSw_func(1/2.96, 4.549, Pc, 1, 0.124), haw, label = 'B0012004DUP - fit')
ax.plot(normSw_func(1/2.88, 4.39, Pc, 1, 0.221), haw, label = 'B004 - fit')
ax.plot(normSw_func(1/2.88, 2.7, Pc, 1, 0.089), haw, label = 'B006 - fit')

#median entry pressures of different soils - https://doi.org/10.1016/j.enggeo.2020.105911
por_dis = 4.3
ax.plot(normSw_func(1/6, 4.4, Pc, 1, 0.221), haw, '--', label = 'Silty-Sand')
ax.plot(normSw_func(1/2, 4.4, Pc, 1, 0.09), haw, '--', label = 'Gravely-Sand')

ax.set_title('Normalized ORA SWCC')
ax.set_xlabel('Effective Saturation [-]')
ax.set_ylabel('HAW [cm]')
ax.set_xlim(-0.05,1.05)
ax.grid()
ax.legend()
lines = ax.get_lines()
#%%
Sw = np.linspace(0.01, 1, 50)
Pc = van_g_pc(Sw, 0.2, 4) #kPa = kN/m2
phi = 0.42
sigma = 71.98/(10**6) #SFT of water kN/m

dSw = np.linspace(0.6, 1)
Awi = phi/sigma * np.trapz(Pc, dSw, 0.1) #1/m
Awi_cm = Awi*100 #1/cm

    