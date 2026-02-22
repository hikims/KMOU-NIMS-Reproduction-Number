"""
Main entry script for age-structured influenza Rt estimation.

This script reproduces the original workflow while making paths GitHub-friendly:
- avoids hard-coded local absolute paths
- uses an optional environment variable INFLUENZA_PROJECT_ROOT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import math
import time
import os  
from pathlib import Path

# Project root (override via environment variable if needed)
PROJECT_ROOT = Path(os.environ.get('INFLUENZA_PROJECT_ROOT', Path(__file__).resolve().parent)).resolve()
PATH = str(PROJECT_ROOT)

from SIR_calc import SIR_calc
from Gorji import Gorji
from particle import ps_age_Rt 

OUT_DIR = os.path.join(PATH, 'Pre_defined', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed()
start = time.time()

# # Contact Matrix
# survey = PATH + '/data/mij.csv'
# dfmatrix = pd.read_csv(survey)
# contact = dfmatrix.to_numpy()
# contact_leng = len(contact)

# Population
fn = PATH + '/data/Pops_Dec2022.csv'
with open(fn, encoding='EUC-KR') as file:
    pop_raw = pd.read_csv(file)

pops = [int(s.replace(",", "")) for s in pop_raw.iloc[0, 3:]]

lbeag = [0, 7, 13, 19, 50, 65]
ubeag = [6, 12, 18, 49, 64, 100]  
population = [sum(pops[lbeag[i]:ubeag[i]+1]) for i in range(6)]

# Synthetic epicurve
length = 240

b_max = [0.037, 0.026, 0.026, 0.029, 0.029, 0.025]
phi = [-40, -20, 0, -120, -80, -40]
period = [180, 180, 180, 720, 720, 720]
b_base = [0.037, 0.027, 0.027, 0.032, 0.03, 0.03]

b = np.zeros((6, length))
for i in range(6):
    b[i, :] = b_base[i] + b_max[i] * np.sin(2*np.pi/period[i] * (np.arange(1, length + 1) - phi[i]))

# Initial condition    
S = np.zeros((6, length))
I = np.zeros((6, length))
R = np.zeros((6, length))


population_age = np.array(population) 
N = population_age.astype(float)

# Contact matrix
M = np.array([
    [1.76, 0.21, 0.03, 0.2, 0.05, 0.05],
    [0.33, 3.75, 0.3, 0.31, 0.14, 0.13],
    [0.05, 0.31, 3.65, 0.22, 0.31, 0.12],
    [2.07, 2.11, 1.38, 1.78, 1.26, 0.84],
    [0.47, 0.88, 1.87, 1.18, 1.97, 1.47],
    [0.3, 0.47, 0.42, 0.45, 0.83, 2.5]
])

# Mean infectious period 5 days
sigma = 1/5.

I0 = 5 * np.array([119.327, 90.5102, 37.6122, 135.469, 60.4898, 44.6327])
S0 = N - I0
R0 = np.zeros(6)

S[:, 0] = S0
I[:, 0] = I0
R[:, 0] = R0

# Time evolution  
for t in range(length - 1):
    output = SIR_calc(S[:, t], I[:, t], R[:, t], N, M, b[:, t], sigma)
    S[:, t + 1] = output[0]
    I[:, t + 1] = output[1]
    R[:, t + 1] = output[2]
    
pre_confirm = pd.DataFrame({
    't': np.arange(1, length),  
    'age1': R[0, 1:], 
    'age2': R[1, 1:], 
    'age3': R[2, 1:], 
    'age4': R[3, 1:], 
    'age5': R[4, 1:], 
    'age6': R[5, 1:]
})  

pre_confirm1 = pre_confirm["age1"]
pre_confirm2 = pre_confirm["age2"]
pre_confirm3 = pre_confirm["age3"]
pre_confirm4 = pre_confirm["age4"]
pre_confirm5 = pre_confirm["age5"]
pre_confirm6 = pre_confirm["age6"]

T = min(len(pre_confirm1), len(pre_confirm2), len(pre_confirm3),
        len(pre_confirm4), len(pre_confirm5), len(pre_confirm6))

# Pre_Rt
Rt = np.zeros((length, 6))
S_over_N = S.T / N.reshape(1, -1) 
for j in range(6):
    for t in range(length):
        for i in range(6):
            Rt[t, j] += b[i, t] / sigma * S_over_N[t, i] * M[i, j]

pre_Rt = pd.DataFrame({
    't': np.arange(1, length),  
    'age1': Rt[1:, 0], 
    'age2': Rt[1:, 1], 
    'age3': Rt[1:, 2], 
    'age4': Rt[1:, 3], 
    'age5': Rt[1:, 4], 
    'age6': Rt[1:, 5]
})   
            
pre_confirm.to_csv(os.path.join(OUT_DIR, 'pre_confirm.csv'), index=False)
pre_Rt.to_csv(os.path.join(OUT_DIR, 'pre_Rt.csv'), index=False)        

# Gorji Method
pre_Gorji_Rt1, pre_Gorji_Rt2, pre_Gorji_Rt3, pre_Gorji_Rt4, pre_Gorji_Rt5, pre_Gorji_Rt6 = Gorji(pre_confirm1, pre_confirm2, pre_confirm3, pre_confirm4, pre_confirm5, pre_confirm6, M, "Pre_Gorji")

Gorji_Rt = pd.DataFrame({
        'age1': pre_Gorji_Rt1,  
        'age2': pre_Gorji_Rt2,
        'age3': pre_Gorji_Rt3,
        'age4': pre_Gorji_Rt4,
        'age5': pre_Gorji_Rt5,
        'age6': pre_Gorji_Rt6,
    })

Gorji_Rt.to_csv(os.path.join(OUT_DIR, 'Gorji_Rt.csv'), index=False)

# Particle Method
pre_age_confirm1, pre_age_confirm2, pre_age_confirm3, pre_age_confirm4, \
pre_age_confirm5, pre_age_confirm6, pre_age_pf_Rt1, pre_age_pf_Rt2, \
pre_age_pf_Rt3, pre_age_pf_Rt4, pre_age_pf_Rt5, pre_age_pf_Rt6, \
pre_age_ps_Rt1, pre_age_ps_Rt2, pre_age_ps_Rt3, pre_age_ps_Rt4, \
pre_age_ps_Rt5, pre_age_ps_Rt6 = ps_age_Rt(
    T, pre_confirm1, pre_confirm2, pre_confirm3,
    pre_confirm4, pre_confirm5, pre_confirm6, "Pre-defined PS Rt"
)

pre_age_confirm = pd.DataFrame({
        'age1': pre_age_confirm1,  
        'age2': pre_age_confirm2,
        'age3': pre_age_confirm3,
        'age4': pre_age_confirm4,
        'age5': pre_age_confirm5,
        'age6': pre_age_confirm6,
    })

pre_Pf_Rt = pd.DataFrame({
        'age1': pre_age_pf_Rt1[1:],  
        'age2': pre_age_pf_Rt2[1:],
        'age3': pre_age_pf_Rt3[1:],
        'age4': pre_age_pf_Rt4[1:],
        'age5': pre_age_pf_Rt5[1:],
        'age6': pre_age_pf_Rt6[1:],
    })

pre_Ps_Rt = pd.DataFrame({
        'age1': pre_age_ps_Rt1[1:],  
        'age2': pre_age_ps_Rt2[1:],
        'age3': pre_age_ps_Rt3[1:],
        'age4': pre_age_ps_Rt4[1:],
        'age5': pre_age_ps_Rt5[1:],
        'age6': pre_age_ps_Rt6[1:],
    })

pre_age_confirm.to_csv(os.path.join(OUT_DIR, 'pre_age_confirm.csv'), index=False)
pre_Pf_Rt.to_csv(os.path.join(OUT_DIR, 'pre_Pf_Rt.csv'), index=False)
pre_Ps_Rt.to_csv(os.path.join(OUT_DIR, 'pre_Ps_Rt.csv'), index=False)


# import numpy as np
# import pandas as pd
# import math
# import time
# # from data_process import data_process
# # from data_process import moving_average1 
# # from distribution import weibull_dist
# # from case import All_case, Age_case
# # from instantaneous import All_instantaneous, Age_instantaneous
# # from plot import Rt_plotting, confirm_plotting, SEIR_confirm_plotting, SEIR_Rt_plotting, whole_pre_Rt_plotting, part_pre_Rt_plotting, pre_Rt_age_plotting
# # from SEIR import All_SEIR, Age_SEIR
# from pre_Rt import Pre_Rt
# from Gorji import Gorji
# # import matplotlib.pyplot as plt
# PATH = '/var2/mkim/Final_influenza/'

# np.random.seed()
# start = time.time()

# window = 7

# # Contact Matrix
# survey = PATH + '/data/mij.csv'
# dfmatrix = pd.read_csv(survey)
# contact = dfmatrix.to_numpy()
# contact_leng = len(contact)

# # Pre-defined Rt
# day = 240
# pre_col = 6
# pre_sigma = 1.0 / 5.0
# pre_beta = np.zeros([day, pre_col])

# # Sinusoidal wave
# b_max = np.array([0.037, 0.026, 0.026, 0.029, 0.029, 0.025])
# phi = np.array([-40, -20, 0, -120, -80, -40])
# period = np.array([180, 180, 180, 720, 720, 720])
# b_base = np.array([0.037, 0.027, 0.027, 0.032, 0.03, 0.03])
# pre_beta = np.zeros([day, pre_col])

# for i in range(pre_col): 
#     pre_beta[:, i] = b_base[i] + b_max[i] * np.sin(2*np.pi/period[i] * ((np.arange(, day)) - phi[i]))
    
     
# pre_Rt1, pre_Rt2, pre_Rt3, pre_Rt4, pre_Rt5, pre_Rt6, Pre_confirm1, Pre_confirm2, Pre_confirm3, Pre_confirm4, Pre_confirm5, Pre_confirm6, pre_age_confirm1, pre_age_confirm2, pre_age_confirm3, pre_age_confirm4, pre_age_confirm5, pre_age_confirm6, pre_age_SEIR_pf_Rt1, pre_age_SEIR_pf_Rt2, pre_age_SEIR_pf_Rt3, pre_age_SEIR_pf_Rt4, pre_age_SEIR_pf_Rt5, pre_age_SEIR_pf_Rt6, pre_age_SEIR_ps_Rt1, pre_age_SEIR_ps_Rt2, pre_age_SEIR_ps_Rt3, pre_age_SEIR_ps_Rt4, pre_age_SEIR_ps_Rt5, pre_age_SEIR_ps_Rt6 = Pre_Rt(day, pre_col, pre_sigma, pre_beta, "Pre-defined Rt")
# pre_Gorji_Rt1, pre_Gorji_Rt2, pre_Gorji_Rt3, pre_Gorji_Rt4, pre_Gorji_Rt5, pre_Gorji_Rt6 = Gorji(pre_age_confirm1, pre_age_confirm2, pre_age_confirm3, pre_age_confirm4, pre_age_confirm5, pre_age_confirm6, contact, window, "Pre_Gorji")

# pre_confirm = pd.DataFrame({
#         'age1': Pre_confirm1,  
#         'age2': Pre_confirm2,
#         'age3': Pre_confirm3,
#         'age4': Pre_confirm4,
#         'age5': Pre_confirm5,
#         'age6': Pre_confirm6,
#     })

# pre_age_confirm = pd.DataFrame({
#         'age1': pre_age_confirm1,  
#         'age2': pre_age_confirm2,
#         'age3': pre_age_confirm3,
#         'age4': pre_age_confirm4,
#         'age5': pre_age_confirm5,
#         'age6': pre_age_confirm6,
#     })

# pre_Rt = pd.DataFrame({
#         'age1': pre_Rt1,  
#         'age2': pre_Rt2,
#         'age3': pre_Rt3,
#         'age4': pre_Rt4,
#         'age5': pre_Rt5,
#         'age6': pre_Rt6,
#     })

# Pf_Rt = pd.DataFrame({
#         'age1': pre_age_SEIR_pf_Rt1,  
#         'age2': pre_age_SEIR_pf_Rt2,
#         'age3': pre_age_SEIR_pf_Rt3,
#         'age4': pre_age_SEIR_pf_Rt4,
#         'age5': pre_age_SEIR_pf_Rt5,
#         'age6': pre_age_SEIR_pf_Rt6,
#     })

# Ps_Rt = pd.DataFrame({
#         'age1': pre_age_SEIR_ps_Rt1,  
#         'age2': pre_age_SEIR_ps_Rt2,
#         'age3': pre_age_SEIR_ps_Rt3,
#         'age4': pre_age_SEIR_ps_Rt4,
#         'age5': pre_age_SEIR_ps_Rt5,
#         'age6': pre_age_SEIR_ps_Rt6,
#     })

# Gorji_Rt = pd.DataFrame({
#         'age1': pre_Gorji_Rt1,  
#         'age2': pre_Gorji_Rt2,
#         'age3': pre_Gorji_Rt3,
#         'age4': pre_Gorji_Rt4,
#         'age5': pre_Gorji_Rt5,
#         'age6': pre_Gorji_Rt6,
#     })

# pre_confirm.to_csv('./Pre_defined/data/pre_confirm.csv', index=False)
# pre_age_confirm.to_csv('./Pre_defined/data/pre_age_confirm.csv', index=False)
# pre_Rt.to_csv('./Pre_defined/data/pre_Rt.csv', index=False)
# Ps_Rt.to_csv('./Pre_defined/data/Ps_Rt.csv', index=False)
# Pf_Rt.to_csv('./Pre_defined/data/Pf_Rt.csv', index=False)
# Gorji_Rt.to_csv('./Pre_defined/data/Gorji_Rt.csv', index=False)


# end = time.time()
# print("\n")
# print("Computation time = " + str(end - start) + " seconds")  
# print("\n")

# # Plotting
# # # Pre-defined Rt
# # whole_pre_Rt_plotting(pre_Gorji_Rt1, pre_age_SEIR_pf_Rt1, pre_age_SEIR_ps_Rt1, pre_Rt1, "0-6y Age1")
# # whole_pre_Rt_plotting(pre_Gorji_Rt2, pre_age_SEIR_pf_Rt2, pre_age_SEIR_ps_Rt2, pre_Rt2, "7-12y Age2")
# # whole_pre_Rt_plotting(pre_Gorji_Rt3, pre_age_SEIR_pf_Rt3, pre_age_SEIR_ps_Rt3, pre_Rt3, "13-18y Age3")
# # whole_pre_Rt_plotting(pre_Gorji_Rt4, pre_age_SEIR_pf_Rt4, pre_age_SEIR_ps_Rt4, pre_Rt4, "19-49y Age4")
# # whole_pre_Rt_plotting(pre_Gorji_Rt5, pre_age_SEIR_pf_Rt5, pre_age_SEIR_ps_Rt5, pre_Rt5, "50-64y Age5")
# # whole_pre_Rt_plotting(pre_Gorji_Rt6, pre_age_SEIR_pf_Rt6, pre_age_SEIR_ps_Rt6, pre_Rt6, "65+y Age6")

# # part_pre_Rt_plotting(pre_Gorji_Rt1, pre_age_SEIR_pf_Rt1, pre_age_SEIR_ps_Rt1, pre_Rt1, "0-6y Age1")
# # part_pre_Rt_plotting(pre_Gorji_Rt2, pre_age_SEIR_pf_Rt2, pre_age_SEIR_ps_Rt2, pre_Rt2, "7-12y Age2")
# # part_pre_Rt_plotting(pre_Gorji_Rt3, pre_age_SEIR_pf_Rt3, pre_age_SEIR_ps_Rt3, pre_Rt3, "13-18y Age3")
# # part_pre_Rt_plotting(pre_Gorji_Rt4, pre_age_SEIR_pf_Rt4, pre_age_SEIR_ps_Rt4, pre_Rt4, "19-49y Age4")
# # part_pre_Rt_plotting(pre_Gorji_Rt5, pre_age_SEIR_pf_Rt5, pre_age_SEIR_ps_Rt5, pre_Rt5, "50-64y Age5")
# # part_pre_Rt_plotting(pre_Gorji_Rt6, pre_age_SEIR_pf_Rt6, pre_age_SEIR_ps_Rt6, pre_Rt6, "65+y Age6")

# # pre_Rt_age_plotting(pre_age_SEIR_pf_Rt1, pre_age_SEIR_pf_Rt2, pre_age_SEIR_pf_Rt3, pre_age_SEIR_pf_Rt4, pre_age_SEIR_pf_Rt5, pre_age_SEIR_pf_Rt6, "Particle Filter")
# # pre_Rt_age_plotting(pre_age_SEIR_ps_Rt1, pre_age_SEIR_ps_Rt2, pre_age_SEIR_ps_Rt3, pre_age_SEIR_ps_Rt4, pre_age_SEIR_ps_Rt5, pre_age_SEIR_ps_Rt6, "Particle Smoother")
# # pre_Rt_age_plotting(pre_Gorji_Rt1, pre_Gorji_Rt2, pre_Gorji_Rt3, pre_Gorji_Rt4, pre_Gorji_Rt5, pre_Gorji_Rt6, "Gorji Method")
