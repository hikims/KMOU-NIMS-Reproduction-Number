"""
Particle filter / smoother implementation for age-specific Rt estimation.

This file is a GitHub-friendly drop-in replacement of the original script:
- avoids hard-coded absolute paths
- translates remaining Korean comments to English

Environment
-----------
Set INFLUENZA_PROJECT_ROOT to the project directory, or place required data files under ./data.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import time
from scipy.stats import poisson, nbinom 
from scipy.optimize import nnls
import os
from pathlib import Path

# Project root (override via environment variable if needed)
PROJECT_ROOT = Path(os.environ.get('INFLUENZA_PROJECT_ROOT', Path(__file__).resolve().parent)).resolve()
PATH = str(PROJECT_ROOT)


dt = 1.0
scale = 5.0
sigma = 1.0 / 5.0

# particle 
sigma_rw = 0.3
cases = 10**6
n = 30
random_rate = 1.05

fn = PATH + '/data/Pops_Dec2022.csv'
with open(fn, encoding='EUC-KR') as file:
    pop_raw = pd.read_csv(file)

pops = [int(s.replace(",", "")) for s in pop_raw.iloc[0, 3:]]

lbeag = [0, 7, 13, 19, 50, 65]
ubeag = [6, 12, 18, 49, 64, 100]  
pops_agg = [sum(pops[lbeag[i]:ubeag[i]+1]) for i in range(6)]


N1 = pops_agg[0]
N2 = pops_agg[1]
N3 = pops_agg[2]
N4 = pops_agg[3]
N5 = pops_agg[4]
N6 = pops_agg[5]

m11 = 1.76 
m12 = 0.21 
m13 = 0.03 
m14 = 0.2 
m15 = 0.05 
m16 = 0.05  

m21 = 0.33 
m22 = 3.75 
m23 = 0.3 
m24 = 0.31 
m25 = 0.14 
m26 = 0.13

m31 = 0.05 
m32 = 0.31 
m33 = 3.65 
m34 = 0.22 
m35 = 0.31 
m36 = 0.12 

m41 = 2.07 
m42 = 2.11 
m43 = 1.38 
m44 = 1.78 
m45 = 1.26 
m46 = 0.84   

m51 = 0.47 
m52 = 0.88
m53 = 1.87 
m54 = 1.18 
m55 = 1.97 
m56 = 1.47  

m61 = 0.3 
m62 = 0.47 
m63 = 0.42 
m64 = 0.45 
m65 = 0.83 
m66 = 2.5  


# 4th order Runge-Kutta
def RK4(S1, I1, R1, b1, S2, I2, R2, b2, S3, I3, R3, b3, S4, I4, R4, b4, S5, I5, R5, b5, S6, I6, R6, b6):

    # 1st order
    sum1_1 = I1 * m11 + I2 * m12 + I3 * m13 + I4 * m14 + I5 * m15 + I6 * m16
    KS1_1 = -b1 * (S1 / N1) * sum1_1 
    KI1_1 = b1 * (S1 / N1) * sum1_1 - sigma * I1
    KR1_1 = sigma * I1 

    sum1_2 = I1 * m21 + I2 * m22 + I3 * m23 + I4 * m24 + I5 * m25 + I6 * m26
    KS1_2 = -b2 * (S2 / N2) * sum1_2
    KI1_2 = b2 * (S2 / N2) * sum1_2 - sigma * I2
    KR1_2 = sigma * I2 

    sum1_3 = I1 * m31 + I2 * m32 + I3 * m33 + I4 * m34 + I5 * m35 + I6 * m36
    KS1_3 = -b3 * (S3 / N3) * sum1_3
    KI1_3 = b3 * (S3 / N3) * sum1_3 - sigma * I3
    KR1_3 = sigma * I3 

    sum1_4 = I1 * m41 + I2 * m42 + I3 * m43 + I4 * m44 + I5 * m45 + I6 * m46
    KS1_4 = -b4 * (S4 / N4) * sum1_4
    KI1_4 = b4 * (S4 / N4) * sum1_4 - sigma * I4
    KR1_4 = sigma * I4 

    sum1_5 = I1 * m51 + I2 * m52 + I3 * m53 + I4 * m54 + I5 * m55 + I6 * m56
    KS1_5 = -b5 * (S5 / N5) * sum1_5
    KI1_5 = b5 * (S5 / N5) * sum1_5 - sigma * I5
    KR1_5 = sigma * I5 

    sum1_6 = I1 * m61 + I2 * m62 + I3 * m63 + I4 * m64 + I5 * m65 + I6 * m66
    KS1_6 = -b6 * (S6 / N6) * sum1_6
    KI1_6 = b6 * (S6 / N6) * sum1_6 - sigma * I6
    KR1_6 = sigma * I6 

    # 2nd order
    sum2_1 = (I1 + KI1_1 * 0.5 * dt) * m11 + (I2 + KI1_2 * 0.5 * dt) * m12 + (I3 + KI1_3 * 0.5 * dt) * m13 + (I4 + KI1_4 * 0.5 * dt) * m14 + (I5 + KI1_5 * 0.5 * dt) * m15 + (I6 + KI1_6 * 0.5 * dt) * m16
    KS2_1 = -b1 * ((S1 + KS1_1 * 0.5 * dt) / N1) * sum2_1
    KI2_1 = b1 * ((S1 + KS1_1 * 0.5 * dt) / N1) * sum2_1 - sigma * (I1 + KI1_1 * 0.5 * dt)
    KR2_1 = sigma * (I1 + KI1_1 * 0.5 * dt)

    sum2_2 = (I1 + KI1_1 * 0.5 * dt) * m21 + (I2 + KI1_2 * 0.5 * dt) * m22 + (I3 + KI1_3 * 0.5 * dt) * m23 + (I4 + KI1_4 * 0.5 * dt) * m24 + (I5 + KI1_5 * 0.5 * dt) * m25 + (I6 + KI1_6 * 0.5 * dt) * m26
    KS2_2 = -b2 * ((S2 + KS1_2 * 0.5 * dt) / N2) * sum2_2
    KI2_2 = b2 * ((S2 + KS1_2 * 0.5 * dt) / N2) * sum2_2 - sigma * (I2 + KI1_2 * 0.5 * dt)
    KR2_2 = sigma * (I2 + KI1_2 * 0.5 * dt)

    sum2_3 = (I1 + KI1_1 * 0.5 * dt) * m31 + (I2 + KI1_2 * 0.5 * dt) * m32 + (I3 + KI1_3 * 0.5 * dt) * m33 + (I4 + KI1_4 * 0.5 * dt) * m34 + (I5 + KI1_5 * 0.5 * dt) * m35 + (I6 + KI1_6 * 0.5 * dt) * m36
    KS2_3 = -b3 * ((S3 + KS1_3 * 0.5 * dt) / N3) * sum2_3
    KI2_3 = b3 * ((S3 + KS1_3 * 0.5 * dt) / N3) * sum2_3 - sigma * (I3 + KI1_3 * 0.5 * dt)
    KR2_3 = sigma * (I3 + KI1_3 * 0.5 * dt)

    sum2_4 = (I1 + KI1_1 * 0.5 * dt) * m41 + (I2 + KI1_2 * 0.5 * dt) * m42 + (I3 + KI1_3 * 0.5 * dt) * m43 + (I4 + KI1_4 * 0.5 * dt) * m44 + (I5 + KI1_5 * 0.5 * dt) * m45 + (I6 + KI1_6 * 0.5 * dt) * m46
    KS2_4 = -b4 * ((S4 + KS1_4 * 0.5 * dt) / N4) * sum2_4
    KI2_4 = b4 * ((S4 + KS1_4 * 0.5 * dt) / N4) * sum2_4 - sigma * (I4 + KI1_4 * 0.5 * dt)
    KR2_4 = sigma * (I4 + KI1_4 * 0.5 * dt)

    sum2_5 = (I1 + KI1_1 * 0.5 * dt) * m51 + (I2 + KI1_2 * 0.5 * dt) * m52 + (I3 + KI1_3 * 0.5 * dt) * m53 + (I4 + KI1_4 * 0.5 * dt) * m54 + (I5 + KI1_5 * 0.5 * dt) * m55 + (I6 + KI1_6 * 0.5 * dt) * m56
    KS2_5 = -b5 * ((S5 + KS1_5 * 0.5 * dt) / N5) * sum2_5
    KI2_5 = b5 * ((S5 + KS1_5 * 0.5 * dt) / N5) * sum2_5 - sigma * (I5 + KI1_5 * 0.5 * dt)
    KR2_5 = sigma * (I5 + KI1_5 * 0.5 * dt)

    sum2_6 = (I1 + KI1_1 * 0.5 * dt) * m61 + (I2 + KI1_2 * 0.5 * dt) * m62 + (I3 + KI1_3 * 0.5 * dt) * m63 + (I4 + KI1_4 * 0.5 * dt) * m64 + (I5 + KI1_5 * 0.5 * dt) * m65 + (I6 + KI1_6 * 0.5 * dt) * m66
    KS2_6 = -b6 * ((S6 + KS1_6 * 0.5 * dt) / N6) * sum2_6
    KI2_6 = b6 * ((S6 + KS1_6 * 0.5 * dt) / N6) * sum2_6 - sigma * (I6 + KI1_6 * 0.5 * dt)
    KR2_6 = sigma * (I6 + KI1_6 * 0.5 * dt)

    # 3rd order
    sum3_1 = (I1 + KI2_1 * 0.5 * dt) * m11 + (I2 + KI2_2 * 0.5 * dt) * m12 + (I3 + KI2_3 * 0.5 * dt) * m13 + (I4 + KI2_4 * 0.5 * dt) * m14 + (I5 + KI2_5 * 0.5 * dt) * m15 + (I6 + KI2_6 * 0.5 * dt) * m16
    KS3_1 = -b1 * ((S1 + KS2_1 * 0.5 * dt) / N1) * sum3_1
    KI3_1 = b1 * ((S1 + KS2_1 * 0.5 * dt) / N1) * sum3_1 - sigma * (I1 + KI2_1 * 0.5 * dt)
    KR3_1 = sigma * (I1 + KI2_1 * 0.5 * dt)

    sum3_2 = (I1 + KI2_1 * 0.5 * dt) * m21 + (I2 + KI2_2 * 0.5 * dt) * m22 + (I3 + KI2_3 * 0.5 * dt) * m23 + (I4 + KI2_4 * 0.5 * dt) * m24 + (I5 + KI2_5 * 0.5 * dt) * m25 + (I6 + KI2_6 * 0.5 * dt) * m26
    KS3_2 = -b2 * ((S2 + KS2_2 * 0.5 * dt) / N2) * sum3_2
    KI3_2 = b2 * ((S2 + KS2_2 * 0.5 * dt) / N2) * sum3_2 - sigma * (I2 + KI2_2 * 0.5 * dt)
    KR3_2 = sigma * (I2 + KI2_2 * 0.5 * dt)

    sum3_3 = (I1 + KI2_1 * 0.5 * dt) * m31 + (I2 + KI2_2 * 0.5 * dt) * m32 + (I3 + KI2_3 * 0.5 * dt) * m33 + (I4 + KI2_4 * 0.5 * dt) * m34 + (I5 + KI2_5 * 0.5 * dt) * m35 + (I6 + KI2_6 * 0.5 * dt) * m36
    KS3_3 = -b3 * ((S3 + KS2_3 * 0.5 * dt) / N3) * sum3_3
    KI3_3 = b3 * ((S3 + KS2_3 * 0.5 * dt) / N3) * sum3_3 - sigma * (I3 + KI2_3 * 0.5 * dt)
    KR3_3 = sigma * (I3 + KI2_3 * 0.5 * dt)

    sum3_4 = (I1 + KI2_1 * 0.5 * dt) * m41 + (I2 + KI2_2 * 0.5 * dt) * m42 + (I3 + KI2_3 * 0.5 * dt) * m43 + (I4 + KI2_4 * 0.5 * dt) * m44 + (I5 + KI2_5 * 0.5 * dt) * m45 + (I6 + KI2_6 * 0.5 * dt) * m46
    KS3_4 = -b4 * ((S4 + KS2_4 * 0.5 * dt) / N4) * sum3_4
    KI3_4 = b4 * ((S4 + KS2_4 * 0.5 * dt) / N4) * sum3_4 - sigma * (I4 + KI2_4 * 0.5 * dt)
    KR3_4 = sigma * (I4 + KI2_4 * 0.5 * dt)

    sum3_5 = (I1 + KI2_1 * 0.5 * dt) * m51 + (I2 + KI2_2 * 0.5 * dt) * m52 + (I3 + KI2_3 * 0.5 * dt) * m53 + (I4 + KI2_4 * 0.5 * dt) * m54 + (I5 + KI2_5 * 0.5 * dt) * m55 + (I6 + KI2_6 * 0.5 * dt) * m56
    KS3_5 = -b5 * ((S5 + KS2_5 * 0.5 * dt) / N5) * sum3_5
    KI3_5 = b5 * ((S5 + KS2_5 * 0.5 * dt) / N5) * sum3_5 - sigma * (I5 + KI2_5 * 0.5 * dt)
    KR3_5 = sigma * (I5 + KI2_5 * 0.5 * dt)

    sum3_6 = (I1 + KI2_1 * 0.5 * dt) * m61 + (I2 + KI2_2 * 0.5 * dt) * m62 + (I3 + KI2_3 * 0.5 * dt) * m63 + (I4 + KI2_4 * 0.5 * dt) * m64 + (I5 + KI2_5 * 0.5 * dt) * m65 + (I6 + KI2_6 * 0.5 * dt) * m66
    KS3_6 = -b6 * ((S6 + KS2_6 * 0.5 * dt) / N6) * sum3_6
    KI3_6 = b6 * ((S6 + KS2_6 * 0.5 * dt) / N6) * sum3_6 - sigma * (I6 + KI2_6 * 0.5 * dt)
    KR3_6 = sigma * (I6 + KI2_6 * 0.5 * dt)

    # 4nd order
    sum4_1 = (I1 + KI3_1 * dt) * m11 + (I2 + KI3_2 * dt) * m12 + (I3 + KI3_3 * dt) * m13 + (I4 + KI3_4 * dt) * m14 + (I5 + KI3_5 * dt) * m15 + (I6 + KI3_6 * dt) * m16
    KS4_1 = -b1 * ((S1 + KS3_1 * dt) / N1) * sum4_1
    KI4_1 = b1 * ((S1 + KS3_1 * dt) / N1) * sum4_1 - sigma * (I1 + KI3_1 * dt)
    KR4_1 = sigma * (I1 + KI3_1 * dt)

    sum4_2 = (I1 + KI3_1 * dt) * m21 + (I2 + KI3_2 * dt) * m22 + (I3 + KI3_3 * dt) * m23 + (I4 + KI3_4 * dt) * m24 + (I5 + KI3_5 * dt) * m25 + (I6 + KI3_6 * dt) * m26
    KS4_2 = -b2 * ((S2 + KS3_2 * dt) / N2) * sum4_2
    KI4_2 = b2 * ((S2 + KS3_2 * dt) / N2) * sum4_2 - sigma * (I2 + KI3_2 * dt)
    KR4_2 = sigma * (I2 + KI3_2 * dt)

    sum4_3 = (I1 + KI3_1 * dt) * m31 + (I2 + KI3_2 * dt) * m32 + (I3 + KI3_3 * dt) * m33 + (I4 + KI3_4 * dt) * m34 + (I5 + KI3_5 * dt) * m35 + (I6 + KI3_6 * dt) * m36
    KS4_3 = -b3 * ((S3 + KS3_3 * dt) / N3) * sum4_3
    KI4_3 = b3 * ((S3 + KS3_3 * dt) / N3) * sum4_3 - sigma * (I3 + KI3_3 * dt)
    KR4_3 = sigma * (I3 + KI3_3 * dt)

    sum4_4 = (I1 + KI3_1 * dt) * m41 + (I2 + KI3_2 * dt) * m42 + (I3 + KI3_3 * dt) * m43 + (I4 + KI3_4 * dt) * m44 + (I5 + KI3_5 * dt) * m45 + (I6 + KI3_6 * dt) * m46
    KS4_4 = -b4 * ((S4 + KS3_4 * dt) / N4) * sum4_4
    KI4_4 = b4 * ((S4 + KS3_4 * dt) / N4) * sum4_4 - sigma * (I4 + KI3_4 * dt)
    KR4_4 = sigma * (I4 + KI3_4 * dt)

    sum4_5 = (I1 + KI3_1 * dt) * m51 + (I2 + KI3_2 * dt) * m52 + (I3 + KI3_3 * dt) * m53 + (I4 + KI3_4 * dt) * m54 + (I5 + KI3_5 * dt) * m55 + (I6 + KI3_6 * dt) * m56
    KS4_5 = -b5 * ((S5 + KS3_5 * dt) / N5) * sum4_5
    KI4_5 = b5 * ((S5 + KS3_5 * dt) / N5) * sum4_5 - sigma * (I5 + KI3_5 * dt)
    KR4_5 = sigma * (I5 + KI3_5 * dt)

    sum4_6 = (I1 + KI3_1 * dt) * m61 + (I2 + KI3_2 * dt) * m62 + (I3 + KI3_3 * dt) * m63 + (I4 + KI3_4 * dt) * m64 + (I5 + KI3_5 * dt) * m65 + (I6 + KI3_6 * dt) * m66
    KS4_6 = -b6 * ((S6 + KS3_6 * dt) / N6) * sum4_6
    KI4_6 = b6 * ((S6 + KS3_6 * dt) / N6) * sum4_6 - sigma * (I6 + KI3_6 * dt)
    KR4_6 = sigma * (I6 + KI3_6 * dt)  

    S1 = S1 + (KS1_1 + 2.0 * KS2_1 + 2.0 * KS3_1 + KS4_1) / 6.0
    I1 = I1 + (KI1_1 + 2.0 * KI2_1 + 2.0 * KI3_1 + KI4_1) / 6.0
    R1 = R1 + (KR1_1 + 2.0 * KR2_1 + 2.0 * KR3_1 + KR4_1) / 6.0
    confirm1 = (KR1_1 + 2.0 * KR2_1 + 2.0 * KR3_1 + KR4_1) / 6.0

    S2 = S2 + (KS1_2 + 2.0 * KS2_2 + 2.0 * KS3_2 + KS4_2) / 6.0
    I2 = I2 + (KI1_2 + 2.0 * KI2_2 + 2.0 * KI3_2 + KI4_2) / 6.0
    R2 = R2 + (KR1_2 + 2.0 * KR2_2 + 2.0 * KR3_2 + KR4_2) / 6.0
    confirm2 = (KR1_2 + 2.0 * KR2_2 + 2.0 * KR3_2 + KR4_2) / 6.0

    S3 = S3 + (KS1_3 + 2.0 * KS2_3 + 2.0 * KS3_3 + KS4_3) / 6.0
    I3 = I3 + (KI1_3 + 2.0 * KI2_3 + 2.0 * KI3_3 + KI4_3) / 6.0
    R3 = R3 + (KR1_3 + 2.0 * KR2_3 + 2.0 * KR3_3 + KR4_3) / 6.0
    confirm3 = (KR1_3 + 2.0 * KR2_3 + 2.0 * KR3_3 + KR4_3) / 6.0

    S4 = S4 + (KS1_4 + 2.0 * KS2_4 + 2.0 * KS3_4 + KS4_4) / 6.0
    I4 = I4 + (KI1_4 + 2.0 * KI2_4 + 2.0 * KI3_4 + KI4_4) / 6.0
    R4 = R4 + (KR1_4 + 2.0 * KR2_4 + 2.0 * KR3_4 + KR4_4) / 6.0
    confirm4 = (KR1_4 + 2.0 * KR2_4 + 2.0 * KR3_4 + KR4_4) / 6.0

    S5 = S5 + (KS1_5 + 2.0 * KS2_5 + 2.0 * KS3_5 + KS4_5) / 6.0
    I5 = I5 + (KI1_5 + 2.0 * KI2_5 + 2.0 * KI3_5 + KI4_5) / 6.0
    R5 = R5 + (KR1_5 + 2.0 * KR2_5 + 2.0 * KR3_5 + KR4_5) / 6.0
    confirm5 = (KR1_5 + 2.0 * KR2_5 + 2.0 * KR3_5 + KR4_5) / 6.0

    S6 = S6 + (KS1_6 + 2.0 * KS2_6 + 2.0 * KS3_6 + KS4_6) / 6.0
    I6 = I6 + (KI1_6 + 2.0 * KI2_6 + 2.0 * KI3_6 + KI4_6) / 6.0
    R6 = R6 + (KR1_6 + 2.0 * KR2_6 + 2.0 * KR3_6 + KR4_6) / 6.0
    confirm6 = (KR1_6 + 2.0 * KR2_6 + 2.0 * KR3_6 + KR4_6) / 6.0

    return np.array([S1, I1, R1, S2, I2, R2, S3, I3, R3, S4, I4, R4, S5, I5, R5, S6, I6, R6, confirm1, confirm2, confirm3, confirm4, confirm5, confirm6])


def ps_age_Rt(date_leng, pre_confirm1, pre_confirm2, pre_confirm3, pre_confirm4, pre_confirm5, pre_confirm6, sentence):
    print("\n")  
    print(f"Start Rt for {sentence}") 
    
    pre_col = 6

    np.random.seed()

    lambda_i1 = scale * 119.327
    lambda_i2 = scale * 90.5102
    lambda_i3 = scale * 37.6122
    lambda_i4 = scale * 135.469
    lambda_i5 = scale * 60.4898
    lambda_i6 = scale * 44.6327
    
    Pre_S1 = np.zeros(date_leng + 1)
    Pre_I1 = np.zeros(date_leng + 1)
    Pre_R1 = np.zeros(date_leng + 1)

    Pre_S2 = np.zeros(date_leng + 1)
    Pre_I2 = np.zeros(date_leng + 1)
    Pre_R2 = np.zeros(date_leng + 1)

    Pre_S3 = np.zeros(date_leng + 1)
    Pre_I3 = np.zeros(date_leng + 1)
    Pre_R3 = np.zeros(date_leng + 1)

    Pre_S4 = np.zeros(date_leng + 1)
    Pre_I4 = np.zeros(date_leng + 1)
    Pre_R4 = np.zeros(date_leng + 1)

    Pre_S5 = np.zeros(date_leng + 1)
    Pre_I5 = np.zeros(date_leng + 1)
    Pre_R5 = np.zeros(date_leng + 1)

    Pre_S6 = np.zeros(date_leng + 1)
    Pre_I6 = np.zeros(date_leng + 1)
    Pre_R6 = np.zeros(date_leng + 1)
    
    Pre_b1 = np.zeros([date_leng + 1, cases])
    Pre_b2 = np.zeros([date_leng + 1, cases])
    Pre_b3 = np.zeros([date_leng + 1, cases])
    Pre_b4 = np.zeros([date_leng + 1, cases])
    Pre_b5 = np.zeros([date_leng + 1, cases])
    Pre_b6 = np.zeros([date_leng + 1, cases])

    # Initial condition
    Pre_I1[0] = lambda_i1
    Pre_R1[0] = 0.0
    Pre_S1[0] = N1 - Pre_I1[0] - Pre_R1[0]

    Pre_I2[0] = lambda_i2
    Pre_R2[0] = 0.0
    Pre_S2[0] = N2 - Pre_I2[0] - Pre_R2[0]

    Pre_I3[0] = lambda_i3
    Pre_R3[0] = 0.0
    Pre_S3[0] = N3 - Pre_I3[0] - Pre_R3[0]

    Pre_I4[0] = lambda_i4
    Pre_R4[0] = 0.0
    Pre_S4[0] = N4 - Pre_I4[0] - Pre_R4[0]

    Pre_I5[0] = lambda_i5
    Pre_R5[0] = 0.0
    Pre_S5[0] = N5 - Pre_I5[0] - Pre_R5[0]    

    Pre_I6[0] = lambda_i6
    Pre_R6[0] = 0.0
    Pre_S6[0] = N6 - Pre_I6[0] - Pre_R6[0]
    
    Pf_S1 = np.zeros([date_leng + 1, cases])
    Pf_I1 = np.zeros([date_leng + 1, cases])
    Pf_R1 = np.zeros([date_leng + 1, cases])
    Pf_b1 = np.zeros([date_leng + 1, cases])

    rv_index1 = np.zeros([date_leng + 1, cases], dtype = int)
    Pf_confirm1 = np.zeros([date_leng, cases])
    confirm1 = np.zeros(date_leng)
    
    Pf_S2 = np.zeros([date_leng + 1, cases])
    Pf_I2 = np.zeros([date_leng + 1, cases])
    Pf_R2 = np.zeros([date_leng + 1, cases])
    Pf_b2 = np.zeros([date_leng + 1, cases])

    rv_index2 = np.zeros([date_leng + 1, cases], dtype = int)
    Pf_confirm2 = np.zeros([date_leng, cases])
    confirm2 = np.zeros(date_leng)
    
    Pf_S3 = np.zeros([date_leng + 1, cases])
    Pf_I3 = np.zeros([date_leng + 1, cases])
    Pf_R3 = np.zeros([date_leng + 1, cases])
    Pf_b3 = np.zeros([date_leng + 1, cases])

    rv_index3 = np.zeros([date_leng + 1, cases], dtype = int)
    Pf_confirm3 = np.zeros([date_leng, cases])
    confirm3 = np.zeros(date_leng)
    
    Pf_S4 = np.zeros([date_leng + 1, cases])
    Pf_I4 = np.zeros([date_leng + 1, cases])
    Pf_R4 = np.zeros([date_leng + 1, cases])
    Pf_b4 = np.zeros([date_leng + 1, cases])

    rv_index4 = np.zeros([date_leng + 1, cases], dtype = int)
    Pf_confirm4 = np.zeros([date_leng, cases])
    confirm4 = np.zeros(date_leng)
    
    Pf_S5 = np.zeros([date_leng + 1, cases])
    Pf_I5 = np.zeros([date_leng + 1, cases])
    Pf_R5 = np.zeros([date_leng + 1, cases])
    Pf_b5 = np.zeros([date_leng + 1, cases])

    rv_index5 = np.zeros([date_leng + 1, cases], dtype = int)
    Pf_confirm5 = np.zeros([date_leng, cases])
    confirm5 = np.zeros(date_leng)
    
    Pf_S6 = np.zeros([date_leng + 1, cases])
    Pf_I6 = np.zeros([date_leng + 1, cases])
    Pf_R6 = np.zeros([date_leng + 1, cases])
    Pf_b6 = np.zeros([date_leng + 1, cases])

    rv_index6 = np.zeros([date_leng + 1, cases], dtype = int)
    Pf_confirm6 = np.zeros([date_leng, cases])
    confirm6 = np.zeros(date_leng)
    
    
    Age1 = np.zeros(date_leng)
    Age2 = np.zeros(date_leng)
    Age3 = np.zeros(date_leng)
    Age4 = np.zeros(date_leng)
    Age5 = np.zeros(date_leng)
    Age6 = np.zeros(date_leng)

    for i in range(date_leng):
        Age1[i] = pre_confirm1[i]
        Age2[i] = pre_confirm2[i]
        Age3[i] = pre_confirm3[i]
        Age4[i] = pre_confirm4[i]
        Age5[i] = pre_confirm5[i]
        Age6[i] = pre_confirm6[i]
    
    print("Random No. of Age1")
    random_numbers1 = np.random.normal(0, sigma_rw, cases * date_leng)
    normal_b1 = np.reshape(random_numbers1, (date_leng, cases))

    print("Random No. of Age2")
    random_numbers2 = np.random.normal(0, sigma_rw, cases * date_leng)
    normal_b2 = np.reshape(random_numbers2, (date_leng, cases))

    print("Random No. of Age3")
    random_numbers3 = np.random.normal(0, sigma_rw, cases * date_leng)
    normal_b3 = np.reshape(random_numbers3, (date_leng, cases))

    print("Random No. of Age4")
    random_numbers4 = np.random.normal(0, sigma_rw, cases * date_leng)
    normal_b4 = np.reshape(random_numbers4, (date_leng, cases))

    print("Random No. of Age5")
    random_numbers5 = np.random.normal(0, sigma_rw, cases * date_leng)
    normal_b5 = np.reshape(random_numbers5, (date_leng, cases))

    print("Random No. of Age6")
    random_numbers6 = np.random.normal(0, sigma_rw, cases * date_leng)
    normal_b6 = np.reshape(random_numbers6, (date_leng, cases))

    Pre_b1[0, :] = random_rate * np.exp(normal_b1[0, :])
    Pre_b2[0, :] = random_rate * np.exp(normal_b2[0, :])
    Pre_b3[0, :] = random_rate * np.exp(normal_b3[0, :])
    Pre_b4[0, :] = random_rate * np.exp(normal_b4[0, :])
    Pre_b5[0, :] = random_rate * np.exp(normal_b5[0, :])
    Pre_b6[0, :] = random_rate * np.exp(normal_b5[0, :])


    Pf_I1[0, :] = np.random.poisson(lambda_i1, size = cases)
    Pf_I2[0, :] = np.random.poisson(lambda_i2, size = cases)
    Pf_I3[0, :] = np.random.poisson(lambda_i3, size = cases)
    Pf_I4[0, :] = np.random.poisson(lambda_i4, size = cases)
    Pf_I5[0, :] = np.random.poisson(lambda_i5, size = cases)
    Pf_I6[0, :] = np.random.poisson(lambda_i6, size = cases)

    Pf_S1[0, :] = N1 - Pf_I1[0, :] - Pf_R1[0, :]
    Pf_S2[0, :] = N2 - Pf_I2[0, :] - Pf_R2[0, :]
    Pf_S3[0, :] = N3 - Pf_I3[0, :] - Pf_R3[0, :]
    Pf_S4[0, :] = N4 - Pf_I4[0, :] - Pf_R4[0, :]
    Pf_S5[0, :] = N5 - Pf_I5[0, :] - Pf_R5[0, :]
    Pf_S6[0, :] = N6 - Pf_I6[0, :] - Pf_R6[0, :]

    A_const = np.zeros((6,6))

    A_const[0, 0] = (1.0 / sigma) * (Pf_S1[0, 0] / N1) * m11 
    A_const[0, 1] = (1.0 / sigma) * (Pf_S2[0, 0] / N2) * m21
    A_const[0, 2] = (1.0 / sigma) * (Pf_S3[0, 0] / N3) * m31
    A_const[0, 3] = (1.0 / sigma) * (Pf_S4[0, 0] / N4) * m41
    A_const[0, 4] = (1.0 / sigma) * (Pf_S5[0, 0] / N5) * m51
    A_const[0, 5] = (1.0 / sigma) * (Pf_S6[0, 0] / N6) * m61

    A_const[1, 0] = (1.0 / sigma) * (Pf_S1[0, 0] / N1) * m12
    A_const[1, 1] = (1.0 / sigma) * (Pf_S2[0, 0] / N2) * m22
    A_const[1, 2] = (1.0 / sigma) * (Pf_S3[0, 0] / N3) * m32
    A_const[1, 3] = (1.0 / sigma) * (Pf_S4[0, 0] / N4) * m42
    A_const[1, 4] = (1.0 / sigma) * (Pf_S5[0, 0] / N5) * m52
    A_const[1, 5] = (1.0 / sigma) * (Pf_S6[0, 0] / N6) * m62

    A_const[2, 0] = (1.0 / sigma) * (Pf_S1[0, 0] / N1) * m13
    A_const[2, 1] = (1.0 / sigma) * (Pf_S2[0, 0] / N2) * m23
    A_const[2, 2] = (1.0 / sigma) * (Pf_S3[0, 0] / N3) * m33
    A_const[2, 3] = (1.0 / sigma) * (Pf_S4[0, 0] / N4) * m43
    A_const[2, 4] = (1.0 / sigma) * (Pf_S5[0, 0] / N5) * m53
    A_const[2, 5] = (1.0 / sigma) * (Pf_S6[0, 0] / N6) * m63

    A_const[3, 0] = (1.0 / sigma) * (Pf_S1[0, 0] / N1) * m14
    A_const[3, 1] = (1.0 / sigma) * (Pf_S2[0, 0] / N2) * m24
    A_const[3, 2] = (1.0 / sigma) * (Pf_S3[0, 0] / N3) * m34
    A_const[3, 3] = (1.0 / sigma) * (Pf_S4[0, 0] / N4) * m44
    A_const[3, 4] = (1.0 / sigma) * (Pf_S5[0, 0] / N5) * m54
    A_const[3, 5] = (1.0 / sigma) * (Pf_S6[0, 0] / N6) * m64

    A_const[4, 0] = (1.0 / sigma) * (Pf_S1[0, 0] / N1) * m15
    A_const[4, 1] = (1.0 / sigma) * (Pf_S2[0, 0] / N2) * m25
    A_const[4, 2] = (1.0 / sigma) * (Pf_S3[0, 0] / N3) * m35
    A_const[4, 3] = (1.0 / sigma) * (Pf_S4[0, 0] / N4) * m45
    A_const[4, 4] = (1.0 / sigma) * (Pf_S5[0, 0] / N5) * m55
    A_const[4, 5] = (1.0 / sigma) * (Pf_S6[0, 0] / N6) * m65

    A_const[5, 0] = (1.0 / sigma) * (Pf_S1[0, 0] / N1) * m16
    A_const[5, 1] = (1.0 / sigma) * (Pf_S2[0, 0] / N2) * m26
    A_const[5, 2] = (1.0 / sigma) * (Pf_S3[0, 0] / N3) * m36
    A_const[5, 3] = (1.0 / sigma) * (Pf_S4[0, 0] / N4) * m46
    A_const[5, 4] = (1.0 / sigma) * (Pf_S5[0, 0] / N5) * m56
    A_const[5, 5] = (1.0 / sigma) * (Pf_S6[0, 0] / N6) * m66

    Pf_b_initial = np.zeros((6, cases))
    for j in range(cases):
        b_vec = np.array([
            Pre_b1[0, j],
            Pre_b2[0, j],
            Pre_b3[0, j],
            Pre_b4[0, j],
            Pre_b5[0, j],
            Pre_b6[0, j]
        ])
        # nnls expects shape (m,n) A and (m,) b
        x_j, rnorm = nnls(A_const, b_vec)
        Pf_b_initial[:, j] = x_j

    Pf_b1[0, :] = Pf_b_initial[0, :]
    Pf_b2[0, :] = Pf_b_initial[1, :]
    Pf_b3[0, :] = Pf_b_initial[2, :]
    Pf_b4[0, :] = Pf_b_initial[3, :]
    Pf_b5[0, :] = Pf_b_initial[4, :]
    Pf_b6[0, :] = Pf_b_initial[5, :]

    # term1 = np.zeros(cases)
    # term2 = np.zeros(cases)
    # term3 = np.zeros(cases)
    # term4 = np.zeros(cases)
    # term5 = np.zeros(cases)
    # term6 = np.zeros(cases)
    
    # b1_term = np.zeros(cases)
    # b2_term = np.zeros(cases)
    # b3_term = np.zeros(cases)
    # b4_term = np.zeros(cases)
    # b5_term = np.zeros(cases)
    # b6_term = np.zeros(cases)
    
    # # Age1
    # print("Initial b of Age1")
    # Pf_S1 = np.zeros([date_leng + 1, cases])
    # Pf_I1 = np.zeros([date_leng + 1, cases])
    # Pf_R1 = np.zeros([date_leng + 1, cases])
    # Pf_b1 = np.zeros([date_leng + 1, cases])

    # rv_index1 = np.zeros([date_leng + 1, cases], dtype = int)
    # Pf_confirm1 = np.zeros([date_leng, cases])
    # confirm1 = np.zeros(date_leng)

    # lambda_i1 = Age1[0]
    # # Pre_b1[0, :] = random_rate * np.exp(normal_b1[0, :])
    # term1[:] = (1 / sigma) * ((Pre_b2[0, :] * Pre_S2[0] / N2) * m21 + (Pre_b3[0, :] * Pre_S3[0] / N3) * m31 + (Pre_b4[0, :] * Pre_S4[0] / N4) * m41 + (Pre_b5[0, :] * Pre_S5[0] / N5) * m51 + (Pre_b6[0, :] * Pre_S6[0] / N6) * m61)
    # b1_term[:] = Pre_b1[0, :] - term1[:]
    # for j in range(cases):
    #     if (b1_term[j] <= 0):
    #        b1_term[j] = Pre_b1[0, j]
    # Pf_I1[0, :] = np.random.poisson(lambda_i1, size = cases)
    # Pf_S1[0, :] = N1 - Pf_I1[0, :] - Pf_R1[0, :]
    # Pf_b1[0, :] = b1_term * sigma * N1 / Pf_S1[0, :]
    # # Pf_b1[0, :] = (term1 * 1.05 * np.exp(normal_b1[0, :]) * sigma * N1 / Pf_S1[0, :]) 

    # # Age2
    # print("Initial b of Age2")
    # Pf_S2 = np.zeros([date_leng + 1, cases])
    # Pf_I2 = np.zeros([date_leng + 1, cases])
    # Pf_R2 = np.zeros([date_leng + 1, cases])
    # Pf_b2 = np.zeros([date_leng + 1, cases])

    # rv_index2 = np.zeros([date_leng + 1, cases], dtype = int)
    # Pf_confirm2 = np.zeros([date_leng, cases])
    # confirm2 = np.zeros(date_leng)

    # lambda_i2 = Age2[0]
    # # Pre_b2[0, :] = random_rate * np.exp(normal_b2[0, :])
    # term2[:] = (1 / sigma) * ((Pre_b1[0, :] * Pre_S1[0] / N1) * m12 + (Pre_b3[0, :] * Pre_S3[0] / N3) * m32 + (Pre_b4[0, :] * Pre_S4[0] / N4) * m42 + (Pre_b5[0, :] * Pre_S5[0] / N5) * m52 + (Pre_b6[0, :] * Pre_S6[0] / N6) * m62)
    # b2_term[:] = Pre_b2[0, :] - term2[:]
    # for j in range(cases):
    #     if (b2_term[j] <= 0):
    #        b2_term[j] = Pre_b2[0, j]     
    # Pf_I2[0, :] = np.random.poisson(lambda_i2, size = cases)
    # Pf_S2[0, :] = N2 - Pf_I2[0, :] - Pf_R2[0, :]  
    # Pf_b2[0, :] = b2_term * sigma * N2 / Pf_S2[0, :]
    # # Pf_b2[0, :] = (term2 * 1.05 * np.exp(normal_b2[0, :]) * sigma * N2 / Pf_S2[0, :]) 

    # # Age3
    # print("Initial b of Age3")
    # Pf_S3 = np.zeros([date_leng + 1, cases])
    # Pf_I3 = np.zeros([date_leng + 1, cases])
    # Pf_R3 = np.zeros([date_leng + 1, cases])
    # Pf_b3 = np.zeros([date_leng + 1, cases])

    # rv_index3 = np.zeros([date_leng + 1, cases], dtype = int)
    # Pf_confirm3 = np.zeros([date_leng, cases])
    # confirm3 = np.zeros(date_leng)

    # lambda_i3 = Age3[0]
    # # Pre_b3[0, :] = random_rate * np.exp(normal_b3[0, :])
    # term3[:] = (1 / sigma) * ((Pre_b1[0, :] * Pre_S1[0] / N1) * m13 + (Pre_b2[0, :] * Pre_S2[0] / N2) * m23 + (Pre_b4[0, :] * Pre_S4[0] / N4) * m43 + (Pre_b5[0, :] * Pre_S5[0] / N5) * m53 + (Pre_b6[0, :] * Pre_S6[0] / N6) * m63)
    # b3_term[:] = Pre_b3[0, :] - term3[:]
    # for j in range(cases):
    #     if (b3_term[j] <= 0):
    #        b3_term[j] = Pre_b3[0, j]
    # Pf_I3[0, :] = np.random.poisson(lambda_i3, size = cases)
    # Pf_S3[0, :] = N3 - Pf_I3[0, :] - Pf_R3[0, :]   
    # Pf_b3[0, :] = b3_term * sigma * N3 / Pf_S3[0, :]
    # # Pf_b3[0, :] = (term3 * 1.05 * np.exp(normal_b3[0, :]) * sigma * N3 / Pf_S3[0, :]) 
    
    # # Age4
    # print("Initial b of Age4")
    # Pf_S4 = np.zeros([date_leng + 1, cases])
    # Pf_I4 = np.zeros([date_leng + 1, cases])
    # Pf_R4 = np.zeros([date_leng + 1, cases])
    # Pf_b4 = np.zeros([date_leng + 1, cases])

    # rv_index4 = np.zeros([date_leng + 1, cases], dtype = int)
    # Pf_confirm4 = np.zeros([date_leng, cases])
    # confirm4 = np.zeros(date_leng)

    # lambda_i4 = Age4[0]
    # # Pre_b4[0, :] = random_rate * np.exp(normal_b4[0, :])
    # term4[:] = (1 / sigma) * ((Pre_b1[0, :] * Pre_S1[0] / N1) * m14 + (Pre_b2[0, :] * Pre_S2[0] / N2) * m24 + (Pre_b3[0, :] * Pre_S3[0] / N3) * m34 + (Pre_b5[0, :] * Pre_S5[0] / N5) * m54 + (Pre_b6[0, :] * Pre_S6[0] / N6) * m64)
    # b4_term[:] = Pre_b4[0, :] - term4[:]
    # for j in range(cases):
    #     if (b4_term[j] <= 0):
    #        b4_term[j] = Pre_b4[0, j]     
    # Pf_I4[0, :] = np.random.poisson(lambda_i4, size = cases)
    # Pf_S4[0, :] = N4 - Pf_I4[0, :] - Pf_R4[0, :]  
    # Pf_b4[0, :] = b4_term * sigma * N4 / Pf_S4[0, :]
    # # Pf_b4[0, :] = (term4 * 1.05 * np.exp(normal_b4[0, :]) * sigma * N4 / Pf_S4[0, :])  

    # # Age5
    # print("Initial b of Age5")
    # Pf_S5 = np.zeros([date_leng + 1, cases])
    # Pf_I5 = np.zeros([date_leng + 1, cases])
    # Pf_R5 = np.zeros([date_leng + 1, cases])
    # Pf_b5 = np.zeros([date_leng + 1, cases])

    # rv_index5 = np.zeros([date_leng + 1, cases], dtype = int)
    # Pf_confirm5 = np.zeros([date_leng, cases])
    # confirm5 = np.zeros(date_leng)

    # lambda_i5 = Age5[0]
    # # Pre_b5[0, :] = random_rate * np.exp(normal_b5[0, :])
    # term5[:] = (1 / sigma) * ((Pre_b1[0, :] * Pre_S1[0] / N1) * m15 + (Pre_b2[0, :] * Pre_S2[0] / N2) * m25 + (Pre_b3[0, :] * Pre_S3[0] / N3) * m35 + (Pre_b4[0, :] * Pre_S4[0] / N4) * m45 + (Pre_b6[0, :] * Pre_S6[0] / N6) * m65)
    # b5_term[:] = Pre_b5[0, :] - term5[:]
    # for j in range(cases):
    #     if (b5_term[j] <= 0):
    #        b5_term[j] = Pre_b5[0, j]
    # Pf_I5[0, :] = np.random.poisson(lambda_i5, size = cases)
    # Pf_S5[0, :] = N5 - Pf_I5[0, :] - Pf_R5[0, :]
    # Pf_b5[0, :] = b5_term * sigma * N5 / Pf_S5[0, :]
    # # Pf_b5[0, :] = (term5 * 1.05 * np.exp(normal_b5[0, :]) * sigma * N5 / Pf_S5[0, :]) 

    # # Age6
    # print("Initial b of Age6")
    # Pf_S6 = np.zeros([date_leng + 1, cases])
    # Pf_I6 = np.zeros([date_leng + 1, cases])
    # Pf_R6 = np.zeros([date_leng + 1, cases])
    # Pf_b6 = np.zeros([date_leng + 1, cases])

    # rv_index6 = np.zeros([date_leng + 1, cases], dtype = int)
    # Pf_confirm6 = np.zeros([date_leng, cases])
    # confirm6 = np.zeros(date_leng)

    # lambda_i6 = Age6[0]
    # # Pre_b6[0, :] = random_rate * np.exp(normal_b6[0, :])
    # term6[:] = (1 / sigma) * ((Pre_b1[0, :] * Pre_S1[0] / N1) * m16 + (Pre_b2[0, :] * Pre_S2[0] / N2) * m26 + (Pre_b3[0, :] * Pre_S3[0] / N3) * m36 + (Pre_b4[0, :] * Pre_S4[0] / N4) * m46 + (Pre_b5[0, :] * Pre_S5[0] / N6) * m56)
    # b6_term[:] = Pre_b6[0, :] - term6[:]
    # for j in range(cases):
    #     if (b6_term[j] <= 0):
    #        b6_term[j] = Pre_b6[0, j]
    # Pf_I6[0, :] = np.random.poisson(lambda_i6, size = cases)
    # Pf_S6[0, :] = N6 - Pf_I6[0, :] - Pf_R6[0, :]
    # Pf_b6[0, :] = b6_term * sigma * N6 / Pf_S6[0, :]
    # # Pf_b6[0, :] = (term6 * 1.05 * np.exp(normal_b6[0, :]) * sigma * N6 / Pf_S6[0, :]) 


    # time evolution
    for k in range(date_leng):
        print("Calculation:", k)
        Pf_b1[k + 1, :] = Pf_b1[k, :] * np.exp(normal_b1[k, :])
        Pf_b2[k + 1, :] = Pf_b2[k, :] * np.exp(normal_b2[k, :])
        Pf_b3[k + 1, :] = Pf_b3[k, :] * np.exp(normal_b3[k, :])
        Pf_b4[k + 1, :] = Pf_b4[k, :] * np.exp(normal_b4[k, :])
        Pf_b5[k + 1, :] = Pf_b5[k, :] * np.exp(normal_b5[k, :])
        Pf_b6[k + 1, :] = Pf_b6[k, :] * np.exp(normal_b6[k, :])
        
    
        Pf_results = RK4(Pf_S1[k, :], Pf_I1[k, :], Pf_R1[k, :], Pf_b1[k + 1, :], 
                        Pf_S2[k, :], Pf_I2[k, :], Pf_R2[k, :], Pf_b2[k + 1, :], 
                        Pf_S3[k, :], Pf_I3[k, :], Pf_R3[k, :], Pf_b3[k + 1, :], 
                        Pf_S4[k, :], Pf_I4[k, :], Pf_R4[k, :], Pf_b4[k + 1, :],
                        Pf_S5[k, :], Pf_I5[k, :], Pf_R5[k, :], Pf_b5[k + 1, :], 
                        Pf_S6[k, :], Pf_I6[k, :], Pf_R6[k, :], Pf_b6[k + 1, :])
        [Pf_S1[k + 1, :], Pf_I1[k + 1, :], Pf_R1[k + 1, :],
        Pf_S2[k + 1, :], Pf_I2[k + 1, :], Pf_R2[k + 1, :],
        Pf_S3[k + 1, :], Pf_I3[k + 1, :], Pf_R3[k + 1, :],
        Pf_S4[k + 1, :], Pf_I4[k + 1, :], Pf_R4[k + 1, :],
        Pf_S5[k + 1, :], Pf_I5[k + 1, :], Pf_R5[k + 1, :],
        Pf_S6[k + 1, :], Pf_I6[k + 1, :], Pf_R6[k + 1, :]] = Pf_results[0:18]
        [Pf_confirm1[k, :], 
        Pf_confirm2[k, :], 
        Pf_confirm3[k, :], 
        Pf_confirm4[k, :], 
        Pf_confirm5[k, :], 
        Pf_confirm6[k, :]] = Pf_results[18:24]

        #weight1 = poisson.pmf(round(Age1[k]), mu = Pf_confirm1[k, :])
        weight1 = nbinom.pmf(round(Age1[k]), n, n / (Pf_confirm1[k, :] + n))
        if np.any(np.isnan(weight1)):
            print("NaN has been observed in Age1!!!")
        weight1 = weight1/sum(weight1)
        randomList1 = np.random.choice(np.arange(cases), size = cases, replace = True, p = weight1)
        rv_index1[k + 1, :] = randomList1 
    
        #weight2 = poisson.pmf(round(Age2[k]), mu = Pf_confirm2[k, :])
        weight2 = nbinom.pmf(round(Age2[k]), n, n / (Pf_confirm2[k, :] + n)) 
        if np.any(np.isnan(weight2)):
            print("NaN has been observed in Age2!!!")
        weight2 = weight2/sum(weight2)
        randomList2 = np.random.choice(np.arange(cases), size = cases, replace = True, p = weight2)
        rv_index2[k + 1, :] = randomList2 
    
        #weight3 = poisson.pmf(round(Age3[k]), mu = Pf_confirm3[k, :])
        weight3 = nbinom.pmf(round(Age3[k]), n, n / (Pf_confirm3[k, :] + n))
        if np.any(np.isnan(weight3)):
            print("NaN has been observed in Age3!!!")
        weight3 = weight3/sum(weight3)
        randomList3 = np.random.choice(np.arange(cases), size = cases, replace = True, p = weight3)
        rv_index3[k + 1, :] = randomList3 
    
        #weight4 = poisson.pmf(round(Age4[k]), mu = Pf_confirm4[k, :])
        weight4 = nbinom.pmf(round(Age4[k]), n, n / (Pf_confirm4[k, :] + n)) 
        if np.any(np.isnan(weight4)):
            print("NaN has been observed in Age4!!!")
        weight4 = weight4/sum(weight4)
        randomList4 = np.random.choice(np.arange(cases), size = cases, replace = True, p = weight4)
        rv_index4[k + 1, :] = randomList4
    
        #weight5 = poisson.pmf(round(Age5[k]), mu = Pf_confirm5[k, :])
        weight5 = nbinom.pmf(round(Age5[k]), n, n / (Pf_confirm5[k, :] + n)) 
        if np.any(np.isnan(weight5)):
            print("NaN has been observed in Age5!!!")
        weight5 = weight5/sum(weight5)
        randomList5 = np.random.choice(np.arange(cases), size = cases, replace = True, p = weight5)
        rv_index5[k + 1, :] = randomList5
    
        #weight6 = poisson.pmf(round(Age6[k]), mu = Pf_confirm6[k, :]) 
        weight6 = nbinom.pmf(round(Age6[k]), n, n / (Pf_confirm6[k, :] + n))
        if np.any(np.isnan(weight6)):
            print("NaN has been observed in Age6!!!")
        weight6 = weight6/sum(weight6)
        randomList6 = np.random.choice(np.arange(cases), size = cases, replace = True, p = weight6)
        rv_index6[k + 1, :] = randomList6

        # # weight1 = poisson.pmf(round(Age1[k]), mu = Pf_confirm1[k, :])
        # weight1 = nbinom.pmf(round(Age1[k]), n, n / (Pf_confirm1[k, :] + n))
        # weight1 = np.nan_to_num(weight1, nan=0.0, posinf=0.0, neginf=0.0)
        # weight1 = np.clip(weight1, 0.0, None)
        # s = weight1.sum()
        # if s == 0 or not np.isfinite(s):
        #     p = np.ones_like(weight1) / weight1.size
        # else:
        #     p = weight1 / s
        # randomList1 = np.random.choice(np.arange(cases), size=cases, replace=True, p=p)
        # rv_index1[k + 1, :] = randomList1
    
        # # weight2 = poisson.pmf(round(Age2[k]), mu = Pf_confirm2[k, :])
        # weight2 = nbinom.pmf(round(Age2[k]), n, n / (Pf_confirm2[k, :] + n)) 
        # weight2 = np.nan_to_num(weight2, nan=0.0, posinf=0.0, neginf=0.0)
        # weight2 = np.clip(weight2, 0.0, None)
        # s = weight2.sum()
        # if s == 0 or not np.isfinite(s):
        #     p = np.ones_like(weight2) / weight2.size
        # else:
        #     p = weight2 / s
        # randomList2 = np.random.choice(np.arange(cases), size=cases, replace=True, p=p)
        # rv_index2[k + 1, :] = randomList2
    
        # # weight3 = poisson.pmf(round(Age3[k]), mu = Pf_confirm3[k, :])
        # weight3 = nbinom.pmf(round(Age3[k]), n, n / (Pf_confirm3[k, :] + n))
        # weight3 = np.nan_to_num(weight3, nan=0.0, posinf=0.0, neginf=0.0)
        # weight3 = np.clip(weight3, 0.0, None)
        # s = weight3.sum()
        # if s == 0 or not np.isfinite(s):
        #     p = np.ones_like(weight3) / weight3.size
        # else:
        #     p = weight3 / s
        # randomList3 = np.random.choice(np.arange(cases), size=cases, replace=True, p=p)
        # rv_index3[k + 1, :] = randomList3
    
        # # weight4 = poisson.pmf(round(Age4[k]), mu = Pf_confirm4[k, :])
        # weight4 = nbinom.pmf(round(Age4[k]), n, n / (Pf_confirm4[k, :] + n)) 
        # weight4 = np.nan_to_num(weight4, nan=0.0, posinf=0.0, neginf=0.0)
        # weight4 = np.clip(weight4, 0.0, None)
        # s = weight4.sum()
        # if s == 0 or not np.isfinite(s):
        #     p = np.ones_like(weight4) / weight4.size
        # else:
        #     p = weight4 / s
        # randomList4 = np.random.choice(np.arange(cases), size=cases, replace=True, p=p)
        # rv_index4[k + 1, :] = randomList4
    
        # # weight5 = poisson.pmf(round(Age5[k]), mu = Pf_confirm5[k, :])
        # weight5 = nbinom.pmf(round(Age5[k]), n, n / (Pf_confirm5[k, :] + n)) 
        # weight5 = np.nan_to_num(weight5, nan=0.0, posinf=0.0, neginf=0.0)
        # weight5 = np.clip(weight5, 0.0, None)
        # s = weight5.sum()
        # if s == 0 or not np.isfinite(s):
        #     p = np.ones_like(weight5) / weight5.size
        # else:
        #     p = weight5 / s
        # randomList5 = np.random.choice(np.arange(cases), size=cases, replace=True, p=p)
        # rv_index5[k + 1, :] = randomList5
    
        # # weight6 = poisson.pmf(round(Age6[k]), mu = Pf_confirm6[k, :]) 
        # weight6 = nbinom.pmf(round(Age6[k]), n, n / (Pf_confirm6[k, :] + n))
        # weight6 = np.nan_to_num(weight6, nan=0.0, posinf=0.0, neginf=0.0)
        # weight6 = np.clip(weight6, 0.0, None)
        # s = weight6.sum()
        # if s == 0 or not np.isfinite(s):
        #     p = np.ones_like(weight6) / weight6.size
        # else:
        #     p = weight6 / s
        # randomList6 = np.random.choice(np.arange(cases), size=cases, replace=True, p=p)
        # rv_index6[k + 1, :] = randomList6

        Pf_S1[k + 1, :] = Pf_S1[k + 1, randomList1]
        Pf_I1[k + 1, :] = Pf_I1[k + 1, randomList1]
        Pf_R1[k + 1, :] = Pf_R1[k + 1, randomList1]
        Pf_b1[k + 1, :] = Pf_b1[k + 1, randomList1]

        Pf_S2[k + 1, :] = Pf_S2[k + 1, randomList2]
        Pf_I2[k + 1, :] = Pf_I2[k + 1, randomList2]
        Pf_R2[k + 1, :] = Pf_R2[k + 1, randomList2]
        Pf_b2[k + 1, :] = Pf_b2[k + 1, randomList2]

        Pf_S3[k + 1, :] = Pf_S3[k + 1, randomList3]
        Pf_I3[k + 1, :] = Pf_I3[k + 1, randomList3]
        Pf_R3[k + 1, :] = Pf_R3[k + 1, randomList3]
        Pf_b3[k + 1, :] = Pf_b3[k + 1, randomList3]

        Pf_S4[k + 1, :] = Pf_S4[k + 1, randomList4]
        Pf_I4[k + 1, :] = Pf_I4[k + 1, randomList4]
        Pf_R4[k + 1, :] = Pf_R4[k + 1, randomList4]
        Pf_b4[k + 1, :] = Pf_b4[k + 1, randomList4]

        Pf_S5[k + 1, :] = Pf_S5[k + 1, randomList5]
        Pf_I5[k + 1, :] = Pf_I5[k + 1, randomList5]
        Pf_R5[k + 1, :] = Pf_R5[k + 1, randomList5]
        Pf_b5[k + 1, :] = Pf_b5[k + 1, randomList5]

        Pf_S6[k + 1, :] = Pf_S6[k + 1, randomList6]
        Pf_I6[k + 1, :] = Pf_I6[k + 1, randomList6]
        Pf_R6[k + 1, :] = Pf_R6[k + 1, randomList6]
        Pf_b6[k + 1, :] = Pf_b6[k + 1, randomList6]

    # Confirmation and reproduction number
    confirm1 = np.mean(Pf_confirm1[0:, :], axis = 1)
    confirm2 = np.mean(Pf_confirm2[0:, :], axis = 1)
    confirm3 = np.mean(Pf_confirm3[0:, :], axis = 1)
    confirm4 = np.mean(Pf_confirm4[0:, :], axis = 1)
    confirm5 = np.mean(Pf_confirm5[0:, :], axis = 1)
    confirm6 = np.mean(Pf_confirm6[0:, :], axis = 1)

    Pf_Rt1 = np.zeros(date_leng)
    Pf_Rt2 = np.zeros(date_leng)
    Pf_Rt3 = np.zeros(date_leng)
    Pf_Rt4 = np.zeros(date_leng)
    Pf_Rt5 = np.zeros(date_leng)
    Pf_Rt6 = np.zeros(date_leng)
    
    term_Pf_Rt1 = np.zeros([date_leng, cases])
    term_Pf_Rt2 = np.zeros([date_leng, cases])
    term_Pf_Rt3 = np.zeros([date_leng, cases])
    term_Pf_Rt4 = np.zeros([date_leng, cases])
    term_Pf_Rt5 = np.zeros([date_leng, cases])
    term_Pf_Rt6 = np.zeros([date_leng, cases])

    for t in range(1,date_leng):
        term_Pf_Rt1[t, :] = (1 / sigma) * ((Pf_b1[t, :] * Pf_S1[t, :] / N1) * m11 + (Pf_b2[t, :] * Pf_S2[t, :] / N2) * m21 + (Pf_b3[t, :] * Pf_S3[t, :] / N3) * m31 + (Pf_b4[t, :] * Pf_S4[t, :] / N4) * m41 + (Pf_b5[t, :] * Pf_S5[t, :] / N5) * m51 + (Pf_b6[t, :] * Pf_S6[t, :] / N6) * m61)
        term_Pf_Rt2[t, :] = (1 / sigma) * ((Pf_b1[t, :] * Pf_S1[t, :] / N1) * m12 + (Pf_b2[t, :] * Pf_S2[t, :] / N2) * m22 + (Pf_b3[t, :] * Pf_S3[t, :] / N3) * m32 + (Pf_b4[t, :] * Pf_S4[t, :] / N4) * m42 + (Pf_b5[t, :] * Pf_S5[t, :] / N5) * m52 + (Pf_b6[t, :] * Pf_S6[t, :] / N6) * m62)
        term_Pf_Rt3[t, :] = (1 / sigma) * ((Pf_b1[t, :] * Pf_S1[t, :] / N1) * m13 + (Pf_b2[t, :] * Pf_S2[t, :] / N2) * m23 + (Pf_b3[t, :] * Pf_S3[t, :] / N3) * m33 + (Pf_b4[t, :] * Pf_S4[t, :] / N4) * m43 + (Pf_b5[t, :] * Pf_S5[t, :] / N5) * m53 + (Pf_b6[t, :] * Pf_S6[t, :] / N6) * m63)
        term_Pf_Rt4[t, :] = (1 / sigma) * ((Pf_b1[t, :] * Pf_S1[t, :] / N1) * m14 + (Pf_b2[t, :] * Pf_S2[t, :] / N2) * m24 + (Pf_b3[t, :] * Pf_S3[t, :] / N3) * m34 + (Pf_b4[t, :] * Pf_S4[t, :] / N4) * m44 + (Pf_b5[t, :] * Pf_S5[t, :] / N5) * m54 + (Pf_b6[t, :] * Pf_S6[t, :] / N6) * m64)
        term_Pf_Rt5[t, :] = (1 / sigma) * ((Pf_b1[t, :] * Pf_S1[t, :] / N1) * m15 + (Pf_b2[t, :] * Pf_S2[t, :] / N2) * m25 + (Pf_b3[t, :] * Pf_S3[t, :] / N3) * m35 + (Pf_b4[t, :] * Pf_S4[t, :] / N4) * m45 + (Pf_b5[t, :] * Pf_S5[t, :] / N5) * m55 + (Pf_b6[t, :] * Pf_S6[t, :] / N6) * m65)
        term_Pf_Rt6[t, :] = (1 / sigma) * ((Pf_b1[t, :] * Pf_S1[t, :] / N1) * m16 + (Pf_b2[t, :] * Pf_S2[t, :] / N2) * m26 + (Pf_b3[t, :] * Pf_S3[t, :] / N3) * m36 + (Pf_b4[t, :] * Pf_S4[t, :] / N4) * m46 + (Pf_b5[t, :] * Pf_S5[t, :] / N5) * m56 + (Pf_b6[t, :] * Pf_S6[t, :] / N6) * m66)
       
        Pf_Rt1[t] = np.mean(term_Pf_Rt1[t, :])
        Pf_Rt2[t] = np.mean(term_Pf_Rt2[t, :])
        Pf_Rt3[t] = np.mean(term_Pf_Rt3[t, :])
        Pf_Rt4[t] = np.mean(term_Pf_Rt4[t, :])
        Pf_Rt5[t] = np.mean(term_Pf_Rt5[t, :])
        Pf_Rt6[t] = np.mean(term_Pf_Rt6[t, :])
        
        # Pf_Rt1[t] = (1 / sigma) * (np.mean(Pf_b1[t, :]) * (np.mean(Pf_S1[t, :]) / N1) * m11 + np.mean(Pf_b2[t, :]) * (np.mean(Pf_S2[t, :]) / N2) * m21 + np.mean(Pf_b3[t, :]) * (np.mean(Pf_S3[t, :]) / N3) * m31 + np.mean(Pf_b4[t, :]) * (np.mean(Pf_S4[t, :]) / N4) * m41 + np.mean(Pf_b5[t, :]) * (np.mean(Pf_S5[t, :]) / N5) * m51 + np.mean(Pf_b6[t, :]) * (np.mean(Pf_S6[t, :]) / N6) * m61)
        # Pf_Rt2[t] = (1 / sigma) * (np.mean(Pf_b1[t, :]) * (np.mean(Pf_S1[t, :]) / N1) * m12 + np.mean(Pf_b2[t, :]) * (np.mean(Pf_S2[t, :]) / N2) * m22 + np.mean(Pf_b3[t, :]) * (np.mean(Pf_S3[t, :]) / N3) * m32 + np.mean(Pf_b4[t, :]) * (np.mean(Pf_S4[t, :]) / N4) * m42 + np.mean(Pf_b5[t, :]) * (np.mean(Pf_S5[t, :]) / N5) * m52 + np.mean(Pf_b6[t, :]) * (np.mean(Pf_S6[t, :]) / N6) * m62)
        # Pf_Rt3[t] = (1 / sigma) * (np.mean(Pf_b1[t, :]) * (np.mean(Pf_S1[t, :]) / N1) * m13 + np.mean(Pf_b2[t, :]) * (np.mean(Pf_S2[t, :]) / N2) * m23 + np.mean(Pf_b3[t, :]) * (np.mean(Pf_S3[t, :]) / N3) * m33 + np.mean(Pf_b4[t, :]) * (np.mean(Pf_S4[t, :]) / N4) * m43 + np.mean(Pf_b5[t, :]) * (np.mean(Pf_S5[t, :]) / N5) * m53 + np.mean(Pf_b6[t, :]) * (np.mean(Pf_S6[t, :]) / N6) * m63)
        # Pf_Rt4[t] = (1 / sigma) * (np.mean(Pf_b1[t, :]) * (np.mean(Pf_S1[t, :]) / N1) * m14 + np.mean(Pf_b2[t, :]) * (np.mean(Pf_S2[t, :]) / N2) * m24 + np.mean(Pf_b3[t, :]) * (np.mean(Pf_S3[t, :]) / N3) * m34 + np.mean(Pf_b4[t, :]) * (np.mean(Pf_S4[t, :]) / N4) * m44 + np.mean(Pf_b5[t, :]) * (np.mean(Pf_S5[t, :]) / N5) * m54 + np.mean(Pf_b6[t, :]) * (np.mean(Pf_S6[t, :]) / N6) * m64)
        # Pf_Rt5[t] = (1 / sigma) * (np.mean(Pf_b1[t, :]) * (np.mean(Pf_S1[t, :]) / N1) * m15 + np.mean(Pf_b2[t, :]) * (np.mean(Pf_S2[t, :]) / N2) * m25 + np.mean(Pf_b3[t, :]) * (np.mean(Pf_S3[t, :]) / N3) * m35 + np.mean(Pf_b4[t, :]) * (np.mean(Pf_S4[t, :]) / N4) * m45 + np.mean(Pf_b5[t, :]) * (np.mean(Pf_S5[t, :]) / N5) * m55 + np.mean(Pf_b6[t, :]) * (np.mean(Pf_S6[t, :]) / N6) * m65)
        # Pf_Rt6[t] = (1 / sigma) * (np.mean(Pf_b1[t, :]) * (np.mean(Pf_S1[t, :]) / N1) * m16 + np.mean(Pf_b2[t, :]) * (np.mean(Pf_S2[t, :]) / N2) * m26 + np.mean(Pf_b3[t, :]) * (np.mean(Pf_S3[t, :]) / N3) * m36 + np.mean(Pf_b4[t, :]) * (np.mean(Pf_S4[t, :]) / N4) * m46 + np.mean(Pf_b5[t, :]) * (np.mean(Pf_S5[t, :]) / N5) * m56 + np.mean(Pf_b6[t, :]) * (np.mean(Pf_S6[t, :]) / N6) * m66)

    # Particle smoother Rt 
    # Age1
    Ps_b1 = np.zeros([date_leng + 1, cases])

    # t = date_leng
    Ps_sample1 = np.arange(cases)
    Ps_b1[date_leng, :] = Pf_b1[date_leng, Ps_sample1]
    
    # t = date_leng - 1
    Ps_sample1 = rv_index1[date_leng, Ps_sample1]
    Ps_b1[date_leng - 1, :] = Pf_b1[date_leng - 1, Ps_sample1]

    for k in range(date_leng - 2, -1, -1):
        Ps_sample1 = rv_index1[k + 1, Ps_sample1]
        Ps_b1[k, :] = Pf_b1[k, Ps_sample1]

    # Age2
    Ps_b2 = np.zeros([date_leng + 1, cases])

    # t = date_leng
    Ps_sample2 = np.arange(cases)
    Ps_b2[date_leng, :] = Pf_b2[date_leng, Ps_sample2]
    
    # t = date_leng - 1
    Ps_sample2 = rv_index2[date_leng, Ps_sample2]
    Ps_b2[date_leng - 1, :] = Pf_b2[date_leng - 1, Ps_sample2]

    for k in range(date_leng - 2, -1, -1):
        Ps_sample2 = rv_index2[k + 1, Ps_sample2]
        Ps_b2[k, :] = Pf_b2[k, Ps_sample2]

    # Age3
    Ps_b3 = np.zeros([date_leng + 1, cases])

    # t = date_leng
    Ps_sample3 = np.arange(cases)
    Ps_b3[date_leng, :] = Pf_b3[date_leng, Ps_sample3]
    
    # t = date_leng - 1
    Ps_sample3 = rv_index3[date_leng, Ps_sample3]
    Ps_b3[date_leng - 1, :] = Pf_b3[date_leng - 1, Ps_sample3]

    for k in range(date_leng - 2, -1, -1):
        Ps_sample3 = rv_index3[k + 1, Ps_sample3]
        Ps_b3[k, :] = Pf_b3[k, Ps_sample3]

    # Age4
    Ps_b4 = np.zeros([date_leng + 1, cases])

    # t = date_leng
    Ps_sample4 = np.arange(cases)
    Ps_b4[date_leng, :] = Pf_b4[date_leng, Ps_sample4]
    
    # t = date_leng - 1
    Ps_sample4 = rv_index4[date_leng, Ps_sample4]
    Ps_b4[date_leng - 1, :] = Pf_b4[date_leng - 1, Ps_sample4]

    for k in range(date_leng - 2, -1, -1):
        Ps_sample4 = rv_index4[k + 1, Ps_sample4]
        Ps_b4[k, :] = Pf_b4[k, Ps_sample4]
    
    # Age5
    Ps_b5 = np.zeros([date_leng + 1, cases])

    # t = date_leng
    Ps_sample5 = np.arange(cases)
    Ps_b5[date_leng, :] = Pf_b5[date_leng, Ps_sample5]
    
    # t = date_leng - 1
    Ps_sample5 = rv_index5[date_leng, Ps_sample5]
    Ps_b5[date_leng - 1, :] = Pf_b5[date_leng - 1, Ps_sample5]

    for k in range(date_leng - 2, -1, -1):
        Ps_sample5 = rv_index5[k + 1, Ps_sample5]
        Ps_b5[k, :] = Pf_b5[k, Ps_sample5]

    # Age6
    Ps_b6 = np.zeros([date_leng + 1, cases])

    # t = date_leng
    Ps_sample6 = np.arange(cases)
    Ps_b6[date_leng, :] = Pf_b6[date_leng, Ps_sample6]
    
    # t = date_leng - 1
    Ps_sample6 = rv_index6[date_leng, Ps_sample6]
    Ps_b6[date_leng - 1, :] = Pf_b6[date_leng - 1, Ps_sample6]

    for k in range(date_leng - 2, -1, -1):
        Ps_sample6 = rv_index6[k + 1, Ps_sample6]
        Ps_b6[k, :] = Pf_b6[k, Ps_sample6]

    Ps_Rt1 = np.zeros(date_leng)
    Ps_Rt2 = np.zeros(date_leng)
    Ps_Rt3 = np.zeros(date_leng)
    Ps_Rt4 = np.zeros(date_leng)
    Ps_Rt5 = np.zeros(date_leng)
    Ps_Rt6 = np.zeros(date_leng)

    term_Ps_Rt1 = np.zeros([date_leng, cases])
    term_Ps_Rt2 = np.zeros([date_leng, cases])
    term_Ps_Rt3 = np.zeros([date_leng, cases])
    term_Ps_Rt4 = np.zeros([date_leng, cases])
    term_Ps_Rt5 = np.zeros([date_leng, cases])
    term_Ps_Rt6 = np.zeros([date_leng, cases])

    for t in range(1,date_leng):
        term_Ps_Rt1[t, :] = (1 / sigma) * ((Ps_b1[t, :] * Pf_S1[t, :] / N1) * m11 + (Ps_b2[t, :] * Pf_S2[t, :] / N2) * m21 + (Ps_b3[t, :] * Pf_S3[t, :] / N3) * m31 + (Ps_b4[t, :] * Pf_S4[t, :] / N4) * m41 + (Ps_b5[t, :] * Pf_S5[t, :] / N5) * m51 + (Ps_b6[t, :] * Pf_S6[t, :] / N6) * m61)
        term_Ps_Rt2[t, :] = (1 / sigma) * ((Ps_b1[t, :] * Pf_S1[t, :] / N1) * m12 + (Ps_b2[t, :] * Pf_S2[t, :] / N2) * m22 + (Ps_b3[t, :] * Pf_S3[t, :] / N3) * m32 + (Ps_b4[t, :] * Pf_S4[t, :] / N4) * m42 + (Ps_b5[t, :] * Pf_S5[t, :] / N5) * m52 + (Ps_b6[t, :] * Pf_S6[t, :] / N6) * m62)
        term_Ps_Rt3[t, :] = (1 / sigma) * ((Ps_b1[t, :] * Pf_S1[t, :] / N1) * m13 + (Ps_b2[t, :] * Pf_S2[t, :] / N2) * m23 + (Ps_b3[t, :] * Pf_S3[t, :] / N3) * m33 + (Ps_b4[t, :] * Pf_S4[t, :] / N4) * m43 + (Ps_b5[t, :] * Pf_S5[t, :] / N5) * m53 + (Ps_b6[t, :] * Pf_S6[t, :] / N6) * m63)
        term_Ps_Rt4[t, :] = (1 / sigma) * ((Ps_b1[t, :] * Pf_S1[t, :] / N1) * m14 + (Ps_b2[t, :] * Pf_S2[t, :] / N2) * m24 + (Ps_b3[t, :] * Pf_S3[t, :] / N3) * m34 + (Ps_b4[t, :] * Pf_S4[t, :] / N4) * m44 + (Ps_b5[t, :] * Pf_S5[t, :] / N5) * m54 + (Ps_b6[t, :] * Pf_S6[t, :] / N6) * m64)
        term_Ps_Rt5[t, :] = (1 / sigma) * ((Ps_b1[t, :] * Pf_S1[t, :] / N1) * m15 + (Ps_b2[t, :] * Pf_S2[t, :] / N2) * m25 + (Ps_b3[t, :] * Pf_S3[t, :] / N3) * m35 + (Ps_b4[t, :] * Pf_S4[t, :] / N4) * m45 + (Ps_b5[t, :] * Pf_S5[t, :] / N5) * m55 + (Ps_b6[t, :] * Pf_S6[t, :] / N6) * m65)
        term_Ps_Rt6[t, :] = (1 / sigma) * ((Ps_b1[t, :] * Pf_S1[t, :] / N1) * m16 + (Ps_b2[t, :] * Pf_S2[t, :] / N2) * m26 + (Ps_b3[t, :] * Pf_S3[t, :] / N3) * m36 + (Ps_b4[t, :] * Pf_S4[t, :] / N4) * m46 + (Ps_b5[t, :] * Pf_S5[t, :] / N5) * m56 + (Ps_b6[t, :] * Pf_S6[t, :] / N6) * m66)
       
        Ps_Rt1[t] = np.mean(term_Ps_Rt1[t, :])
        Ps_Rt2[t] = np.mean(term_Ps_Rt2[t, :])
        Ps_Rt3[t] = np.mean(term_Ps_Rt3[t, :])
        Ps_Rt4[t] = np.mean(term_Ps_Rt4[t, :])
        Ps_Rt5[t] = np.mean(term_Ps_Rt5[t, :])
        Ps_Rt6[t] = np.mean(term_Ps_Rt6[t, :])
        
    # # =========================
    # # Particle smoother Rt (block)
    # # =========================
    # block = -1  # 30-step block

    # Ps_b1 = np.zeros((date_leng+1, cases)); Ps_b2 = np.zeros((date_leng+1, cases)); Ps_b3 = np.zeros((date_leng+1, cases))
    # Ps_b4 = np.zeros((date_leng+1, cases)); Ps_b5 = np.zeros((date_leng+1, cases)); Ps_b6 = np.zeros((date_leng+1, cases))

    # # Backtrace each block for age1..6 (kept in the original structure)
    # for start in range(date_leng, 0, -block):
    #     end = max(0, start - block)
    #     # Age1
    #     Ps_idx = np.arange(cases)
    #     for t in range(start, end-1, -1):
    #         Ps_b1[t,:] = Pf_b1[t, Ps_idx]
    #         if t>0: Ps_idx = rv_index1[t, Ps_idx]
    #     # Age2
    #     Ps_idx = np.arange(cases)
    #     for t in range(start, end-1, -1):
    #         Ps_b2[t,:] = Pf_b2[t, Ps_idx]
    #         if t>0: Ps_idx = rv_index2[t, Ps_idx]
    #     # Age3
    #     Ps_idx = np.arange(cases)
    #     for t in range(start, end-1, -1):
    #         Ps_b3[t,:] = Pf_b3[t, Ps_idx]
    #         if t>0: Ps_idx = rv_index3[t, Ps_idx]
    #     # Age4
    #     Ps_idx = np.arange(cases)
    #     for t in range(start, end-1, -1):
    #         Ps_b4[t,:] = Pf_b4[t, Ps_idx]
    #         if t>0: Ps_idx = rv_index4[t, Ps_idx]
    #     # Age5
    #     Ps_idx = np.arange(cases)
    #     for t in range(start, end-1, -1):
    #         Ps_b5[t,:] = Pf_b5[t, Ps_idx]
    #         if t>0: Ps_idx = rv_index5[t, Ps_idx]
    #     # Age6
    #     Ps_idx = np.arange(cases)
    #     for t in range(start, end-1, -1):
    #         Ps_b6[t,:] = Pf_b6[t, Ps_idx]
    #         if t>0: Ps_idx = rv_index6[t, Ps_idx]

    # Ps_Rt1 = np.zeros(date_leng); Ps_Rt2 = np.zeros(date_leng); Ps_Rt3 = np.zeros(date_leng)
    # Ps_Rt4 = np.zeros(date_leng); Ps_Rt5 = np.zeros(date_leng); Ps_Rt6 = np.zeros(date_leng)

    # term_Ps_Rt1 = np.zeros((date_leng, cases)); term_Ps_Rt2 = np.zeros((date_leng, cases))
    # term_Ps_Rt3 = np.zeros((date_leng, cases)); term_Ps_Rt4 = np.zeros((date_leng, cases))
    # term_Ps_Rt5 = np.zeros((date_leng, cases)); term_Ps_Rt6 = np.zeros((date_leng, cases))

    # for t in range(1, date_leng):
    #     term_Ps_Rt1[t,:] = (1/sigma)*((Ps_b1[t,:]*Pf_S1[t,:]/N1)*m11 + (Ps_b2[t,:]*Pf_S2[t,:]/N2)*m21 + (Ps_b3[t,:]*Pf_S3[t,:]/N3)*m31 + (Ps_b4[t,:]*Pf_S4[t,:]/N4)*m41 + (Ps_b5[t,:]*Pf_S5[t,:]/N5)*m51 + (Ps_b6[t,:]*Pf_S6[t,:]/N6)*m61)
    #     term_Ps_Rt2[t,:] = (1/sigma)*((Ps_b1[t,:]*Pf_S1[t,:]/N1)*m12 + (Ps_b2[t,:]*Pf_S2[t,:]/N2)*m22 + (Ps_b3[t,:]*Pf_S3[t,:]/N3)*m32 + (Ps_b4[t,:]*Pf_S4[t,:]/N4)*m42 + (Ps_b5[t,:]*Pf_S5[t,:]/N5)*m52 + (Ps_b6[t,:]*Pf_S6[t,:]/N6)*m62)
    #     term_Ps_Rt3[t,:] = (1/sigma)*((Ps_b1[t,:]*Pf_S1[t,:]/N1)*m13 + (Ps_b2[t,:]*Pf_S2[t,:]/N2)*m23 + (Ps_b3[t,:]*Pf_S3[t,:]/N3)*m33 + (Ps_b4[t,:]*Pf_S4[t,:]/N4)*m43 + (Ps_b5[t,:]*Pf_S5[t,:]/N5)*m53 + (Ps_b6[t,:]*Pf_S6[t,:]/N6)*m63)
    #     term_Ps_Rt4[t,:] = (1/sigma)*((Ps_b1[t,:]*Pf_S1[t,:]/N1)*m14 + (Ps_b2[t,:]*Pf_S2[t,:]/N2)*m24 + (Ps_b3[t,:]*Pf_S3[t,:]/N3)*m34 + (Ps_b4[t,:]*Pf_S4[t,:]/N4)*m44 + (Ps_b5[t,:]*Pf_S5[t,:]/N5)*m54 + (Ps_b6[t,:]*Pf_S6[t,:]/N6)*m64)
    #     term_Ps_Rt5[t,:] = (1/sigma)*((Ps_b1[t,:]*Pf_S1[t,:]/N1)*m15 + (Ps_b2[t,:]*Pf_S2[t,:]/N2)*m25 + (Ps_b3[t,:]*Pf_S3[t,:]/N3)*m35 + (Ps_b4[t,:]*Pf_S4[t,:]/N4)*m45 + (Ps_b5[t,:]*Pf_S5[t,:]/N5)*m55 + (Ps_b6[t,:]*Pf_S6[t,:]/N6)*m65)
    #     term_Ps_Rt6[t,:] = (1/sigma)*((Ps_b1[t,:]*Pf_S1[t,:]/N1)*m16 + (Ps_b2[t,:]*Pf_S2[t,:]/N2)*m26 + (Ps_b3[t,:]*Pf_S3[t,:]/N3)*m36 + (Ps_b4[t,:]*Pf_S4[t,:]/N4)*m46 + (Ps_b5[t,:]*Pf_S5[t,:]/N5)*m56 + (Ps_b6[t,:]*Pf_S6[t,:]/N6)*m66)

    #     Ps_Rt1[t] = term_Ps_Rt1[t,:].mean()
    #     Ps_Rt2[t] = term_Ps_Rt2[t,:].mean()
    #     Ps_Rt3[t] = term_Ps_Rt3[t,:].mean()
    #     Ps_Rt4[t] = term_Ps_Rt4[t,:].mean()
    #     Ps_Rt5[t] = term_Ps_Rt5[t,:].mean()
    #     Ps_Rt6[t] = term_Ps_Rt6[t,:].mean()
        
    print(f"Complete SIR Age Rt for {sentence}")  
    print("\n")
    
    return confirm1, confirm2, confirm3, confirm4, confirm5, confirm6, Pf_Rt1, Pf_Rt2, Pf_Rt3, Pf_Rt4, Pf_Rt5, Pf_Rt6, Ps_Rt1, Ps_Rt2, Ps_Rt3, Ps_Rt4, Ps_Rt5, Ps_Rt6





# import pandas as pd
# import numpy as np
# from scipy.stats import poisson, nbinom, norm
# from scipy.optimize import nnls
# from SIR_calc import SIR_calc

# # Particle filter Calculation
# def nonneg_lsq(A, b):
#     res, _ = nnls(A, b)
#     return res

# def compute_b_with_nnls(A, b_matrix):
#     Np, n_cols = b_matrix.shape
#     b_result = np.zeros((n_cols, Np))
#     for j in range(n_cols):
#         b_result[j], _ = nnls(A, b_matrix[:, j])
#     return b_result.T  

# def wsample(weights, size):
#     return np.random.choice(len(weights), size=size, p=weights)

# def ps_age_Rt(ts_mat, m, N, Np, L, sigma, r, sigma_rw):
#     date_lengs = ts_mat.shape[1]
#     age_group = ts_mat.shape[0]

#     S = np.zeros((age_group, Np, date_lengs))
#     I = np.zeros((age_group, Np))
#     R = np.zeros((age_group, Np))
#     b = np.zeros((age_group, Np, date_lengs))
#     Rt = np.zeros((3, age_group, date_lengs))
#     N_mat = np.repeat(N[:, None], Np, axis=1)

#     for i in range(age_group):
#         I[i] = poisson.rvs(mu=5 * ts_mat[i,0], size=Np)
#     S[:, :, 0] = np.tile(N, (Np,1)).T - I
#     R[:, :] = 0

#     b_input = (1.05 * np.exp(np.random.normal(0, sigma_rw, size=(Np, age_group)))).T
#     b_solution = compute_b_with_nnls(m.T, b_input)

#     n_cols = m.shape[1]

#     for j in range(n_cols):
#         denom = (S[:, j, 0] / N).reshape(-1, 1)  
#         b[:, j, 0] = sigma * b_solution[j, :] / denom 
    
#     b[:, j, 0] = (sigma * b_solution[j, :]) / (S[:, j, 0] / N)[..., np.newaxis]    

#     for t in range(1, date_lengs):
#         b[:, :, t] = b[:, :, t-1] * np.exp(np.random.normal(0, sigma_rw, size=(age_group, Np)))

#         results = SIR_calc(S[:, :, t-1], I[:, :], R[:, :], N, m, b[:, :, t], sigma, Np)
#         S[:, :, t], I, R = results

#         for i in range(age_group):
#             mu = R[i, :] / r
#             p = r / (r + mu)
#             weights = nbinom.pmf(np.round(ts_mat[i, t]), r, p)
#             weights = weights / np.sum(weights)
#             inds = wsample(weights, Np)

#             S[i, :, t] = S[i, inds, t]
#             I[i, :] = I[inds]
#             R[i, :] = R[i, inds]
#             b[i, :, t] = b[i, inds, t]

#     ci = 0.95
#     for t in range(date_lengs):
#         Rt_array = (1 / sigma) * (m.T @ (b[:, :, t] * S[:, :, t] / N_mat))
#         Rt[0, :, t] = np.mean(Rt_array, axis=1)
#         Rt[1, :, t] = np.quantile(Rt_array, (1 - ci)/2, axis=1)
#         Rt[2, :, t] = np.quantile(Rt_array, 1 - (1 - ci)/2, axis=1)

#     return Rt
