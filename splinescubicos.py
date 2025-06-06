# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:56:38 2025

@author: Tobi
"""

# Implementacion del metodo de filtrado por interpolacion x splines cubicos

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
fs = 1000


qrs = mat_struct['qrs_detections'].flatten()

plt.figure()
plt.plot(ecg_one_lead, label='ECG', linewidth=2,color='blue')
plt.plot(qrs, ecg_one_lead[qrs],'rx', label = 'QRS',)
 
plt.title('ECG ')
plt.ylabel('Amplitud')
plt.xlabel('Muestras') # mentira, es frecuencia
plt.grid()
          
plt.show()

# Desde los qrs, viendo el gráfico, deberé ir 90 muestras hacia atrás

pre = 90 # muestras antes QRS
post = 20 # muestras post QRS
i = 0

segmento=[]
for i in qrs:
    

