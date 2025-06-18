# -*- coding: utf-8 -*-
"""
Created on Thu May 29 16:16:10 2025

@author: Tobi
"""

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

# Gráficos interactivos
#%matplotlib ipympl
# Gráficos estáticos
#%matplotlib inline

from pytc2.sistemas_lineales import plot_plantilla


# Tipo de aproximación.
        
aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip'

fs = 1000
nyq_frec = fs/2
fs_norma = 2
# Diseño filtro: bandpass

fpass = np.array( [1.0, 35] )/nyq_frec 
ripple = 1 # dB alfamax
fstop = np.array( [0.1, 50] )/nyq_frec
attenuation = 40 # dB alfamin

sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype = aprox_name, output= 'sos', fs = fs_norma) 

# Análisis filtro

npoints = 1000

# Uso esto cuando no quiera los puntos equiespaciados como generan los npoints
w_rad = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad = np.append(w_rad,np.linspace(40, nyq_frec, 500, endpoint=True)) / nyq_frec

w, hh = sig.sosfreqz(sos, worN=npoints)
plt.plot(w/np.pi, 20*np.log10(np.abs(hh)+1e-15), label='sos')


plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()
#ax.set_xlim([0, 1])
#ax.set_ylim([-60, 1])

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs_norma)
plt.legend()
plt.show()

# %%

# Uso filtro con ECG

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg = mat_struct['ecg_lead'].flatten()

ecg_filtrado = sig.sosfiltfilt(sos, ecg)

plt.figure()
plt.plot(ecg)
plt.plot(ecg_filtrado)
