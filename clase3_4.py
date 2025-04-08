# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 20:12:45 2025

@author: Tobi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
from scipy import signal
from scipy.fft import fft, fftshift



fs =  1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
nro_xp = 10
SNR = 10
ts = 1/fs # tiempo de muestreo
f0 = fs/4
df = (2*np.pi)/N # resolución espectral
a1 = np.sqrt(2)


fr = np.random.uniform(-1/2,1/2,size=(1,nro_xp))

f1 = f0 + fr * df

tt = np.linspace(0, (N-1)*ts, N).reshape((N,1))
tt = np.tile(tt, nro_xp)

s = a1 * np.sin(2*np.pi*tt*f1)

nn = np.random.normal(0,np.sqrt(10**(-1*SNR/10)),N).reshape((N,1))
nn = np.tile(nn,nro_xp)

sr = s + nn

plt.figure(4)

#plt.plot(tt, srq, lw=1, linestyle='-', color='blue', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$ s_{RQ} = Q_{B,V_F}\{s_R\}$')
plt.plot(tt, sr, lw=1, color='green', marker='o', markersize='2', ls='dotted', label='$ s_R = s + n $')
#plt.plot(tt, s, lw=1, color='yellow', ls='--', label='$ s $ (analog)')

plt.title('Señal muestreada por un ADCV' )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()





