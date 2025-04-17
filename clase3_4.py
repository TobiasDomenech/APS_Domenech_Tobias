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



fs =  1000 # Frecuencia de muestreo (Hz)
N = 1000 # Cantidad de muestras
nro_xp = 200 # Cantidad de realizaciones
SNR = 10 # Relación señal ruido
ts = 1/fs # Tiempo de muestreo
f0 = fs/4
df = (2*np.pi)/N # Resolución espectral
a1 = np.sqrt(2)


fr = np.random.uniform(-1/2,1/2,size=(1,nro_xp))

f1 = f0 + fr * df

tt = np.linspace(0, (N-1)*ts, N).reshape((N,1))
tt = np.tile(tt, nro_xp)

s = a1 * np.sin(2*np.pi*tt*f1) # Señal generada

nn = np.random.normal(0,np.sqrt(10**(-1*SNR/10)),N).reshape((N,1)) # Ruido
nn = np.tile(nn,nro_xp)

sr = s + nn # Señal con ruido

# plt.figure(1)

# #plt.plot(tt, srq, lw=1, linestyle='-', color='blue', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$ s_{RQ} = Q_{B,V_F}\{s_R\}$')
# #plt.plot(tt, sr, lw=1, color='green', marker='o', markersize='2', ls='dotted', label='$ s_R = s + n $')
# plt.plot(tt, s, lw=1, color='yellow', ls='--', label='$ s $ (analog)')

# plt.title('Señal muestreada por un ADCV' )
# plt.xlabel('tiempo [segundos]')
# plt.ylabel('Amplitud [V]')
# axes_hdl = plt.gca()
# axes_hdl.legend()
# plt.show()

#Genero ventanas

w_BKH = signal.windows.blackmanharris(N).reshape((N,1))
w_Flatt = signal.windows.flattop(N).reshape((N,1))
w_Box = signal.windows.boxcar(N).reshape((N,1))

# Ventaneo señal

sBKH = sr * w_BKH
sflatt = sr * w_Flatt
sbox = sr * w_Box

# Transformo y calculo modulo de la fft

fft_BKH = np.abs(1/N * np.fft.fft(sBKH,N,axis=0))
fft_flatt = np.abs(1/N * np.fft.fft(sflatt,N,axis=0))
fft_box = np.abs(1/N * np.fft.fft(sbox,N,axis=0)) 

plt.figure(2)

ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

plt.plot( ff[bfrec], 10* np.log10(2*(fft_BKH[bfrec])**2))

plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()








