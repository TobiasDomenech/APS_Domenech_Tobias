# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 20:12:45 2025

@author: Tobi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift



fs =  1000 # Frecuencia de muestreo (Hz)
N = 1000 # Cantidad de muestras
nro_xp = 200 # Cantidad de realizaciones
SNR = 10 # Relación señal ruido
ts = 1/fs # Tiempo de muestreo
f0 = fs/4
df = fs/N # Resolución espectral
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
w_Hann = signal.windows.hann(N).reshape((N,1))

# Ventaneo señal

sBKH = sr * w_BKH
sflatt = sr * w_Flatt
sbox = sr * w_Box
shann = sr * w_Hann

# Transformo y calculo modulo de la fft

fft_BKH = np.abs(1/N * np.fft.fft(sBKH , N , axis = 0))
fft_flatt = np.abs(1/N * np.fft.fft(sflatt , N , axis = 0))
fft_box = np.abs(1/N * np.fft.fft(sbox , N , axis = 0)) 
fft_hann = np.abs(1/N * np.fft.fft(shann, N, axis = 0))

plt.figure(2)

ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

plt.plot( ff[bfrec], 10* np.log10(2*(fft_BKH[bfrec])**2))
plt.plot( ff[bfrec], 10* np.log10(2*(fft_flatt[bfrec])**2))
plt.plot( ff[bfrec], 10* np.log10(2*(fft_box[bfrec])**2))
plt.plot( ff[bfrec], 10* np.log10(2*(fft_hann[bfrec])**2))

plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.grid(visible='True')
axes_hdl = plt.gca()

# Quiero el estimador de a1, para ello debo tomar la fila 250 (N/4), en cada ventana.

a1_BKH = fft_BKH[250]
a1_flatt = fft_flatt[250]
a1_box = fft_box[250]
a1_hann = fft_hann[250]

# Histograma de los estimadores

plt.figure(3)
bins = 30 

plt.hist(a1_BKH, bins=bins, alpha=0.6, label='Blackman-Harris')
plt.hist(a1_flatt, bins=bins, alpha=0.6, label='Flat Top')
plt.hist(a1_box, bins=bins, alpha=0.6, label='Boxcar')
plt.hist(a1_hann, bins=bins, alpha=0.6, label='Hann')

plt.xlabel('Estimador de $a_1$')
plt.ylabel('Frecuencia')
plt.title('Histogramas del estimador de $a_1$ según la ventana')
plt.legend()
plt.grid(True)
plt.show()

# Calculo los sesgos de los estimadores

sesgo_a1_BKH = np.mean(a1_BKH)-a1
sesgo_a1_flatt = np.mean(a1_flatt)-a1
sesgo_a1_box = np.mean(a1_box)-a1
sesgo_a1_hann = np.mean(a1_hann)-a1


# Calculo varianza de los estimadores

varianza_a1_BKH = np.var(a1_BKH)
varianza_a1_flatt = np.var(a1_flatt)
varianza_a1_box = np.var(a1_box)
varianza_a1_hann = np.var(a1_hann)


#### Busco el estimador de omega 1
N2 = 10000 # Defino para hacer un 0 padding y campiar resolucion espectral

fft_BKH = np.abs(1/N * np.fft.fft(sBKH , n=N2 , axis = 0))
fft_flatt = np.abs(1/N * np.fft.fft(sflatt , n=N2 , axis = 0))
fft_box = np.abs(1/N * np.fft.fft(sbox , n=N2 , axis = 0)) 
fft_hann = np.abs(1/N * np.fft.fft(shann, n=N2, axis = 0))

arg_BKH = np.argmax(fft_BKH[:N2//2, : ], axis = 0)
arg_flatt = np.argmax(fft_flatt[:N2//2, : ], axis = 0)
arg_box = np.argmax(fft_box[:N2//2, : ], axis = 0)
arg_hann = np.argmax(fft_hann[:N2//2, : ], axis = 0)

df = fs/N2

f1_BKH = arg_BKH * df
f1_flatt = arg_flatt * df
f1_box = arg_box * df
f1_hann = arg_hann * df

plt.figure(4)
bins = 30 

plt.hist(f1_BKH, bins=bins, alpha=0.6, label='Blackman-Harris')
plt.hist(f1_flatt, bins=bins, alpha=0.6, label='Flat Top')
plt.hist(f1_box, bins=bins, alpha=0.6, label='Boxcar')
plt.hist(f1_hann, bins=bins, alpha=0.6, label='Hann')

plt.xlabel('Estimador de $f_1$')
plt.ylabel('Frecuencia')
plt.title('Histogramas del estimador de $f_1$ según la ventana')
plt.legend()
plt.grid(True)
plt.show()




