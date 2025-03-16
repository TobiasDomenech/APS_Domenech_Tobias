# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:28:36 2025

@author: Tobi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def funcion_senoidal (Vmax, dc, ff, ph, nn, fs):
    
    # Vmax = Amplitud máxima de la senoidal [V]
    # ff = Frecuencia [Hz]
    # fs = Frecuencia de muestreo [Hz]
    # tt = Tiempo [s]
    # ph = Fase [Rads]
    # dc = Valor medio [V]
    # nn = Cantidad de muestras
    
    Ts = 1/fs # Calculo período de muestreo
    tt = np.linspace(0,(nn-1)*Ts,nn)
    xx = Vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    
    plt.figure(1)
    plt.plot(tt,xx)
    plt.title('Señal: Senoidal')
    plt.xlabel('tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.legend(['ff = {:d} Hz'.format(ff)],loc='upper right')
    plt.grid(True)
    
    return

def funcion_triangular(fs, n, ff, vmax, ph, dc):
    
    # vmax = Amplitud máxima de la senoidal [V]
    # ff = Frecuencia [Hz]
    # fs = Frecuencia de muestreo [Hz]
    # t = Tiempo [s]
    # ph = Fase [Rads]
    # dc = Valor medio [V]
    # n = Cantidad de muestras
    
    ts = 1/fs # Calculo período de muestreo
    t = np.linspace(0, (n-1)*ts,n)
    x = vmax * signal.sawtooth(2 * np.pi * t * ff + ph, width=0.5) + dc
    
    plt.figure(2)
    plt.plot(t,x, color='red')
    plt.title('Señal: Triangular')
    plt.xlabel('tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.legend(['ff = {:d} Hz'.format(ff)],loc='upper right')
    plt.grid(True)
    
    return

N = 1000 # Cantidad de muestras a analizar
Fs = 1000
funcion_senoidal(1, 0, 1, 0, N, Fs)
funcion_triangular(Fs, N, 5,1,0,0)

