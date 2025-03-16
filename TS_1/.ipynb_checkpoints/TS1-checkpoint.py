# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:28:36 2025

@author: Tobi
"""

import numpy as np
import matplotlib.pyplot as plt

def funcion_senoidal (Vmax, dc, ff, ph, nn, fs):
    
    Ts = 1/fs # Calculo período de muestreo
    t = np.linspace(0,(nn-1)*Ts,nn)
    x = Vmax * np.sin(2 * np.pi * ff * t + ph) + dc
    return[t,x]


N = 1000
Fs = 1000
tt,xx = funcion_senoidal(1, 0, 10, 0, N, Fs)

plt.plot(tt,xx)
plt.title('Señal: Senoidal')
plt.xlabel('tiempo [s]')
plt.ylabel('Amplitud [V]')
