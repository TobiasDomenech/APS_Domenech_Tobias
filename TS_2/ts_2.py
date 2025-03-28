# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 12:32:33 2025

@author: Tobi
"""

import numpy as np
import matplotlib.pyplot as plt

w = np.logspace(-5,5,10000) # Vector de frecuencia en escala logarítmica
wo = 1
Q = 1/np.sqrt(2) # Factor de selectividad

def pasa_banda_rlc(w,wo,Q):
    

    s = 1j * w
    
    H = (wo/Q) * s/((wo/Q)*s + wo**2 + s**2) # Función de transferencia del pasabanda 
       
    escallog = 20*np.log10(np.abs(H)) # Módulo de la función de transeferencia en escala logarítmica (resultado en decibeles)
    
    fase = np.angle(H,False) # Fase de la función de transferencia
        
    plt.figure(1)
    plt.title("Respuesta en módulo - Pasa Banda")
    plt.xlabel("w [Rad/s]")
    plt.ylabel("|H|[dB]")
    plt.semilogx(w,escallog)
    plt.grid(True)
    
    plt.figure(2)
    plt.title("Respuesta en Fase - Pasa Banda")
    plt.xlabel("w [Rad/s]")
    plt.ylabel("Fase [Rad]")
    plt.semilogx(w,fase)
    plt.grid(True)
   
    
    return

def pasa_altos_2do_orden (w,wo,Q):
    

    s = 1j * w
    
    H = s**2 /(s**2 + wo**2 + s*(wo/Q)) # Función de transferencia del pasaaltos de segundo orden
    
    escallog = 20*np.log10(np.abs(H))
    
    fase = np.angle(H,False)
    
    plt.figure(3)
    plt.title("Respuesta en módulo - Pasa Altos 2do Orden")
    plt.xlabel("w [Rad/s]")
    plt.ylabel(" |H|[dB]")
    plt.semilogx(w,escallog)
    plt.grid(True)
    
    plt.figure(4)
    plt.title("Respuesta en fase - Pasa Altos 2do Orden")
    plt.xlabel("w [Rad/s]")
    plt.ylabel("Fase [Rad]")
    plt.semilogx(w,fase)
    plt.grid(True)
    
    return

pasa_banda_rlc(w,wo,Q)
pasa_altos_2do_orden(w,wo,Q)
    