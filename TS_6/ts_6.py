# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:49:55 2025

@author: Tobi
"""


import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.style.use('default')

def analyze_system(b, a, system_name):
  
    # Crear figura con 1 fila, 3 columnas
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f'Análisis del Sistema {system_name}', y=1.05)
    
    # Calcular respuesta en frecuencia
    w, h = sig.freqz(b, a)
    
    
    # 1. Gráfico de magnitud (escala lineal)
    ax1.plot(w, np.abs(h), 'b', linewidth=2)
    ax1.set_xlabel('ω [rad/muestra]')
    ax1.set_ylabel('Magnitud')
    ax1.set_title('Respuesta en Magnitud')
    ax1.grid(True, alpha=0.3)
    
    # 2. Gráfico de fase (radianes)
    ax2.plot(w, np.unwrap(np.angle(h)), 'r', linewidth=2)
    ax2.set_xlabel('ω [rad/muestra]')
    ax2.set_ylabel('Fase [rad]')
    ax2.set_title('Respuesta en Fase')
    ax2.grid(True, alpha=0.3)
    
    # 3. Diagrama de polos y ceros usando tf2zpk
    zeros, poles, _ = sig.tf2zpk(b, a)
    
    # Cálculo de polos en z=0 (multiplicidad = len(a)-1 - len(poles))
    pole_multiplicity = len(b) - 1 - len(poles)
    
    # Círculo unitario
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='black', alpha=0.5)
    ax3.add_patch(circle)
    
    # Graficar polos y ceros
    ax3.scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none',
               edgecolors='r', s=80, label='Ceros', linewidths=2)
   # Graficar polos múltiples en z=0
    if pole_multiplicity > 0:
        ax3.scatter([0], [0], marker='x', color='b', s=80, 
                   linewidths=2, label=f'Polo en z=0 (multiplicidad {pole_multiplicity})')
    
    ax3.set_xlabel('Re')
    ax3.set_ylabel('Im')
    ax3.set_title('Diagrama de Polos y Ceros')
    ax3.legend(loc='upper left')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.show()

# Definición de los sistemas

a = [1, 1, 1, 1]     # y(n) = x(n-3) + x(n-2) + x(n-1) + x(n)
b = [1, 1, 1, 1, 1]   # y(n) = x(n-4) + ... + x(n)
c = [1, -1]           # y(n) = x(n) - x(n-1)
d = [1, 0, -1]        # y(n) = x(n) - x(n-2)

analyze_system(a, 1, 'a')
analyze_system(b, 1, 'b')
analyze_system(c, 1, 'c')
analyze_system(d, 1, 'd')