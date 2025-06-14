# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:03:23 2025

@author: Tobi
"""

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

# MATCHED FILTRER

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
fs = 1000

patron = (mat_struct['qrs_pattern1']/np.std(mat_struct['qrs_pattern1'])).flatten()

ecg_one_lead = ecg_one_lead/np.std(ecg_one_lead)

correlacion = np.correlate(patron,ecg_one_lead)
correlacion = correlacion/np.std(correlacion)

#♣picos = sig.find_peaks(correlacion,height= 1, distance=100 )

plt.figure()
plt.plot(correlacion,label='ECG_correlacionado')
plt.plot(ecg_one_lead,label='ECG')
#plt.plot(picos,'rx', label='picos')
plt.legend()
plt.grid(True)
plt.show()

# %%

# FILTRADO ESPACIAL

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = (mat_struct['ecg_lead']/np.std(mat_struct['ecg_lead'])).flatten()
N = len(ecg_one_lead)
fs = 1000

picos = mat_struct['qrs_detections'].flatten()

pre_samples = int(0.100 * fs)  # 100 ms antes
post_samples = int(0.300 * fs)  # 300 ms después
ventana = pre_samples + post_samples

latidos = []

for r in picos:
   inicio = r - pre_samples
   fin = r + post_samples

    # Verificar que no se salga de los límites de la señal
   if inicio >= 0 and fin <= N:
        latido = ecg_one_lead[inicio:fin]
        latido = latido - np.mean(latido)
        latidos.append(latido)

latidos = np.array(latidos)

# Eje de tiempo en milisegundos
tiempo = np.linspace(-100, 300, ventana)

# Graficar todos los latidos superpuestos
plt.figure(figsize=(10, 5))
for latido in latidos:
  plt.plot(tiempo, latido, alpha=0.6)
plt.title("Latidos sincronizados por picos R")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud ECG")
plt.grid(True)
plt.show()

# Antes de hacer promedio, conviene separar entre los latidos normales y ectopicos
# Para eseo puedo mirar valores tipicos del grafico y separarlos pq si no arruinará promedio 

#latido_prom = np.mean(latidos, axis=0)

