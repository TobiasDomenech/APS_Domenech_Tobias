# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 22:52:18 2025

@author: Tobi
"""
######################
# FILTRADO NO LINEAL #
######################

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import pandas as pd

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()

ecg_one_lead = ecg_one_lead/np.std(ecg_one_lead) # Normalizo datos ecg

N = len(ecg_one_lead)

fs= 1000 #Hz
nyquist=fs/2
# %%


#####################
# FILTRO DE MEDIANA #
#####################

window1 = 201  #muestras; como fs = 1000 y tiempo viene en milisegundos => muestras = tiempo
window2 = 601  #muestras; deben ser impares

median_filter = sig.medfilt(ecg_one_lead, kernel_size= window1)
median_filter = sig.medfilt(median_filter, kernel_size= window2)

ecg_filt_mediana = ecg_one_lead - median_filter

## Tomare los datos de 720 a 740 mil muestras para mostrar resultado filtro

zoom_region = np.arange(np.max([0, 12*60*fs]), np.min([N, 12.4*60*fs]), dtype='uint')

datos = ecg_one_lead[zoom_region]


plt.figure(1, figsize=(14, 8))

plt.subplot(3,1,1)
plt.plot(ecg_one_lead, label='ECG', linewidth=1,color='green')
plt.plot(median_filter, label = 'Linea de base', linewidth=1, color='red')
 
plt.title('ECG y su Linea de Base')
plt.ylabel('Amplitud')
plt.legend(loc='lower left')
plt.grid()

plt.subplot(3,1,2)
plt.plot(ecg_one_lead, label='ECG', linewidth=0.5, alpha=0.6 ,color='green')
plt.plot(ecg_filt_mediana, label = 'ECG_filt_mediana', linewidth=0.5, color='red')
 
plt.title('ECG filtrado con mediana')
plt.ylabel('Amplitud')
plt.legend(loc='lower left')
plt.grid()

plt.subplot(3,1,3)
plt.plot(zoom_region, datos, label='ECG', linewidth=1,color='green')
plt.plot(zoom_region, ecg_filt_mediana[zoom_region], label = 'ECG_filt_mediana', linewidth=1, color='red')
 
plt.title('ECG filtrado con mediana')
plt.ylabel('Amplitud')
plt.xlabel('Muestras [#]') 
plt.grid()

plt.legend(loc='lower left')
plt.tight_layout()     
plt.show()
# %%


###################################
# INTERPOLACION X SPLINES CÚBICOS #
###################################

qrs = mat_struct['qrs_detections'].flatten() # Extraigo los valores donde hay latidos


n0 = int(0.1 * fs)  # 100 ms antes del QRS para inicio del segmento PQ
window_length = int(0.02 * fs)  # 20 ms para el segmento de análisis



mi = qrs - n0 # Punto a distancia n0 respecto de latido
s_mi = [] # Valores que tomará señal en punto de interes mi

# Promedio para ventana de 20 ms (20 muestras); permite eliminar interferencia de 50 Hz
for qrs_pos in qrs:
    segment_start = int(qrs_pos - n0)
    segment_end = segment_start + window_length
    
    # Asegurarse que el segmento está dentro del rango
    if segment_start >= 0 and segment_end < N:
        segment = ecg_one_lead[segment_start:segment_end]
        s_mi.append(np.mean(segment))  # Valor promedio del segmento

s_mi = np.array(s_mi)


cs = CubicSpline(mi, s_mi)
baseline_estimate = cs(np.arange(N))  # Evaluar interpolacion en todos los puntos

ecg_filt_s3 = ecg_one_lead - baseline_estimate # Remuevo Linea de Base al ecg (filtro)


plt.figure(2, figsize=(14, 8))


plt.subplot(3, 1, 1)
plt.plot(ecg_one_lead, label='ECG Original', linewidth=1)
plt.plot(baseline_estimate, label = 'Linea de base', linewidth=1, color='orange')
 
plt.title('ECG y su Linea de Base')
plt.ylabel('Amplitud')
plt.legend(loc='lower left')
plt.grid(True)

plt.subplot(3, 1, 2)

plt.plot(ecg_one_lead, 'b', label='ECG Original', alpha=0.5)
plt.plot(ecg_filt_s3, color='orange', label='ECG_filt_s3', linewidth=1)
plt.title('ECG filtrado con Splines3')
plt.ylabel('Amplitud')
plt.legend(loc='lower left')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(zoom_region, datos, label='ECG', linewidth=1,color='blue')
plt.plot(zoom_region, ecg_filt_s3[zoom_region], label = 'ECG_filt_s3', linewidth=1, color='orange')
 
plt.title('ECG filtrado con Splines3')
plt.ylabel('Amplitud')
plt.xlabel('Muestras [#]') 
plt.legend(loc='lower left')
plt.grid(True)

plt.tight_layout()
plt.show()
# %%

##################
# MATCHED FILTER #
##################

patron = (mat_struct['qrs_pattern1']).flatten()
patron_norm = (patron - np.mean(patron)) / np.std(patron) # Normalizo patron

ecg = mat_struct['ecg_lead'].flatten()
ecg_norm = (ecg - np.mean(ecg)) / np.std(ecg) # Normalizo ECG

# Correlaciono
correlacion = np.correlate(ecg_norm,patron_norm, mode='same')

# Reescalado para comparación
ecg = ecg_norm / np.max(np.abs(ecg_norm))
correlacion = correlacion / np.max(np.abs(correlacion))

# Busco picos de la señal de correlacion
picos, _ = find_peaks(correlacion, height=0.15 , distance=200 )

plt.figure(3, figsize=(14,8))

plt.subplot(3,1,1)
plt.plot(correlacion,label='Correlacion')
plt.legend(loc='lower left')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(zoom_region, ecg[zoom_region], label='ECG', linewidth=1,color='blue')
plt.plot(zoom_region, correlacion[zoom_region], label = 'correlacion', linewidth=1, color='orange')
zoom_picos = picos[(picos >= 12*60*fs) & (picos < 12.4*60*fs)]
zoom_qrs = qrs[(qrs >= 12*60*fs) & (qrs < 12.4*60*fs)]
plt.plot(zoom_picos, correlacion[zoom_picos] , 'xr', label = 'picos detectados')
plt.plot(zoom_qrs, ecg[zoom_qrs] , 'xg', label = 'latidos')
 
plt.title('ECG vs correlacion')
plt.ylabel('Amplitud')
plt.legend(loc='lower left')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(ecg, label='ECG', linewidth=1, alpha = 0.45, color='blue')
plt.plot(correlacion, label = 'correlacion', alpha = 0.65, linewidth=1, color='orange')
plt.plot(picos, correlacion[picos] , 'xr', label = 'picos detectados')

 
plt.title('ECG vs correlacion')
plt.ylabel('Amplitud')
plt.xlabel('Muestras [#]') 
plt.legend(loc='lower left')
plt.grid(True)

plt.tight_layout()
plt.show()

# %%

################################
# PERFORMANCE DETECTOR LATIDOS #
################################

# Definir la tolerancia de 100 ms (en muestras)
tolerance_samples = int(0.1 * fs)  # 100 ms

# Inicializar contadores para la tabla de confusión
true_positives = 0
false_positives = 0
false_negatives = 0

# Crear copias de los arrays para no modificar los originales
remaining_detections = qrs.copy()
remaining_peaks = picos.copy()

# Evaluar cada latido real (qrs_detections)
for qrs_pos in qrs:
    # Encontrar picos detectados dentro de la ventana de tolerancia
    matches = remaining_peaks[(remaining_peaks >= qrs_pos - tolerance_samples) & 
                             (remaining_peaks <= qrs_pos + tolerance_samples)]
    
    if len(matches) > 0:
        true_positives += 1
        # Eliminar el pico detectado que ya fue emparejado
        remaining_peaks = remaining_peaks[remaining_peaks != matches[0]]
    else:
        false_negatives += 1

# Los picos restantes son falsos positivos
false_positives = len(remaining_peaks)

# Calcular métricas de performance
sensitivity = true_positives / (true_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)


# Crear DataFrames para mostrar los resultados
confusion_matrix = pd.DataFrame({
    '': ['Predicción Positiva', 'Predicción Negativa'],
    'Real Positivo': [true_positives, false_negatives],
    'Real Negativo': [false_positives, 'N/A']  # No tenemos verdaderos negativos en este caso
})

metrics_df = pd.DataFrame({
    'Métrica': ['Sensibilidad (Recall)', 'Precisión'],
    'Valor': [f"{sensitivity:.4f}", f"{precision:.4f}"],
    'Porcentaje': [f"{sensitivity*100:.2f}%", f"{precision*100:.2f}%"]
})

# Configuración de estilo para las tablas
def style_table(df, title):
    return (df.style
            .set_caption(title)
            .set_properties(**{'text-align': 'center'})
            .hide(axis='index')
            .set_table_styles([
                {'selector': 'caption', 
                 'props': [('font-size', '16px'), 
                          ('font-weight', 'bold'),
                          ('text-align', 'center')]},
                {'selector': 'th', 
                 'props': [('background-color', '#404040'),
                           ('color', 'white'),
                           ('font-weight', 'bold')]},
                {'selector': 'td',
                 'props': [('padding', '8px')]}
            ]))

# Mostrar resultados
print("\n" + "="*50)
print("Performance del Detector de Latidos")
print("="*50)
print(f"\nTolerancia usada: ±{tolerance_samples} muestras (±100 ms)")

# Mostrar tabla de confusión
display(style_table(confusion_matrix, "Tabla de Confusión"))

# Mostrar métricas de performance
display(style_table(metrics_df, "Métricas de Performance"))

###### PARA VER MATRIZ CONFUSIÓN EN SPYDER Y MÉTRICAS USAR ESTO ########
# # Versión compatible con Spyder
# print("\n" + "="*50)
# print("Performance del Detector de Latidos")
# print("="*50)
# print(f"\nTolerancia usada: ±{tolerance_samples} muestras (±100 ms)")

# # Tabla de confusión en texto plano
# print("\nTabla de Confusión:")
# print(f"{'':<20} | {'Real Positivo':<15} | {'Real Negativo':<15}")
# print("-"*50)
# print(f"{'Predicción Positiva':<20} | {true_positives:<15} | {false_positives:<15}")
# print(f"{'Predicción Negativa':<20} | {false_negatives:<15} | {'N/A':<15}")

# # Métricas
# print("\nMétricas de Performance:")
# print(f"- Sensibilidad (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
# print(f"- Precisión: {precision:.4f} ({precision*100:.2f}%)")

