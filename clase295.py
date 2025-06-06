# -*- coding: utf-8 -*-
"""
Created on Thu May 29 18:12:41 2025

@author: Tobi
"""

#Resumen
# El código a continuación muestra una plantilla de un filtro pasabajos para filtrar un ecg
# También se muestra la respues de fase y módulo de la funcion transferencia
# Se grafica el ECG normal y filtrado 
# Mostramos zonas de interes : frecuencias bajas, altas y donde el ECG original no tenia mucho ruido (antes de empezar a correr en una prueba de esfuerzo)

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

aprox_name = 'butter'
#aprox_name2 = 'cheby1'
# aprox_name = 'cheby2'
#aprox_name = 'ellip'

fs= 1000 #Hz
nyquist=fs/2
fpass = np.array([1.0, 35.0])
ripple = 0.5 # dB
fstop = np.array([.1, 50.])
atenuacion = 40 # dB

sos=sig.iirdesign(fpass,fstop, ripple, atenuacion, ftype=aprox_name, output='sos', fs=fs)

npoints = 1000 
w_rad= np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250)) 
w_rad= np.append(w_rad, np.linspace(40, nyquist, 500, endpoint=True))/ (nyquist* np.pi)
# esto mejoro la resolucion en la subir de 0 a 2 
w, hh = sig.sosfreqz(sos, worN=npoints) 
 
# %% Respuesta de Modulo

# moduloH=np.abs(hh)
# plt.plot(w/np.pi*fs/2,moduloH)
# plt.title('Respuesta de Módulo')  
# plt.xlabel('Frec')  
# plt.ylabel('Módulo de H') 
# plt.legend()  
# plt.show()


# %% Respuesta de Fase

# faseH=np.angle(hh)
# plt.plot(w/np.pi*fs/2,faseH)
# plt.title('Respuesta de Fase')  
# plt.xlabel('Frec')  
# plt.ylabel('Fase de H') 
# plt.legend()  
# plt.show()

#%%

# plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)+1e-15), label='sos')  
#1e-15 para no tener problemas de cero
# w/np.pi*fs/2 de cero a nyquist
  

# plt.title('Plantilla de diseño')  
# plt.xlabel('Frecuencia normalizada a Nyq [#]')  
# plt.ylabel('Amplitud [dB]')  
# plt.grid(which='both', axis='both')  

# ax = plt.gca()  
# ax.set_xlim([0, 1])  
# ax.set_ylim([-60, 10])  

# plot_plantilla(filter_type = 'bandpass', fpass = fpass, ripple = ripple , fstop = fstop, attenuation = atenuacion)  
# plt.legend()  
# plt.show()
# %% FILTRADO

# Cargar datos
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
# Plot señal original
# plt.figure()
# plt.plot(ecg_one_lead)
# plt.title('ECG Original')
# plt.show()

# Filtrado con el filtro definido (suponiendo sos ya definido)
# ecg_filtrado = sig.sosfilt(sos, ecg_one_lead, axis=0)
# demora=68 #lo estimamos a ojo contando la cantidad de muestras entre dos picos entes???
#esto corrige la demora todavia te queda la distorsion de fase: no se parece la formas de las dos senales la filtrada y la no filtrada
fig_dpi=150

# Plot señal filtrada
# plt.figure()
# plt.plot(ecg_filtrado)
# plt.title('ECG Filtrado')
# plt.show()

#%% Regiones de interes

# plt.figure(figsize=(10,5))
# plt.plot(ecg_one_lead, label='ECG Original', color='olive')
# plt.plot(ecg_filtrado, label='ECG Filtrado',color='pink')
# plt.title('ECG Original vs Filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.legend()
# plt.grid(True)
# plt.show()

#%% Regiones de interes
# regs_interes = ( 
#         np.array([5, 5.2]) *60*fs, # minutos a muestras
#         np.array([12, 12.4]) *60*fs, # minutos a muestras
#         np.array([15, 15.2]) *60*fs, # minutos a muestras
#         )

# for ii in regs_interes:
    
#     # intervalo limitado de 0 a cant_muestras
#     zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    
#     plt.figure()
#     plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2,color='olive')
#     #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
#     #plt.plot(zoom_region, ecg_filtrado[zoom_region + demora], label='Win',color='pink')
    
#     plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')
    
#     axes_hdl = plt.gca()
#     axes_hdl.legend()
#     axes_hdl.set_yticks(())
            
#     plt.show()
    
#Esto es lo que hay que ver ademas de modulo, fase y retardo
# En la banda de paso la senal tiene que ser lo mas parecida posible a lo que no esta filtrado:partes limpias sin tanto ruido en la senal que no esta filtrada
# En alta frecuencia busco donde hay movimientos erraticos, bruscos
# banda inferior (baja frecuencia) : donde hay una tendencia, movimientos de muy larga duracion. Donde haya un nivel de continua o movimientos que duran mucho tiempo
# Que sea inocuo y eficiente= 

#Una forma de estimar el orden usan la anchura dde la trnasicion
#Es mas importante una transicion mas abrupta
#Lo primero que tenes que probar es aumentar la atenuacion 

#como no tienen fase lineal en un filtro iir no puede mitigar la demora
#por eso con filtfilt estamos joya mitiga la demora


#estaria buena la comparacion entre filtfilt y filt con la misma atenuacion





#freqz abs(resp modulo) freqz angle(resp modulo)
# acumulador --> pasabajos
#           --> pasabanda
# el que hace la diferencia se llamaba en ts6 diferencia de las ultimas 2 -->diferenciador 



#derivada estimacion diferencia de muestras adyasecentes porque el espaciamiento en x es el mismo (es constante) 



#TS6 c --> pasa altos = derivador (cero en el origen) resp en frecuencia 20 db para arriba en las frec   deriva idealmente en el 25% del ancho de banda digital
#   d --> pasa banda =    se comporta identicamente que un derivador un 40%?, es insensible al ruido de alta frecuencia 


# hay que desenvolver la respuesta de fase un


# #RETARDO DE FASE
# sabemos por la teoria de fourier solo demora respuesta lineal demora en tiempo es linealidad en frec
# #Distorsion por retardo 
# el cambio de las fases relativas afecta mucho a la envolvente te


#ejemplo de distorsion de fase es lo qeu obtuvimos con filt (estaba invertida la senal)

# %%


# Metodo de mediana
## Tomare los datos de 720 a 740 mil muestras


zoom_region = np.arange(np.max([0, 12*60*fs]), np.min([N, 12.4*60*fs]), dtype='uint')
datos = ecg_one_lead[zoom_region]

window1 = 201 #ms
window2 = 601 #ms

median_filter = sig.medfilt(datos, kernel_size= window1)
median_filter = sig.medfilt(median_filter, kernel_size= window2)

# Grafico de resultado

plt.figure()
plt.plot(zoom_region, datos, label='ECG', linewidth=2,color='blue')
plt.plot(zoom_region, median_filter, label = 'mediana', linewidth=2, color='orange')
 
plt.title('ECG filtrado con mediana')
plt.ylabel('Amplitud')
plt.xlabel('Tiempo [s]') # mentira, es frecuencia
plt.grid()
 
axes_hdl = plt.gca()
axes_hdl.legend()
axes_hdl.set_yticks(())
         
plt.show()