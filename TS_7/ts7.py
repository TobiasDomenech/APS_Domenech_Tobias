# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 16:28:21 2025

@author: Tobi
"""

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio


# REQUERIMIETNOS PLANTILLA
# Normalizada en Nyquist

fs = 1000 # Hz
fs_norma = 2
nyq_frec = fs/2
 
fpass = np.array( [1.0, 35.] )
ripple = 1 # dB alfamax
fstop = np.array( [0.1, 50.] )
atenuacion = 40 # dB alfamin
 
# DATOS ECG

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg = mat_struct['ecg_lead'].flatten()

ecg = ecg/np.std(ecg) # Normalizo la señal

regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        ) # CON RUIDO

regs_interes_sin_ruido = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        ) # SIN RUIDO

# %%
###############
# FILTROS IIR #
###############

# Utilizo de aproximación de Chebyshev y Cauer 


# Análisis filtro

npoints = 1000

##########
# CHEBY 1#
##########
    
sos_cheby1 = sig.iirdesign(fpass, fstop, ripple, atenuacion, ftype = 'cheby1', output= 'sos', fs = fs) 
      
w_ch1, hh_ch1 = sig.sosfreqz(sos_cheby1, worN = npoints)

plt.figure(1)
plt.plot(w_ch1/np.pi*nyq_frec, 20*np.log10(np.abs(hh_ch1)+1e-15), label='cheby 1')


plt.title('Plantilla de diseño - IIR: Cheby 1')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = atenuacion, fs = fs)
plt.show()

ecg_filt_ch1 = sig.sosfiltfilt(sos_cheby1, ecg)

##########
# CAUER#
##########

sos_ellip = sig.iirdesign(fpass, fstop, ripple, atenuacion, ftype = 'ellip', output= 'sos', fs = fs) 
      
w_ellip, hh_ellip = sig.sosfreqz(sos_ellip, worN = npoints)

plt.figure(2)
plt.plot(w_ellip/np.pi*nyq_frec, 20*np.log10(np.abs(hh_ellip)+1e-15), label='cheby 1')


plt.title('Plantilla de diseño - IIR: Cauer')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = atenuacion, fs = fs)
plt.show()

ecg_filt_cauer = sig.sosfiltfilt(sos_ellip, ecg)

# %%


###############
# FILTROS FIR #
###############

#############################################
# Método de ventanas; uso ventana de Kaiser.#
#############################################

cant_coef = 4501

npoints = 1000

frecs_ventanas = np.array([0.0, fstop[0], fpass[0], fpass[1], fstop[1], nyq_frec])
gains_ventanas = np.array([0, 0, 1, 1, 0, 0 ])

coef_ventanas = sig.firwin2(cant_coef, frecs_ventanas, gains_ventanas , window=('kaiser',12), fs = fs)

w_win, h_win = sig.freqz(coef_ventanas, worN= npoints)

plt.figure(3)

plt.plot(w_win/np.pi*nyq_frec, 20*np.log10(np.abs(h_win)+1e-15), label='Kaiser-b12')

plt.title('Plantilla de diseño - FIR: Ventanas - Kaiser-b12')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()


plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = atenuacion, fs = fs)

plt.show()

ecg_filt_ventanas = sig.convolve(ecg,coef_ventanas, mode='same')

############################
# Méto de cuadrados mínimos#
############################


cant_coef_ls = 1501

frecs_ls = np.array([0.0,         fstop[0],   fpass[0],  fpass[1],     fpass[1]+1,  nyq_frec  ]) # Hago simétrico con fpass[1]+1
gains_ls = np.array([0,0,1,1,0,0])


coef_numerador_ls = sig.firls(cant_coef_ls, frecs_ls, gains_ls, fs = fs)

w_ls, h_ls = sig.freqz(coef_numerador_ls, worN= npoints)

plt.figure(4)

plt.plot(w_ls/np.pi*nyq_frec, 20*np.log10(np.abs(h_ls)+1e-15), label='Cuadrados Mínimos')

plt.title('Plantilla de diseño - FIR: Cuadrados Mínimos')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()


plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = atenuacion, fs = fs)
plt.show()

ecg_filt_ls = sig.convolve(ecg, coef_numerador_ls, mode='same')

#############################
# Método de Parks-McClellan #
#############################

cant_coef_remez = 2351

frecs_remez = np.array([0.0, fstop[0], fpass[0], fpass[1],  fpass[1]+1,  nyq_frec  ]) # Hago simétrico con fpass[1]+1
gains_remez = np.array([0,0,1,1,0,0])

coef_numerador_remez = sig.remez(numtaps= cant_coef_remez, bands= frecs_remez, desired= gains_remez[::2], fs = fs)

w_remez, h_remez = sig.freqz(coef_numerador_remez, worN= npoints)

plt.figure(5)

plt.plot(w_remez/np.pi*nyq_frec, 20*np.log10(np.abs(h_remez)+1e-15), label='Parks-McClellan')

plt.title('Plantilla de diseño - FIR: Parks-McClellan - Orden: {}'.format(cant_coef_remez-1))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()


plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = atenuacion, fs = fs)
plt.show()

ecg_filt_remez = sig.convolve(ecg, coef_numerador_remez, mode='same')
# %%


#####################################################
# Gráficos de ECG pasado por los distintos filtros


plt.figure(6,figsize=(12, 18))

plt.subplot(5,1,1)
plt.plot(ecg[0:200000], label = 'ECG')
plt.plot(ecg_filt_ch1[0:200000], label = 'Cheby 1')
plt.ylabel('Amplitud')
plt.title('ECG filtrado - IIR: Cheby1')
plt.legend()
plt.grid(True)

plt.subplot(5,1,2)
plt.plot(ecg[0:200000], label = 'ECG')
plt.plot(ecg_filt_cauer[0:200000], label = 'Cauer')
plt.ylabel('Amplitud')
plt.title('ECG filtrado - IIR: Cauer')
plt.legend()
plt.grid(True)

plt.subplot(5,1,3)
plt.plot(ecg[0:200000], label = 'ECG')
plt.plot(ecg_filt_ventanas[0:200000], label = 'Ventanas (Kaiser-b12)')
plt.ylabel('Amplitud')
plt.title('ECG filtrado - FIR: Ventanas (Kaiser-b12)')
plt.legend()
plt.grid(True)

plt.subplot(5,1,4)
plt.plot(ecg[0:200000], label = 'ECG')
plt.plot(ecg_filt_ls[0:200000], label = 'Cuadrados Mínimos')
plt.ylabel('Amplitud')
plt.title('ECG filtrado - FIR: Cuadrados Mínimos')
plt.legend()
plt.grid(True)

plt.subplot(5,1,5)
plt.plot(ecg[0:200000], label = 'ECG')
plt.plot(ecg_filt_remez[0:200000], label = 'Parks - McClellan')
plt.xlabel('Muestras [#]')
plt.ylabel('Amplitud')
plt.title('ECG filtrado - FIR: Parks - McClellan')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# %%


# Asegurar que las regiones sean enteros
regs_interes_sin_ruido_int = [(int(start), int(end)) for start, end in regs_interes_sin_ruido]

import matplotlib.pyplot as plt
import numpy as np


def plot_region_comparison(original, filters_dict, region, title_suffix):
  
    start, end = map(int, region)
    samples = np.arange(start, end)
    
    fig = plt.figure(figsize=(14, 12), facecolor='white')
    fig.suptitle(f"Análisis Comparativo: {title_suffix}\nMuestras {start:,} a {end:,}".replace(',', '.'),
                y=0.95, fontsize=10)
    
    # Crear subplots con espacio optimizado
    gs = fig.add_gridspec(5, 1, hspace=0.4)
    axs = [fig.add_subplot(gs[i]) for i in range(5)]
    
    for idx, (name, filtered) in enumerate(filters_dict.items()):
        ax = axs[idx]
        ax.plot(samples, original[start:end], lw = 1,  label='ECG')
        ax.plot(samples, filtered[start:end], lw = 1, label=name)
        
        # Configuración profesional del subplot
        ax.set_title(f"Filtro: {name}", pad=8, fontsize= 10)
        ax.set_xlim(start, end)
        ax.set_ylabel('Amplitud', labelpad=8)
        ax.legend(loc='upper right', framealpha=1)
        ax.grid(True)
        
        # Formateo de ejes profesional
        
        # Ocultar ticks y etiquetas del eje X en todos excepto el último subplot
        if idx < 4:
           ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
           ax.set_xlabel('Muestras [#]', labelpad=10)
           ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', '.')))
    
    plt.tight_layout()
    plt.show()

# Diccionario de filtros 
filters = {
    'Chebyshev I': ecg_filt_ch1,
    'Cauer': ecg_filt_cauer,
    'Kaiser': ecg_filt_ventanas,
    'Mínimos Cuadrados': ecg_filt_ls,
    'Parks-McClellan': ecg_filt_remez
}

# Procesamiento de regiones CON ruido

for i, region in enumerate(regs_interes, 1):
    plot_region_comparison(ecg, filters, region, f"Región con ruido {i}")

# Procesamiento de regiones SIN ruido

for i, region in enumerate(regs_interes_sin_ruido, 1):
    plot_region_comparison(ecg, filters, region, f"Región sin ruido {i}")



