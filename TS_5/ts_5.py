# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:30:38 2025

@author: Tobi
"""
#%%
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

import pandas as pd

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)
#%%



def blackman_tukey(x,  M = None):    
    
    # N = len(x)
    x_z = x.shape
    
    N = np.max(x_z)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len];

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    Px = Px.reshape(x_z)

    return Px;

def estimacion_PSD_y_BW (signal, fs, titulo = "Nombre de la señal"):
    N = len(signal)
    
    if titulo == "ECG con ruido":
        M = N//3
    if titulo == "ECG sin ruido":
        M = N//4
    if titulo == "PPG con ruido":
        M = N//4
    if titulo == "PPG sin ruido":
        M = N//5
    if titulo == "La Cucaracha":
        M = N//2
    if titulo == "Prueba de Audio":
        M = N//2
    if titulo == "Silbido":
        M = N//2
    
    
    #Normalizo por varianza
    signal = signal/np.std(signal)
    
    
    #Aplico B-T
    psd = blackman_tukey(signal, M)
    
    ff = np.linspace(0, fs, N, endpoint=False)
    
    # Corroborar Parseval y calculo de energía acumulada
    
    ft_signal = np.fft.fft(signal)
    ft_SIGNAL = np.abs(ft_signal ** 2)
    parseval = np.mean(ft_SIGNAL)
    
    energia = np.sum(signal ** 2)
    
    # Busco trabajar con una proporcion del 95 - 98% de la potencia
    # Trabajo con la mitad de los datos
    mitad_psd = psd[:N//2]
    
    # Energia total
    
    ener_total = np.sum(mitad_psd)
    
    # Energia acumulada normalizada por total
    
    ener_acumulada = np.cumsum(mitad_psd)/ener_total
    
    # Estimación ancho de banda
    
    mitad_ff = ff[:N//2]
    
    indice_95 = np.where(ener_acumulada >= 0.95)[0][0]
    f95 = mitad_ff[indice_95]
    indice_98 = np.where(ener_acumulada >= 0.98)[0][0]
    f98 = mitad_ff[indice_98]
    
    # Gráfico PSD
    plt.plot( ff[:N//2], 10* np.log10( np.abs(psd[:N//2]) + 1e-10) )
    plt.axvline(x=f95, color='r', linestyle='--', label=f'95% energía: {f95:.2f} Hz')
    plt.axvline(x=f98, color='green', linestyle='--', label=f'98% energía: {f98:.2f} Hz')
    plt.ylabel('Densidad de Potencia [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.title(f'PSD: {titulo}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    print(f"Frecuencia al 95% de la energía acumulada: {f95:.2f} Hz")
    print(f"Frecuencia al 98% de la energía acumulada: {f98:.2f} Hz")
 
    
    return f95, f98;

#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_noise = vertical_flaten(mat_struct['ecg_lead'])
ecg_noise = mat_struct['ecg_lead'].reshape(-1, 1)

f95_ECG_noise, f98_ECG_noise = estimacion_PSD_y_BW (ecg_noise, fs_ecg, titulo = "ECG con ruido")
#%%

##################
## ECG sin ruido
##################

ecg_limpio = np.load('ecg_sin_ruido.npy')

f95_ECG_limpio, f98_ECG_limpio = estimacion_PSD_y_BW (ecg_limpio, fs_ecg, titulo = "ECG sin ruido" )

#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

# # Cargar el archivo CSV como un array de NumPy
ppg_noise = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
f95_PPG_noise, f98_PPG_noise = estimacion_PSD_y_BW (ppg_noise, fs_ppg, titulo = "PPG con ruido" )

##################
## PPG sin ruido
##################

ppg_limpio = np.load('ppg_sin_ruido.npy')

f95_PPG_limpio, f98_PPG_limpio = estimacion_PSD_y_BW (ppg_limpio, fs_ppg, titulo = "PPG sin ruido" )


#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy

fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
if wav_data.ndim > 1:
    wav_data = wav_data[:, 0]
wav_data = wav_data.astype(np.float64)
f95_cuca, f98_cuca = estimacion_PSD_y_BW (wav_data, fs_audio, titulo = "La Cucaracha" )

fs_audio2, wav_data2 = sio.wavfile.read('prueba psd.wav')
if wav_data2.ndim > 1:
    wav_data2 = wav_data2[:, 0]
wav_data2 = wav_data2.astype(np.float64)
f95_pruebapsd, f98_pruebapsd = estimacion_PSD_y_BW (wav_data2, fs_audio2, titulo = "Prueba de Audio" )

fs_audio3, wav_data3 = sio.wavfile.read('silbido.wav')
if wav_data3.ndim > 1:
    wav_data3 = wav_data3[:, 0]
wav_data3 = wav_data3.astype(np.float64)
f95_silbido, f98_silbido = estimacion_PSD_y_BW (wav_data3, fs_audio3, titulo = "Silbido")

# Crear tabla de resultados
tabla = pd.DataFrame({
    "Ancho de Banda al 95% [Hz]": [
        f95_ECG_noise,
        f95_ECG_limpio,
        f95_PPG_noise,
        f95_PPG_limpio,
        f95_cuca,
        f95_pruebapsd,
        f95_silbido
    ],
    "Ancho de Banda al 98% [Hz]": [
        f98_ECG_noise,
        f98_ECG_limpio,
        f98_PPG_noise,
        f98_PPG_limpio,
        f98_cuca,
        f98_pruebapsd,
        f98_silbido
    ]
}, index=[
    "ECG con ruido",
    "ECG sin ruido",
    "PPG con ruido",
    "PPG sin ruido",
    "La cucaracha",
    "Prueba de audio",
    "Silbido"
])

# Estilo para la tabla
tabla.style.set_caption("Estimación de Ancho de Banda al 95% y 98% de Energía") \
     .format("{:.2f}") \
     .set_table_styles([{
         "selector": "caption",
         "props": [("font-size", "16px"), ("font-weight", "bold")]
     }])

