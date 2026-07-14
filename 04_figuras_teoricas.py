# =============================================================================
# SCRIPT 4: Figuras te√≥ricas para el informe
# =============================================================================
#
# Este script genera figuras conceptuales/did√°cticas que fundamentan las
# decisiones metodol√≥gicas del trabajo. No depende del an√°lisis principal.

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# -----------------------------------------------------------------------------
# FIGURA T1: Principio de incertidumbre tiempo-frecuencia
# -----------------------------------------------------------------------------
#
# Ilustra el trade-off entre resoluci√≥n temporal y frecuencial:
# - STFT con ventana corta: buena localizaci√≥n temporal, mala en frecuencia
# - STFT con ventana larga: buena resoluci√≥n en frecuencia, mala temporal
# - CWT-Morlet: resoluci√≥n multiresoluci√≥n adaptativa (mejor de ambos mundos
#   dentro de las limitaciones del principio de incertidumbre)
#
# Justificaci√≥n te√≥rica: el principio de incertidumbre de Gabor establece que
# Œît ¬∑ Œîf ‚â• 1/(4œÄ), donde Œît y Œîf son las dispersiones temporal y frecuencial.
# No es posible mejorar simult√°neamente ambas resoluciones.

import pywt

fs_sim = 200  # Hz
t_sim = np.arange(0, 4, 1/fs_sim)

# Se√±al sint√©tica con dos eventos oscilatorios en distintos tiempos y frecuencias
# Evento 1: burst de 10 Hz alrededor de t=1s
# Evento 2: burst de 25 Hz alrededor de t=2.5s
gaussiana1 = np.exp(-((t_sim - 1.0)**2) / (2 * 0.15**2))
gaussiana2 = np.exp(-((t_sim - 2.5)**2) / (2 * 0.10**2))

senal_sim = (gaussiana1 * np.sin(2*np.pi*10*t_sim) +
              gaussiana2 * np.sin(2*np.pi*25*t_sim) +
              0.1*np.random.randn(len(t_sim)))

# STFT con ventana corta (buena resoluci√≥n temporal)
f_stft_corta, t_stft_corta, Sxx_corta = signal.spectrogram(
    senal_sim, fs=fs_sim, nperseg=32, noverlap=28, window='hann')

# STFT con ventana larga (buena resoluci√≥n frecuencial)
f_stft_larga, t_stft_larga, Sxx_larga = signal.spectrogram(
    senal_sim, fs=fs_sim, nperseg=128, noverlap=120, window='hann')

# CWT-Morlet
freqs_cwt_sim = np.arange(4, 41, 1)
wavelet = 'cmor1.5-1.0'
escalas_sim = pywt.central_frequency(wavelet) * fs_sim / freqs_cwt_sim
coefs_cwt, _ = pywt.cwt(senal_sim, escalas_sim, wavelet, sampling_period=1/fs_sim)
potencia_cwt = np.abs(coefs_cwt)**2

# Visualizaci√≥n
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# Panel 1: Se√±al temporal
axes[0, 0].plot(t_sim, senal_sim, color='#424242', linewidth=0.9)
axes[0, 0].axvline(1.0, color='#e63946', linestyle='--', alpha=0.6,
                    label='Evento 1: 10 Hz en t=1s')
axes[0, 0].axvline(2.5, color='#2a9d8f', linestyle='--', alpha=0.6,
                    label='Evento 2: 25 Hz en t=2.5s')
axes[0, 0].set_xlabel('Tiempo [s]')
axes[0, 0].set_ylabel('Amplitud')
axes[0, 0].set_title('Se√±al sint√©tica con dos eventos localizados')
axes[0, 0].legend(fontsize=9)
axes[0, 0].set_xlim(0, 4)

# Panel 2: STFT con ventana corta
im2 = axes[0, 1].pcolormesh(t_stft_corta, f_stft_corta, Sxx_corta,
                              shading='auto', cmap='viridis')
axes[0, 1].set_ylim(0, 50)
axes[0, 1].set_xlabel('Tiempo [s]')
axes[0, 1].set_ylabel('Frecuencia [Hz]')
axes[0, 1].set_title('STFT ventana corta (32 muestras = 160 ms)')

# Panel 3: STFT con ventana larga
im3 = axes[1, 0].pcolormesh(t_stft_larga, f_stft_larga, Sxx_larga,
                              shading='auto', cmap='viridis')
axes[1, 0].set_ylim(0, 50)
axes[1, 0].set_xlabel('Tiempo [s]')
axes[1, 0].set_ylabel('Frecuencia [Hz]')
axes[1, 0].set_title('STFT ventana larga (128 muestras = 640 ms)')

# Panel 4: CWT-Morlet
im4 = axes[1, 1].pcolormesh(t_sim, freqs_cwt_sim, potencia_cwt,
                              shading='auto', cmap='viridis')
axes[1, 1].set_ylim(0, 50)
axes[1, 1].set_xlabel('Tiempo [s]')
axes[1, 1].set_ylabel('Frecuencia [Hz]')
axes[1, 1].set_title('CWT-Morlet\nResoluci√≥n multiresoluci√≥n adaptativa')

plt.suptitle('Principio de incertidumbre tiempo-frecuencia: comparaci√≥n de m√©todos',
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig('fig_T1_incertidumbre_tf.png', dpi=150, bbox_inches='tight')
plt.show()

print("‚úì Figura T1 guardada: fig_T1_incertidumbre_tf.png")

# -----------------------------------------------------------------------------
# FIGURA T2: Comparaci√≥n de filtros IIR pasabanda
# -----------------------------------------------------------------------------
#
# Justifica la elecci√≥n del filtro Butterworth para el an√°lisis de EEG.
# Compara Butterworth, Chebyshev tipo I y el√≠ptico con las mismas
# especificaciones de plantilla.
#
# Justificaci√≥n te√≥rica:
# - Butterworth: respuesta en magnitud "maximally flat" en banda de paso.
#   Sin ripple. Ideal cuando la morfolog√≠a de la se√±al importa.
# - Chebyshev I: ripple en banda de paso pero corte m√°s pronunciado.
#   Menor orden requerido para las mismas especificaciones.
# - El√≠ptico: ripple en ambas bandas, corte muy pronunciado.
#   Orden m√≠nimo pero mayor distorsi√≥n de amplitud.
#
# Para EEG, donde la morfolog√≠a de las ondas (Mu, Beta) es informativa
# fisiol√≥gicamente, se prefiere Butterworth para no introducir ripple
# que altere las relaciones de amplitud entre componentes.

fs_filtro = 160  # Hz (igual al dataset EEGBCI)

# Especificaciones de la plantilla (comunes a los tres dise√±os)
wp = np.array([8.0, 30.0])   # banda de paso
ws = np.array([4.0, 40.0])   # banda de stop
gpass_final = 3               # dB ripple en banda de paso
gstop_final = 40              # dB atenuaci√≥n en banda de stop

# Compensaci√≥n por filtrado bidireccional
gp = gpass_final / 2
gs = gstop_final / 2

# Dise√±o de los tres tipos de filtros
sos_butter = signal.iirdesign(wp=wp, ws=ws, gpass=gp, gstop=gs,
                                ftype='butter', output='sos', fs=fs_filtro)
sos_cheby1 = signal.iirdesign(wp=wp, ws=ws, gpass=gp, gstop=gs,
                                ftype='cheby1', output='sos', fs=fs_filtro)
sos_ellip = signal.iirdesign(wp=wp, ws=ws, gpass=gp, gstop=gs,
                                ftype='ellip', output='sos', fs=fs_filtro)

# C√°lculo de respuestas en frecuencia
w_b, h_b = signal.sosfreqz(sos_butter, worN=4096, fs=fs_filtro)
w_c, h_c = signal.sosfreqz(sos_cheby1, worN=4096, fs=fs_filtro)
w_e, h_e = signal.sosfreqz(sos_ellip, worN=4096, fs=fs_filtro)

# √ìrdenes resultantes
orden_butter = 2 * sos_butter.shape[0]
orden_cheby1 = 2 * sos_cheby1.shape[0]
orden_ellip = 2 * sos_ellip.shape[0]

# Visualización con un solo panel para mejorar la legibilidad al insertar
# la figura en el informe
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(w_b, 20*np.log10(np.abs(h_b) + 1e-12),
        color='#1976d2', linewidth=2.5,
        label=f'Butterworth (orden {orden_butter})')
ax.plot(w_c, 20*np.log10(np.abs(h_c) + 1e-12),
        color='#e63946', linewidth=2.5, linestyle='--',
        label=f'Chebyshev I (orden {orden_cheby1})')
ax.plot(w_e, 20*np.log10(np.abs(h_e) + 1e-12),
        color='#2a9d8f', linewidth=2.5, linestyle=':',
        label=f'Elíptico (orden {orden_ellip})')

# Sombreado de la banda de paso
ax.axvspan(wp[0], wp[1], alpha=0.10, color='green',
           label='Banda de paso (8-30 Hz)')

# Líneas de referencia de la plantilla
ax.axhline(-gpass_final, color='gray', linestyle=':', alpha=0.6,
           label=f'-{gpass_final} dB (ripple máx.)')
ax.axhline(-gstop_final, color='gray', linestyle='-.', alpha=0.6,
           label=f'-{gstop_final} dB (atenuación mín.)')

ax.set_xlabel('Frecuencia [Hz]', fontsize=12)
ax.set_ylabel('Magnitud [dB]', fontsize=12)
ax.set_title('Comparación de filtros IIR pasabanda con la misma plantilla de especificaciones',
             fontsize=13)
ax.set_xlim(0, 50)
ax.set_ylim(-80, 5)
ax.legend(loc='lower center', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_T3_comparacion_filtros.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nÓrdenes resultantes:")
print(f"  Butterworth: {orden_butter}")
print(f"  Chebyshev I: {orden_cheby1}")
print(f"  Elíptico: {orden_ellip}")

print("✓ Figura T3 guardada: fig_T3_comparacion_filtros.png")