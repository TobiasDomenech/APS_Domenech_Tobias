# =============================================================================
# TRABAJO INTEGRADOR FINAL - ANÁLISIS Y PROCESAMIENTO DE SEÑALES
# Universidad Nacional de San Martín - Ingeniería Biomédica
#
# Análisis comparativo de patrones ERD/ERS entre ejecución real
# e imaginación motora en señales EEG
#
# SCRIPT 1: Descarga y exploración inicial del dataset
# =============================================================================

# -----------------------------------------------------------------------------
# BLOQUE 1: Importación de librerías y configuración global
# -----------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
import pickle

# Configuración de MNE: reduce verbosidad de mensajes en consola
mne.set_log_level('WARNING')

# Configuración de matplotlib para gráficos más legibles
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Ruta local donde MNE va a guardar los archivos descargados
# Por defecto usa ~/mne_data, se puede cambiar si querés otra ubicación
RUTA_DATOS = os.path.join(os.path.expanduser('~'), 'mne_data')
print(f"Los datos se descargarán en: {RUTA_DATOS}")


# -----------------------------------------------------------------------------
# BLOQUE 2: Selección de sujetos y runs experimentales
# -----------------------------------------------------------------------------
#
# El EEG Motor Movement/Imagery Dataset contiene 14 runs por sujeto:
#   - Runs 1-2: Baselines (ojos abiertos / ojos cerrados) — no los usamos
#   - Runs 3, 7, 11: Tarea 1 - MOVIMIENTO REAL de puño izquierdo/derecho
#   - Runs 4, 8, 12: Tarea 2 - IMAGINACIÓN MOTORA de puño izquierdo/derecho
#   - Runs 5, 9, 13: Tarea 3 - Movimiento real bilateral (no usamos)
#   - Runs 6, 10, 14: Tarea 4 - Imaginación bilateral (no usamos)
#
# Dentro de cada run, los eventos están marcados con anotaciones:
#   T0 = Reposo (baseline)
#   T1 = Movimiento/imaginación de puño IZQUIERDO
#   T2 = Movimiento/imaginación de puño DERECHO
#
# NOTA: Los sujetos 88, 92 y 100 tienen frecuencia de muestreo diferente,
# por eso se excluyen tradicionalmente del análisis.

# Sujetos seleccionados para el análisis
SUJETOS = [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 42]  # Podés cambiar por otros IDs entre 1 y 109

# Runs de movimiento real e imaginación motora
RUNS_MOVIMIENTO_REAL = [3, 7, 11]
RUNS_IMAGINACION = [4, 8, 12]
RUNS_TOTAL = RUNS_MOVIMIENTO_REAL + RUNS_IMAGINACION

print(f"\nSujetos a analizar: {SUJETOS}")
print(f"Runs de movimiento real: {RUNS_MOVIMIENTO_REAL}")
print(f"Runs de imaginación motora: {RUNS_IMAGINACION}")


# -----------------------------------------------------------------------------
# BLOQUE 3: Descarga y carga de datos con MNE
# -----------------------------------------------------------------------------

def descargar_datos_sujeto(sujeto_id, runs):
    """
    Descarga los archivos EDF del sujeto especificado desde PhysioNet.
    Si los archivos ya existen localmente, no los vuelve a descargar.

    Parámetros
    ----------
    sujeto_id : int
        Número del sujeto (1-109)
    runs : list
        Lista de runs a descargar

    Retorna
    -------
    lista_rutas : list
        Rutas locales a los archivos EDF descargados
    """
    print(f"\nDescargando datos del sujeto S{sujeto_id:03d}...")
    lista_rutas = mne.datasets.eegbci.load_data(
        subjects=sujeto_id,
        runs=runs,
        path=RUTA_DATOS,
        update_path=True
    )
    print(f"  {len(lista_rutas)} archivos descargados/verificados")
    return lista_rutas


# Descargamos los datos de todos los sujetos seleccionados
archivos_por_sujeto = {}
for suj in SUJETOS:
    archivos_por_sujeto[suj] = descargar_datos_sujeto(suj, RUNS_TOTAL)

print("\n✓ Descarga completa")


# -----------------------------------------------------------------------------
# BLOQUE 4: Exploración inicial del registro
# -----------------------------------------------------------------------------
#
# Vamos a inspeccionar en detalle un registro para entender su estructura:
# frecuencia de muestreo, número de canales, duración, tipo de eventos, etc.

# Elegimos el sujeto 1, run 3 (primer bloque de movimiento real) como ejemplo
sujeto_ejemplo = 1
run_ejemplo = 3
indice_run = RUNS_TOTAL.index(run_ejemplo)
archivo_ejemplo = archivos_por_sujeto[sujeto_ejemplo][indice_run]

# Cargamos el archivo EDF con MNE
raw_ejemplo = mne.io.read_raw_edf(archivo_ejemplo, preload=True)

# Estandarizamos los nombres de los canales al sistema 10-10 internacional
# (los archivos originales usan una nomenclatura ligeramente diferente)
mne.datasets.eegbci.standardize(raw_ejemplo)

# Aplicamos el montaje estándar 10-05 para tener coordenadas espaciales
# de cada electrodo (útil para visualizaciones topográficas y filtro Laplaciano)
montaje = mne.channels.make_standard_montage('standard_1005')
raw_ejemplo.set_montage(montaje)

# Impresión de información básica del registro
print("\n" + "="*60)
print(f"INFORMACIÓN DEL REGISTRO: Sujeto {sujeto_ejemplo}, Run {run_ejemplo}")
print("="*60)
print(f"Frecuencia de muestreo: {raw_ejemplo.info['sfreq']} Hz")
print(f"Duración total: {raw_ejemplo.n_times / raw_ejemplo.info['sfreq']:.1f} s")
print(f"Número de canales: {len(raw_ejemplo.ch_names)}")
print(f"Primeros 10 canales: {raw_ejemplo.ch_names[:10]}")

# Extraemos los eventos (anotaciones T0, T1, T2) del registro
eventos, id_eventos = mne.events_from_annotations(raw_ejemplo)
id_eventos_limpio = {str(k): v for k, v in id_eventos.items()}
print(f"\nMapeo de eventos: {id_eventos_limpio}")
print(f"Cantidad total de eventos: {len(eventos)}")

# Contamos cuántos eventos hay de cada tipo
for nombre, codigo in id_eventos.items():
    cantidad = np.sum(eventos[:, 2] == codigo)
    print(f"  {nombre}: {cantidad} eventos")


# -----------------------------------------------------------------------------
# BLOQUE 5: Visualización de la señal cruda en dominio temporal
# -----------------------------------------------------------------------------
#
# Graficamos 10 segundos de señal en C3 y C4 para inspeccionar visualmente
# la calidad del registro: nivel de ruido, presencia de artefactos, amplitud.

# Extraemos los datos como array NumPy
# get_data() retorna una matriz de forma (canales, muestras) en Volts
datos = raw_ejemplo.get_data()
fs = raw_ejemplo.info['sfreq']  # frecuencia de muestreo

# Índices de C3 y C4
idx_C3 = raw_ejemplo.ch_names.index('C3')
idx_C4 = raw_ejemplo.ch_names.index('C4')

# Convertimos de Volts a microVolts (unidad convencional en EEG)
senal_C3 = datos[idx_C3] * 1e6
senal_C4 = datos[idx_C4] * 1e6

# Tomamos los primeros 10 segundos
duracion_seg = 10
n_muestras = int(duracion_seg * fs)
tiempo = np.arange(n_muestras) / fs

# Gráfico
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(tiempo, senal_C3[:n_muestras], color='#2a78d6', linewidth=0.8)
axes[0].set_ylabel('Amplitud [μV]')
axes[0].set_title(f'Señal EEG cruda - Canal C3 (Sujeto {sujeto_ejemplo}, Run {run_ejemplo})')

axes[1].plot(tiempo, senal_C4[:n_muestras], color='#e34948', linewidth=0.8)
axes[1].set_ylabel('Amplitud [μV]')
axes[1].set_xlabel('Tiempo [s]')
axes[1].set_title('Señal EEG cruda - Canal C4')

plt.tight_layout()
plt.savefig('fig_01_senal_cruda.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Figura guardada: fig_01_senal_cruda.png")


# -----------------------------------------------------------------------------
# BLOQUE 6: Análisis espectral inicial de la señal cruda
# -----------------------------------------------------------------------------
#
# Calculamos la Densidad Espectral de Potencia (PSD) mediante el método
# de Welch para caracterizar el contenido frecuencial de la señal cruda.
# Esto nos permite JUSTIFICAR el diseño del filtro que aplicaremos después.
#
# Método de Welch (Welch, 1967):
#   1. Divide la señal en segmentos solapados
#   2. Aplica una ventana a cada segmento (Hann por defecto)
#   3. Calcula el periodograma de cada segmento
#   4. Promedia los periodogramas -> estimación estable de la PSD
#
# Parámetros elegidos:
#   - nperseg = 2 * fs -> ventanas de 2 segundos (buena resolución en Mu/Beta)
#   - noverlap = nperseg // 2 -> solapamiento del 50%
#   - window = 'hann' -> ventana estándar para EEG

nperseg = int(2 * fs)  # 2 segundos = 320 muestras a 160 Hz
noverlap = nperseg // 2

f_C3, psd_C3 = signal.welch(senal_C3, fs=fs, window='hann',
                             nperseg=nperseg, noverlap=noverlap)
f_C4, psd_C4 = signal.welch(senal_C4, fs=fs, window='hann',
                             nperseg=nperseg, noverlap=noverlap)

# Gráfico en escala logarítmica (dB) para ver mejor todo el rango dinámico
fig, ax = plt.subplots(figsize=(12, 5))

ax.semilogy(f_C3, psd_C3, label='C3', color='#2a78d6', linewidth=1.5)
ax.semilogy(f_C4, psd_C4, label='C4', color='#e34948', linewidth=1.5)

# Sombreamos las bandas de interés fisiológico
ax.axvspan(8, 12, alpha=0.15, color='green', label='Banda Mu (8-12 Hz)')
ax.axvspan(13, 30, alpha=0.15, color='orange', label='Banda Beta (13-30 Hz)')

ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('PSD [μV²/Hz]')
ax.set_title(f'Espectro de potencia de la señal cruda (Sujeto {sujeto_ejemplo}, Run {run_ejemplo})')
ax.legend(loc='upper right')
ax.set_xlim(0, fs/2)  # Frecuencia de Nyquist

plt.tight_layout()
plt.savefig('fig_02_psd_cruda.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_02_psd_cruda.png")


# -----------------------------------------------------------------------------
# BLOQUE 7: Guardado de datos para uso en el Script 2
# -----------------------------------------------------------------------------
#
# Guardamos las rutas de los archivos y la configuración en un pickle
# para que el Script 2 pueda acceder directamente a los datos sin
# tener que redescargar ni reprocesar.

configuracion = {
    'sujetos': SUJETOS,
    'runs_movimiento_real': RUNS_MOVIMIENTO_REAL,
    'runs_imaginacion': RUNS_IMAGINACION,
    'runs_total': RUNS_TOTAL,
    'archivos_por_sujeto': archivos_por_sujeto,
    'fs': fs,
    'ruta_datos': RUTA_DATOS
}

with open('configuracion_datos.pkl', 'wb') as f:
    pickle.dump(configuracion, f)

print("\n" + "="*60)
print("SCRIPT 1 COMPLETADO")
print("="*60)
print("Archivos generados:")
print("  - fig_01_senal_cruda.png")
print("  - fig_02_psd_cruda.png")
print("  - configuracion_datos.pkl (usado por el Script 2)")
print("\nProceder con el Script 2: 02_analisis_principal.py")