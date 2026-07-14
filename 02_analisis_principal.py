# =============================================================================
# TRABAJO INTEGRADOR FINAL - ANÁLISIS Y PROCESAMIENTO DE SEÑALES
# Universidad Nacional de San Martín - Ingeniería Biomédica
#
# SCRIPT 2: Análisis principal
#   - Diseño y aplicación del filtro pasabanda
#   - Extracción de épocas
#   - Cálculo del ERD% con Welch
#   - Visualización tiempo-frecuencia con CWT-Morlet
#   - Comparación entre movimiento real e imaginación motora
# =============================================================================

# -----------------------------------------------------------------------------
# BLOQUE 1: Importaciones y carga de la configuración del Script 1
# -----------------------------------------------------------------------------

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
import pywt
import pandas as pd
import seaborn as sns

mne.set_log_level('WARNING')

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Cargamos la configuración generada por el Script 1
with open('configuracion_datos.pkl', 'rb') as f:
    config = pickle.load(f)

SUJETOS = config['sujetos']
RUNS_MOVIMIENTO_REAL = config['runs_movimiento_real']
RUNS_IMAGINACION = config['runs_imaginacion']
RUNS_TOTAL = config['runs_total']
archivos_por_sujeto = config['archivos_por_sujeto']
FS = config['fs']

print(f"Configuración cargada:")
print(f"  Sujetos: {SUJETOS}")
print(f"  Frecuencia de muestreo: {FS} Hz")
print(f"  Runs movimiento real: {RUNS_MOVIMIENTO_REAL}")
print(f"  Runs imaginación: {RUNS_IMAGINACION}")


# -----------------------------------------------------------------------------
# BLOQUE 2: Diseño del filtro pasabanda Butterworth
# -----------------------------------------------------------------------------
#
# JUSTIFICACIÓN DEL DISEÑO:
#
# Objetivo del filtro: aislar las bandas Mu (8-12 Hz) y Beta (13-30 Hz),
# donde ocurren los fenómenos ERD/ERS asociados a la actividad sensoriomotora.
#
# Especificaciones FINALES deseadas (post filtrado bidireccional):
#   - Banda de paso: 8 - 30 Hz con ripple máximo de 3 dB
#   - Banda de stop: hasta 4 Hz y desde 40 Hz, con atenuación mínima de 40 dB
#
# Ajuste por filtrado bidireccional:
#   Como sosfiltfilt aplica el filtro dos veces, la magnitud se eleva al
#   cuadrado, lo que en dB equivale a duplicar el ripple y la atenuación.
#   Por lo tanto, el filtro BASE debe diseñarse con la MITAD de los valores
#   deseados finales:
#      GPASS_base = 1.5 dB   -> resulta en 3 dB tras filtrado bidireccional
#      GSTOP_base = 20 dB    -> resulta en 40 dB tras filtrado bidireccional
#   Esta compensación permite obtener un filtro de menor orden manteniendo
#   las especificaciones finales requeridas.
#
# Elección del tipo de filtro: BUTTERWORTH
#   Respuesta en magnitud "maximally flat" en la banda de paso, sin
#   ondulaciones. Preserva las relaciones de amplitud de las componentes
#   de interés fisiológico. Comparado con Chebyshev (con ripple) o elíptico
#   (con ripple en ambas bandas), Butterworth es preferido en señales
#   biomédicas donde la morfología importa.
#
# Filtrado bidireccional con sosfiltfilt:
#   Aplica el filtro hacia adelante y luego hacia atrás -> FASE NULA.
#   Esto es crítico en ERD/ERS porque necesitamos preservar la ubicación
#   temporal exacta de los eventos para promediar épocas.

from pytc2.sistemas_lineales import plot_plantilla

# Especificaciones FINALES deseadas
FRECUENCIA_PASO = np.array([8.0, 30.0])     # Hz
FRECUENCIA_STOP = np.array([4.0, 40.0])     # Hz
GPASS_FINAL = 3                              # dB - ripple final deseado
GSTOP_FINAL = 40                             # dB - atenuación final deseada

# Compensación por filtrado bidireccional
GPASS = GPASS_FINAL / 2                      # 1.5 dB en el diseño base
GSTOP = GSTOP_FINAL / 2                      # 20 dB en el diseño base

nyquist = FS / 2

# Diseño del filtro en formato SOS
sos = signal.iirdesign(wp=FRECUENCIA_PASO, ws=FRECUENCIA_STOP,
                        gpass=GPASS, gstop=GSTOP,
                        ftype='butter', output='sos', fs=FS)

orden_filtro = 2 * sos.shape[0]
print(f"\nFiltro Butterworth pasabanda diseñado:")
print(f"  Banda de paso: {FRECUENCIA_PASO[0]}-{FRECUENCIA_PASO[1]} Hz")
print(f"  Banda de stop: 0-{FRECUENCIA_STOP[0]} Hz y {FRECUENCIA_STOP[1]}-{nyquist} Hz")
print(f"  Banda de transición izquierda: {FRECUENCIA_PASO[0]-FRECUENCIA_STOP[0]} Hz")
print(f"  Banda de transición derecha: {FRECUENCIA_STOP[1]-FRECUENCIA_PASO[1]} Hz")
print(f"  Ripple diseño base: {GPASS} dB (→ {GPASS_FINAL} dB post filtrado bidireccional)")
print(f"  Atenuación diseño base: {GSTOP} dB (→ {GSTOP_FINAL} dB post filtrado bidireccional)")
print(f"  Orden resultante: {orden_filtro}")

# Cálculo de la respuesta en frecuencia
npoints = 1000
w, h = signal.sosfreqz(sos, worN=npoints)

# Convertimos w de rad/muestra a Hz para el gráfico
frec_hz = w / np.pi * nyquist

# Respuesta bidireccional: se aplica dos veces, en dB se duplica
magnitud_db_base = 20 * np.log10(np.abs(h) + 1e-15)
magnitud_db_bidir = 2 * magnitud_db_base

# Visualización con formato de plantilla de la cátedra
plt.figure(figsize=(12, 6))

plt.plot(frec_hz, magnitud_db_bidir, color='#1976d2', linewidth=2,
         label=f'Butterworth bidireccional (orden {orden_filtro})')

plt.title('Plantilla de diseño - Filtro pasabanda Butterworth')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

# Función oficial de la cátedra para dibujar la plantilla
plot_plantilla(filter_type='bandpass',
               fpass=FRECUENCIA_PASO,
               ripple=GPASS_FINAL,
               fstop=FRECUENCIA_STOP,
               attenuation=GSTOP_FINAL,
               fs=FS)

plt.xlim(0, 50)
plt.ylim(-80, 5)
plt.legend(loc='lower center')
plt.tight_layout()
plt.savefig('fig_03_diseno_filtro.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_03_diseno_filtro.png")

# -----------------------------------------------------------------------------
# FILTRO ESPACIAL LAPLACIANO LOCAL
# -----------------------------------------------------------------------------
#
# JUSTIFICACIÓN (Pfurtscheller & Lopes da Silva, 1999):
#
# El EEG registrado con referencia común (como el mastoide) sufre del
# fenómeno de "volume conduction": cada electrodo capta actividad de
# fuentes distantes por la conductividad del cráneo y cuero cabelludo.
# Esto "difumina" (blurs) los patrones ERD, dificultando la localización
# espacial precisa.
#
# El filtro Laplaciano local resuelve esto restando a cada electrodo
# central el PROMEDIO de sus vecinos ortogonales inmediatos. Esto elimina
# la componente común (ruido de referencia, actividad occipital difundida)
# y realza la actividad LOCAL de la corteza sensoriomotora.
#
# Para C3 (según el sistema 10-10 internacional):
#   Vecinos ortogonales: FC3, CP3, C1, C5
#   C3_laplaciano = C3 - (FC3 + CP3 + C1 + C5) / 4
#
# Para C4:
#   Vecinos ortogonales: FC4, CP4, C2, C6
#   C4_laplaciano = C4 - (FC4 + CP4 + C2 + C6) / 4

def aplicar_laplaciano_local(raw):
    """
    Aplica el filtro Laplaciano local a C3 y C4 restando el promedio
    de sus 4 electrodos vecinos ortogonales.

    Modifica el objeto Raw in-place reemplazando las señales de C3 y C4
    por sus versiones Laplacianas.

    Parámetros
    ----------
    raw : mne.io.Raw
        Objeto Raw con montaje 10-05 aplicado

    Retorna
    -------
    raw : mne.io.Raw
        Mismo objeto con C3 y C4 reemplazados por sus versiones Laplacianas
    """
    vecinos_C3 = ['FC3', 'CP3', 'C1', 'C5']
    vecinos_C4 = ['FC4', 'CP4', 'C2', 'C6']

    # Verificamos que todos los vecinos existan en los canales disponibles
    canales_disponibles = raw.ch_names

    for canal_central, vecinos in [('C3', vecinos_C3), ('C4', vecinos_C4)]:
        # Filtramos vecinos que efectivamente están en el registro
        vecinos_presentes = [v for v in vecinos if v in canales_disponibles]

        if len(vecinos_presentes) < len(vecinos):
            faltantes = set(vecinos) - set(vecinos_presentes)
            print(f"  Aviso: Faltan vecinos para {canal_central}: {faltantes}")

        # Índices de los canales
        idx_central = canales_disponibles.index(canal_central)
        idxs_vecinos = [canales_disponibles.index(v) for v in vecinos_presentes]

        # Extraemos los datos
        datos = raw.get_data()
        senal_central = datos[idx_central]
        senales_vecinos = datos[idxs_vecinos]

        # Aplicamos el Laplaciano: central - promedio(vecinos)
        promedio_vecinos = senales_vecinos.mean(axis=0)
        senal_laplaciana = senal_central - promedio_vecinos

        # Reemplazamos in-place la señal del canal central
        raw._data[idx_central] = senal_laplaciana

    return raw

# -----------------------------------------------------------------------------
# BLOQUE 3: Aplicación del filtro a todos los sujetos y visualización
# -----------------------------------------------------------------------------

def cargar_y_preprocesar_run(archivo):
    """
    Carga un archivo EDF, estandariza canales, aplica montaje 10-05 y
    devuelve el objeto Raw de MNE listo para análisis.
    """
    raw = mne.io.read_raw_edf(archivo, preload=True)
    mne.datasets.eegbci.standardize(raw)
    montaje = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montaje)
    return raw


def filtrar_bidireccional(datos, sos):
    """
    Aplica filtrado bidireccional (fase nula) usando sosfiltfilt.

    Parámetros
    ----------
    datos : array 1D o 2D
        Señal a filtrar. Si es 2D, cada fila es un canal.
    sos : array
        Coeficientes del filtro en formato SOS.

    Retorna
    -------
    datos_filtrados : array de la misma forma que datos
    """
    return signal.sosfiltfilt(sos, datos, axis=-1)


# Cargamos y procesamos: guardamos versión filtrada Y versión sin filtrar
# (necesarias para diferentes análisis)
datos_procesados = {}
datos_sin_filtrar = {}  # NUEVO: para la CWT

for suj in SUJETOS:
    print(f"\nProcesando sujeto S{suj:03d}...")
    datos_procesados[suj] = {'movimiento_real': [], 'imaginacion': []}
    datos_sin_filtrar[suj] = {'movimiento_real': [], 'imaginacion': []}

    for run in RUNS_TOTAL:
        idx_run = RUNS_TOTAL.index(run)
        archivo = archivos_por_sujeto[suj][idx_run]

        # Cargamos DOS veces: una para filtrar, otra para conservar cruda
        raw_filtrado = cargar_y_preprocesar_run(archivo)
        raw_crudo = cargar_y_preprocesar_run(archivo)

        # Filtrado espacial LAPLACIANO (ver Modificación 2 abajo)
        # Se aplica ANTES del filtrado temporal, a ambas versiones
        raw_filtrado = aplicar_laplaciano_local(raw_filtrado)
        raw_crudo = aplicar_laplaciano_local(raw_crudo)

        # Solo la versión "filtrada" recibe el pasabanda 8-30 Hz
        datos_crudos = raw_filtrado.get_data()
        datos_filtrados = filtrar_bidireccional(datos_crudos, sos)
        raw_filtrado._data = datos_filtrados

        # Clasificamos según condición
        if run in RUNS_MOVIMIENTO_REAL:
            datos_procesados[suj]['movimiento_real'].append(raw_filtrado)
            datos_sin_filtrar[suj]['movimiento_real'].append(raw_crudo)
        else:
            datos_procesados[suj]['imaginacion'].append(raw_filtrado)
            datos_sin_filtrar[suj]['imaginacion'].append(raw_crudo)

    print(f"  Runs movimiento real: {len(datos_procesados[suj]['movimiento_real'])}")
    print(f"  Runs imaginación: {len(datos_procesados[suj]['imaginacion'])}")

# Visualización: comparación de una época cruda vs filtrada
sujeto_demo = SUJETOS[0]
raw_demo = datos_procesados[sujeto_demo]['movimiento_real'][0]

# Recargamos el archivo original para comparación
archivo_demo = archivos_por_sujeto[sujeto_demo][0]
raw_original = cargar_y_preprocesar_run(archivo_demo)

# Tomamos 10 segundos de C3
idx_C3 = raw_demo.ch_names.index('C3')
n_muestras = int(10 * FS)
tiempo = np.arange(n_muestras) / FS

senal_cruda = raw_original.get_data()[idx_C3, :n_muestras] * 1e6
senal_filtrada = raw_demo.get_data()[idx_C3, :n_muestras] * 1e6

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(tiempo, senal_cruda, color='gray', linewidth=0.7, alpha=0.8)
axes[0].set_ylabel('Amplitud [μV]')
axes[0].set_title(f'Señal cruda - C3 - Sujeto {sujeto_demo}')

axes[1].plot(tiempo, senal_filtrada, color='#2a78d6', linewidth=1)
axes[1].set_ylabel('Amplitud [μV]')
axes[1].set_xlabel('Tiempo [s]')
axes[1].set_title(f'Señal filtrada 8-30 Hz - C3 - Sujeto {sujeto_demo}')

plt.tight_layout()
plt.savefig('fig_04_comparacion_cruda_filtrada.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Figura guardada: fig_04_comparacion_cruda_filtrada.png")


# -----------------------------------------------------------------------------
# BLOQUE 4: Extracción de épocas por condición y clase
#           con RECHAZO DE ÉPOCAS POR ARTEFACTOS
# -----------------------------------------------------------------------------
#
# METODOLOGÍA (actualizada):
#
# Ventanas de análisis:
#   - Baseline: -2 a 0 s (2 segundos previos al estímulo)
#   - Tarea (VENTANA CORREGIDA): 0.5 a 3.5 s post-estímulo
#     Se descartan los primeros 500 ms para evitar contaminación por el
#     Visual Evoked Potential (VEP) generado por el target visual.
#     Se descarta el último 500 ms para evitar componentes de anticipación
#     del fin del estímulo.
#
# Rechazo de épocas por artefactos:
#   Se descartan épocas cuya amplitud máxima absoluta en cualquier canal
#   supere el umbral UMBRAL_ARTEFACTO = 100 μV. Este criterio elimina
#   épocas contaminadas por parpadeos, movimientos oculares o corporales.
#   El mismo criterio se aplica a ambas condiciones para evitar sesgos.

# Definición de ventanas temporales
TMIN = -2.0     # segundos antes del evento
TMAX = 6.0      # segundos después del evento (mantenemos 4s para tener margen)

# Índices para separar baseline y tarea EFECTIVA
n_baseline = int(2 * FS)                       # 2 segundos de baseline
n_tarea_inicio = int(0.5 * FS)                 # descartar primeros 500 ms
n_tarea_fin = int(3.5 * FS)                    # descartar últimos 500 ms
# La ventana efectiva de tarea es de 0.5s a 3.5s = 3 segundos

# Umbral de rechazo de épocas (en Volts, ya que la señal está en Volts en MNE)
UMBRAL_ARTEFACTO_uV = 100
UMBRAL_ARTEFACTO_V = UMBRAL_ARTEFACTO_uV * 1e-6

# Estructura para almacenar épocas
epocas_todas = {}
contador_rechazos = {}

for suj in SUJETOS:
    epocas_todas[suj] = {
        'movimiento_real': {'izquierda': [], 'derecha': []},
        'imaginacion': {'izquierda': [], 'derecha': []}
    }
    contador_rechazos[suj] = {
        'movimiento_real': {'total': 0, 'rechazadas': 0},
        'imaginacion': {'total': 0, 'rechazadas': 0}
    }

    for condicion in ['movimiento_real', 'imaginacion']:
        for raw in datos_procesados[suj][condicion]:
            eventos, id_eventos = mne.events_from_annotations(raw)
            codigo_T1 = id_eventos.get('T1', None)
            codigo_T2 = id_eventos.get('T2', None)

            epocas_run = mne.Epochs(raw, eventos,
                                     event_id={'T1': codigo_T1, 'T2': codigo_T2},
                                     tmin=TMIN, tmax=TMAX,
                                     baseline=None, preload=True,
                                     picks=['C3', 'C4'])

            # Rechazo por amplitud en la ventana COMPLETA de la época
            epocas_T1 = epocas_run['T1'].get_data()
            epocas_T2 = epocas_run['T2'].get_data()

            for epocas_clase, clase in [(epocas_T1, 'izquierda'),
                                          (epocas_T2, 'derecha')]:
                n_total = epocas_clase.shape[0]
                contador_rechazos[suj][condicion]['total'] += n_total

                # Encontrar máximo absoluto por época en cualquier canal
                max_abs_por_epoca = np.max(np.abs(epocas_clase), axis=(1, 2))
                mascara_valida = max_abs_por_epoca < UMBRAL_ARTEFACTO_V

                n_rechazadas = n_total - np.sum(mascara_valida)
                contador_rechazos[suj][condicion]['rechazadas'] += n_rechazadas

                epocas_validas = epocas_clase[mascara_valida]
                epocas_todas[suj][condicion][clase].append(epocas_validas)

    # Concatenar épocas de todos los runs por clase
    for condicion in ['movimiento_real', 'imaginacion']:
        for clase in ['izquierda', 'derecha']:
            lista = epocas_todas[suj][condicion][clase]
            if len(lista) > 0:
                epocas_todas[suj][condicion][clase] = np.concatenate(lista, axis=0)
            else:
                epocas_todas[suj][condicion][clase] = np.array([])

# Impresión del resumen
print("\n" + "="*60)
print("RESUMEN DE ÉPOCAS (con rechazo por artefactos)")
print("="*60)
print(f"Umbral de rechazo: ±{UMBRAL_ARTEFACTO_uV} μV")
print(f"Ventana de tarea efectiva: 0.5 - 3.5 s post-estímulo")
for suj in SUJETOS:
    print(f"\nSujeto S{suj:03d}:")
    for condicion in ['movimiento_real', 'imaginacion']:
        total = contador_rechazos[suj][condicion]['total']
        rechaz = contador_rechazos[suj][condicion]['rechazadas']
        pct = rechaz / total * 100 if total > 0 else 0
        n_izq = epocas_todas[suj][condicion]['izquierda'].shape[0]
        n_der = epocas_todas[suj][condicion]['derecha'].shape[0]
        etiqueta = 'Movimiento real' if condicion == 'movimiento_real' else 'Imaginación'
        print(f"  {etiqueta}: {n_izq} izq, {n_der} der | "
              f"Rechazadas: {rechaz}/{total} ({pct:.1f}%)")

print("\n✓ Extracción de épocas completada")

# Ventanas efectivas de análisis
n_tarea_inicio = int(0.5 * FS)                 # 0.5 s post-estímulo
n_tarea_fin = int(3.5 * FS)                    # 3.5 s post-estímulo
n_ers_inicio = int(4.5 * FS)                   # 4.5 s post-estímulo (ERS)
n_ers_fin = int(6.0 * FS)                      # 6.0 s post-estímulo (ERS)

# -----------------------------------------------------------------------------
# BLOQUE 4.5: Exclusión de sujetos con calidad de datos insuficiente
# -----------------------------------------------------------------------------
#
# JUSTIFICACIÓN:
#
# Se aplica un criterio de exclusión automática para sujetos cuya tasa de
# épocas válidas (post-rechazo por artefactos) sea inferior al 40% en
# cualquiera de las dos condiciones. Este criterio es coherente con la
# práctica estándar en estudios de EEG: registros con más del 60% de
# épocas contaminadas son considerados de calidad insuficiente para
# análisis estadístico confiable.
#
# El criterio se aplica DESPUÉS de haber definido la lista original de
# sujetos, y de forma independiente al análisis posterior, para evitar
# sesgo de selección basado en los resultados.

UMBRAL_EPOCAS_VALIDAS = 0.40  # 40% mínimo de épocas válidas por condición

sujetos_excluidos = []
for suj in SUJETOS.copy():
    excluir = False
    for condicion in ['movimiento_real', 'imaginacion']:
        total = contador_rechazos[suj][condicion]['total']
        rechaz = contador_rechazos[suj][condicion]['rechazadas']
        tasa_validas = (total - rechaz) / total if total > 0 else 0

        if tasa_validas < UMBRAL_EPOCAS_VALIDAS:
            excluir = True
            break

    if excluir:
        sujetos_excluidos.append(suj)

# Actualizamos la lista de sujetos válidos
SUJETOS_ORIGINALES = SUJETOS.copy()
SUJETOS = [s for s in SUJETOS if s not in sujetos_excluidos]

print("\n" + "="*60)
print("EXCLUSIÓN DE SUJETOS POR CALIDAD DE DATOS")
print("="*60)
print(f"Criterio: <{UMBRAL_EPOCAS_VALIDAS*100:.0f}% épocas válidas en alguna condición")
print(f"Sujetos originales ({len(SUJETOS_ORIGINALES)}): {SUJETOS_ORIGINALES}")
print(f"Sujetos excluidos ({len(sujetos_excluidos)}): {sujetos_excluidos}")
print(f"Sujetos incluidos en análisis ({len(SUJETOS)}): {SUJETOS}")

# Definición de bandas de interés
BANDA_MU = (8, 12)      # Hz
BANDA_BETA = (13, 30)   # Hz

# Parámetros de Welch
NPERSEG_WELCH = int(1 * FS)
NOVERLAP_WELCH = NPERSEG_WELCH // 2

# -----------------------------------------------------------------------------
# BLOQUE 4.7: Determinación de bandas Mu específicas por sujeto
# -----------------------------------------------------------------------------
#
# METODOLOGÍA:
#
# Siguiendo Pfurtscheller et al. (2006), la banda Mu reactiva presenta
# variabilidad interindividual significativa (media ~11.7 Hz ± 0.4).
# Usar la banda canónica 8-12 Hz para todos los sujetos puede diluir el
# ERD real en sujetos cuyo pico se encuentre fuera de ese rango.
#
# Método implementado (basado en reactividad espectral):
#   1. Calcular PSD promedio en baseline (-2 a 0 s) y en tarea (0.5-3.5 s)
#      para cada sujeto en el canal C3 (contralateral a mano derecha).
#   2. Calcular la diferencia P_tarea - P_baseline en el rango 6-15 Hz.
#   3. Identificar la frecuencia donde la diferencia es más NEGATIVA
#      (mayor caída = mayor reactividad al estímulo).
#   4. Definir banda Mu específica como pico ± 2 Hz.
#
# La banda Beta se mantiene canónica (13-30 Hz) porque su variabilidad
# interindividual es menor y es más difícil identificar un pico único.

def determinar_banda_mu_sujeto(sujeto, epocas_todas, fs, n_baseline_muestras,
                                  nperseg, noverlap, rango_busqueda=(6, 15),
                                  ancho_banda=2):
    """
    Determina la banda Mu específica para un sujeto usando su reactividad
    espectral (diferencia baseline vs tarea).

    Retorna
    -------
    banda_mu : tuple (f_min, f_max)
        Banda Mu específica del sujeto
    freq_pico : float
        Frecuencia del pico de reactividad identificado
    """
    # Promediamos épocas de movimiento real, mano derecha, canal C3
    # (donde el ERD debe ser más claro)
    epocas_C3 = epocas_todas[sujeto]['movimiento_real']['derecha'][:, 0, :] * 1e6

    if epocas_C3.shape[0] == 0:
        # Fallback: banda canónica si no hay datos suficientes
        return (8, 12), 10

    diferencias_freq = []
    frecs_ref = None

    for i in range(epocas_C3.shape[0]):
        senal_baseline = epocas_C3[i, :n_baseline_muestras]
        senal_tarea = epocas_C3[i, n_baseline_muestras +
                                    int(0.5*fs):n_baseline_muestras + int(3.5*fs)]

        f, psd_b = signal.welch(senal_baseline, fs=fs, window='hann',
                                 nperseg=nperseg, noverlap=noverlap)
        _, psd_t = signal.welch(senal_tarea, fs=fs, window='hann',
                                 nperseg=nperseg, noverlap=noverlap)

        if frecs_ref is None:
            frecs_ref = f

        diferencias_freq.append(psd_t - psd_b)

    # Promedio de diferencias entre épocas
    diff_promedio = np.mean(diferencias_freq, axis=0)

    # Buscar el pico negativo más pronunciado en el rango de búsqueda
    mascara_busqueda = (frecs_ref >= rango_busqueda[0]) & (frecs_ref <= rango_busqueda[1])
    idx_relativos = np.where(mascara_busqueda)[0]

    if len(idx_relativos) == 0:
        return (8, 12), 10

    # Índice del mínimo (mayor caída de potencia)
    idx_min_local = np.argmin(diff_promedio[mascara_busqueda])
    idx_pico = idx_relativos[idx_min_local]
    freq_pico = frecs_ref[idx_pico]

    # Banda Mu específica: pico ± ancho_banda
    banda_mu = (max(6, freq_pico - ancho_banda),
                 min(15, freq_pico + ancho_banda))

    return banda_mu, freq_pico


# Calculamos las bandas específicas para cada sujeto
bandas_mu_por_sujeto = {}
print("\n" + "="*60)
print("DETERMINACIÓN DE BANDAS Mu ESPECÍFICAS POR SUJETO")
print("="*60)
print(f"{'Sujeto':<10} {'Freq pico [Hz]':<18} {'Banda Mu [Hz]':<20}")
print("-"*50)

for suj in SUJETOS:
    banda_mu, freq_pico = determinar_banda_mu_sujeto(
        suj, epocas_todas, FS, n_baseline,
        NPERSEG_WELCH, NOVERLAP_WELCH
    )
    bandas_mu_por_sujeto[suj] = banda_mu
    print(f"S{suj:03d}        {freq_pico:>6.1f}              "
          f"({banda_mu[0]:.1f}, {banda_mu[1]:.1f})")

# Estadísticas de las bandas encontradas
picos_encontrados = [determinar_banda_mu_sujeto(s, epocas_todas, FS,
                       n_baseline, NPERSEG_WELCH, NOVERLAP_WELCH)[1]
                       for s in SUJETOS]
print(f"\nEstadísticas del pico de reactividad Mu:")
print(f"  Media: {np.mean(picos_encontrados):.2f} Hz")
print(f"  Desvío estándar: {np.std(picos_encontrados):.2f} Hz")
print(f"  Rango: [{min(picos_encontrados):.1f}, {max(picos_encontrados):.1f}] Hz")

# Comparación con la banda canónica
print(f"\nBanda Mu canónica: (8, 12) Hz")
print(f"Referencia literatura (Pfurtscheller 2006): 11.7 ± 0.4 Hz")

# -----------------------------------------------------------------------------
# BLOQUE 5: Cálculo del ERD% por época mediante método de Welch
# -----------------------------------------------------------------------------
#
# METODOLOGÍA:
#
# Fórmula del ERD% (Pfurtscheller & Lopes da Silva, 1999):
#
#     ERD%(banda) = (P_tarea - P_baseline) / P_baseline * 100
#
# Donde:
#   P_baseline = potencia promedio en la banda durante la ventana de reposo
#   P_tarea = potencia promedio en la banda durante la ventana de estímulo
#
# Interpretación:
#   ERD% < 0 -> caída de potencia (DESINCRONIZACIÓN, activación cortical)
#   ERD% > 0 -> aumento de potencia (SINCRONIZACIÓN, deactivación/rebote)
#   ERD% ≈ 0 -> sin cambio significativo
#
# Cálculo por época:
#   Para cada época se aplica Welch por separado en las ventanas baseline
#   y tarea, se integra la PSD en las bandas Mu (8-12 Hz) y Beta (13-30 Hz),
#   y se aplica la fórmula anterior. Esto genera un vector de ERD% por
#   condición, sujeto, banda y electrodo, que luego se resume estadísticamente.
#
# Parámetros de Welch:
#   nperseg = 1 * FS = 160 muestras (ventana de 1 segundo)
#     Elegido para tener resolución frecuencial de 1 Hz, suficiente para
#     separar Mu de Beta. Con épocas de 2s (baseline) y 4s (tarea) esto
#     nos da 3 y 7 sub-ventanas respectivamente con 50% de overlap.
#   noverlap = nperseg // 2 (50% de solapamiento estándar)
#   window = 'hann' (ventana estándar para EEG)


def calcular_potencia_banda(senal, fs, banda, nperseg, noverlap):
    """
    Calcula la potencia promedio en una banda de frecuencia usando Welch.

    Parámetros
    ----------
    senal : array 1D
        Señal temporal
    fs : float
        Frecuencia de muestreo
    banda : tuple (f_min, f_max)
        Límites de la banda en Hz
    nperseg, noverlap : int
        Parámetros del método de Welch

    Retorna
    -------
    potencia : float
        Potencia integrada en la banda [μV²]
    """
    f, psd = signal.welch(senal, fs=fs, window='hann',
                          nperseg=nperseg, noverlap=noverlap)
    # Máscara para la banda de interés
    mascara = (f >= banda[0]) & (f <= banda[1])
    # Integramos la PSD en la banda (aproximación por suma × delta_f)
    delta_f = f[1] - f[0]
    potencia = np.sum(psd[mascara]) * delta_f
    return potencia


def calcular_erd_ers_epoca(epoca, fs, banda, n_baseline,
                             n_tarea_inicio, n_tarea_fin,
                             n_ers_inicio, n_ers_fin,
                             nperseg, noverlap):
    """
    Calcula ERD% (durante el estímulo) y ERS% (post-estímulo).

    Retorna
    -------
    erd_pct : float, ERS_pct : float
    """
    senal_baseline = epoca[:n_baseline]
    senal_tarea = epoca[n_baseline + n_tarea_inicio : n_baseline + n_tarea_fin]
    senal_ers = epoca[n_baseline + n_ers_inicio : n_baseline + n_ers_fin]

    p_baseline = calcular_potencia_banda(senal_baseline, fs, banda,
                                          nperseg, noverlap)
    p_tarea = calcular_potencia_banda(senal_tarea, fs, banda,
                                       nperseg, noverlap)
    p_ers = calcular_potencia_banda(senal_ers, fs, banda,
                                     nperseg, noverlap)

    erd_pct = (p_tarea - p_baseline) / p_baseline * 100
    ers_pct = (p_ers - p_baseline) / p_baseline * 100
    return erd_pct, ers_pct

# Estructura para almacenar todos los valores de ERD%
# Formato: filas de un DataFrame para facilitar los boxplots después
# En el loop de cálculo del ERD%, cambiar la llamada a:
registros_erd = []
for suj in SUJETOS:
    for condicion in ['movimiento_real', 'imaginacion']:
        for clase in ['izquierda', 'derecha']:
            epocas = epocas_todas[suj][condicion][clase]
            if epocas.shape[0] == 0:
                continue
            n_epocas = epocas.shape[0]

            for i_epoca in range(n_epocas):
                for i_canal, nombre_canal in enumerate(['C3', 'C4']):
                    senal_canal = epocas[i_epoca, i_canal, :] * 1e6

                    for nombre_banda in ['Mu', 'Beta']:
                        if nombre_banda == 'Mu':
                            banda = bandas_mu_por_sujeto[suj]  # Banda específica del sujeto
                        else:
                            banda = BANDA_BETA  # Beta se mantiene canónica
                        erd, ers = calcular_erd_ers_epoca(
                            senal_canal, FS, banda,
                            n_baseline, n_tarea_inicio, n_tarea_fin,
                            n_ers_inicio, n_ers_fin,
                            NPERSEG_WELCH, NOVERLAP_WELCH
                        )
                        registros_erd.append({
                            'sujeto': suj,
                            'condicion': condicion,
                            'clase': clase,
                            'canal': nombre_canal,
                            'banda': nombre_banda,
                            'epoca': i_epoca,
                            'ERD_pct': erd,
                            'ERS_pct': ers
                        })

df_erd = pd.DataFrame(registros_erd)

print(f"\n✓ ERD% y ERS% calculados para {len(df_erd)} combinaciones")
print(f"\nMedianas por condición y banda (canal C3):")
resumen_C3 = df_erd[df_erd['canal']=='C3'].groupby(
    ['condicion', 'banda'])[['ERD_pct', 'ERS_pct']].median().round(2)
print(resumen_C3)


# -----------------------------------------------------------------------------
# BLOQUE 6: Espectrograma tiempo-frecuencia con CWT-Morlet
# -----------------------------------------------------------------------------
#
# JUSTIFICACIÓN DEL MÉTODO:
#
# La Transformada Wavelet Continua (CWT) con wavelet de Morlet es el
# estándar en la literatura de EEG para analizar la dinámica temporal
# de eventos oscilatorios como el ERD/ERS (Tallon-Baudry et al., 1997).
#
# IMPORTANTE: La CWT se aplica sobre la señal CRUDA (solo con Laplaciano
# aplicado, sin filtrado temporal 8-30 Hz). La wavelet de Morlet actúa
# ella misma como un banco de filtros pasabanda a diferentes frecuencias,
# por lo que aplicar un pasabanda previo genera artefactos matemáticos
# en los bordes del rango de análisis.
#
# La normalización se hace en decibeles (logratio), que es el estándar
# en MNE y en la literatura moderna de análisis tiempo-frecuencia. Es
# más estable numéricamente que el cambio porcentual y produce escalas
# de colores balanceadas.

def calcular_cwt_promedio(epocas, fs, freqs_deseadas, wavelet='cmor1.5-1.0'):
    """
    Calcula el scalograma CWT promedio a través de épocas.

    IMPORTANTE: Esta función debe recibir épocas SIN FILTRAR en 8-30 Hz.
    La wavelet de Morlet actúa como un banco de filtros en sí misma, así
    que aplicar un pasabanda previo a la CWT produce artefactos en los
    bordes del rango de análisis (divisiones de ruido casi-cero).

    Parámetros
    ----------
    epocas : array 3D (n_epocas, n_canales, n_muestras) o 2D (n_epocas, n_muestras)
        Épocas sin filtrar (con todas las frecuencias originales)
    freqs_deseadas : array
        Frecuencias de análisis en Hz

    Retorna
    -------
    scalograma : array 2D (n_freqs, n_muestras)
        Potencia promedio |CWT|² a través de épocas
    """
    escalas = pywt.central_frequency(wavelet) * fs / freqs_deseadas

    n_epocas = epocas.shape[0]
    n_muestras = epocas.shape[-1]

    potencia_acumulada = np.zeros((len(freqs_deseadas), n_muestras))

    for i in range(n_epocas):
        if epocas.ndim == 3:
            senal = epocas[i, 0, :]
        else:
            senal = epocas[i, :]
        coefs, _ = pywt.cwt(senal, escalas, wavelet, sampling_period=1/fs)
        potencia_acumulada += np.abs(coefs) ** 2

    return potencia_acumulada / n_epocas


def normalizar_por_baseline_db(scalograma, n_baseline_muestras):
    """
    Normaliza el scalograma como cambio en decibeles respecto al baseline
    usando el método logratio (estándar en MNE y en la literatura moderna).

    Fórmula: dB(t, f) = 10 * log10(P(t, f) / P_baseline(f))

    Ventajas sobre la normalización porcentual:
    - Simetría: valores positivos y negativos tienen misma escala visual.
    - Estabilidad numérica: el log comprime valores extremos.
    - Comparabilidad entre estudios: es la métrica estándar.

    Interpretación:
    - Valores negativos: caída de potencia respecto al baseline (ERD).
    - Valores positivos: aumento de potencia (ERS).
    - Cada -3 dB equivale aproximadamente a la mitad de la potencia.

    Retorna
    -------
    scalograma_db : array del mismo tamaño
        Valores en dB respecto al baseline
    """
    baseline_por_freq = scalograma[:, :n_baseline_muestras].mean(axis=1, keepdims=True)
    # Evitar log de cero agregando epsilon
    epsilon = 1e-20
    scalograma_db = 10 * np.log10((scalograma + epsilon) / (baseline_por_freq + epsilon))
    return scalograma_db


# Frecuencias de análisis para la CWT
FREQS_CWT = np.arange(4, 41, 1)  # de 4 a 40 Hz cada 1 Hz


def extraer_epocas_de_raw(raw_lista, tmin, tmax, canal='C3'):
    """
    Extrae épocas de una lista de objetos Raw usando MNE.

    Retorna épocas concatenadas de todos los runs, separadas por clase T1 y T2.
    """
    epocas_T1_lista = []
    epocas_T2_lista = []
    idx_canal = None

    for raw in raw_lista:
        eventos, id_eventos = mne.events_from_annotations(raw)
        codigo_T1 = id_eventos.get('T1', None)
        codigo_T2 = id_eventos.get('T2', None)

        epocas_run = mne.Epochs(raw, eventos,
                                 event_id={'T1': codigo_T1, 'T2': codigo_T2},
                                 tmin=tmin, tmax=tmax,
                                 baseline=None, preload=True,
                                 picks=[canal])

        epocas_T1_lista.append(epocas_run['T1'].get_data()[:, 0, :])
        epocas_T2_lista.append(epocas_run['T2'].get_data()[:, 0, :])

    epocas_T1 = np.concatenate(epocas_T1_lista, axis=0) if epocas_T1_lista else np.array([])
    epocas_T2 = np.concatenate(epocas_T2_lista, axis=0) if epocas_T2_lista else np.array([])

    return epocas_T1, epocas_T2


# Elegimos un sujeto representativo para el scalograma
sujeto_cwt = SUJETOS[0]

# Extraemos épocas DE LA SEÑAL SIN FILTRAR (solo con Laplaciano)
_, epocas_real_C3 = extraer_epocas_de_raw(
    datos_sin_filtrar[sujeto_cwt]['movimiento_real'],
    TMIN, TMAX, canal='C3'
)
_, epocas_imag_C3 = extraer_epocas_de_raw(
    datos_sin_filtrar[sujeto_cwt]['imaginacion'],
    TMIN, TMAX, canal='C3'
)

# Convertir a microVolts
epocas_real_C3 = epocas_real_C3 * 1e6
epocas_imag_C3 = epocas_imag_C3 * 1e6

print(f"\nCalculando CWT para sujeto {sujeto_cwt} en C3...")
print(f"  Épocas movimiento real: {epocas_real_C3.shape[0]}")
print(f"  Épocas imaginación: {epocas_imag_C3.shape[0]}")

scalo_real = calcular_cwt_promedio(epocas_real_C3, FS, FREQS_CWT)
scalo_imag = calcular_cwt_promedio(epocas_imag_C3, FS, FREQS_CWT)

# Normalización en decibeles respecto al baseline
scalo_real_db = normalizar_por_baseline_db(scalo_real, n_baseline)
scalo_imag_db = normalizar_por_baseline_db(scalo_imag, n_baseline)

# Vector de tiempo para el eje X (centrado en el estímulo)
tiempo_epoca = np.arange(scalo_real.shape[1]) / FS + TMIN

# Visualización de los scalogramas
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# Escala común de colores para comparación justa (en dB, típicamente ±6 dB)
vmax = 6
vmin = -vmax

im1 = axes[0].pcolormesh(tiempo_epoca, FREQS_CWT, scalo_real_db,
                          shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
axes[0].axvline(4, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[0].axhline(8, color='green', linestyle=':', alpha=0.5)
axes[0].axhline(12, color='green', linestyle=':', alpha=0.5)
axes[0].axhline(13, color='orange', linestyle=':', alpha=0.5)
axes[0].axhline(30, color='orange', linestyle=':', alpha=0.5)
axes[0].set_title(f'Movimiento REAL - Mano derecha - C3 (Sujeto {sujeto_cwt})')
axes[0].set_xlabel('Tiempo respecto al estímulo [s]')
axes[0].set_ylabel('Frecuencia [Hz]')

im2 = axes[1].pcolormesh(tiempo_epoca, FREQS_CWT, scalo_imag_db,
                          shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].axvline(4, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].axhline(8, color='green', linestyle=':', alpha=0.5)
axes[1].axhline(12, color='green', linestyle=':', alpha=0.5)
axes[1].axhline(13, color='orange', linestyle=':', alpha=0.5)
axes[1].axhline(30, color='orange', linestyle=':', alpha=0.5)
axes[1].set_title(f'IMAGINACIÓN - Mano derecha - C3 (Sujeto {sujeto_cwt})')
axes[1].set_xlabel('Tiempo respecto al estímulo [s]')

# Barra de color común
cbar = fig.colorbar(im2, ax=axes, orientation='vertical', pad=0.02,
                     label='Cambio de potencia respecto al baseline [dB]')

plt.savefig('fig_05_scalogramas_CWT.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_05_scalogramas_CWT.png")

# -----------------------------------------------------------------------------
# BLOQUE 6.5: Curvas ERD/ERS vs tiempo (metodología Band Power de Pfurtscheller)
# -----------------------------------------------------------------------------
#
# METODOLOGÍA:
#
# Método clásico de Pfurtscheller para el cálculo de curvas ERD/ERS
# (Pfurtscheller & Lopes da Silva, 1999):
#
#   1. Filtrar la señal en la banda estrecha de interés (Mu o Beta)
#      Filtro Butterworth pasabanda con transición angosta.
#   2. Elevar al cuadrado muestra a muestra -> potencia instantánea.
#   3. Suavizar con ventana móvil (típicamente 250 ms).
#   4. Promediar a través de épocas de la misma clase.
#   5. Normalizar respecto al baseline como cambio porcentual:
#         ERD%(t) = (P(t) - P_baseline) / P_baseline * 100
#
# Diferencias esperadas entre bandas (Pfurtscheller & Neuper, 2001):
#   - Banda Mu: ERD sostenido durante todo el movimiento/imaginación.
#   - Banda Beta: ERD durante el movimiento SEGUIDO de un fuerte rebote
#     ERS post-movimiento ("beta rebound"), marcador clásico de la
#     finalización del proceso motor.
#
# Diferencias esperadas entre condiciones:
#   - Movimiento real: ERD más profundo y variabilidad menor.
#   - Imaginación motora: ERD menos pronunciado y más variable.

# Parámetros del método Band Power
VENTANA_SUAVIZADO_MS = 250   # ms - ventana móvil para el suavizado
n_ventana_suavizado = int(VENTANA_SUAVIZADO_MS / 1000 * FS)  # muestras


def diseno_filtro_banda_estrecha(banda, fs, gpass_final=3, gstop_final=40):
    """
    Diseña un filtro Butterworth angosto para una banda específica.
    Compensa por filtrado bidireccional (mitad de valores en diseño).
    """
    fpass = np.array(banda, dtype=float)
    ancho_transicion = 2  # Hz a cada lado
    fstop = np.array([banda[0] - ancho_transicion, banda[1] + ancho_transicion])
    sos_banda = signal.iirdesign(
        wp=fpass, ws=fstop,
        gpass=gpass_final / 2, gstop=gstop_final / 2,
        ftype='butter', output='sos', fs=fs
    )
    return sos_banda


def calcular_curva_erd(epocas, fs, banda, n_baseline_muestras,
                       n_ventana_suav):
    """
    Curva ERD/ERS por Band Power (Pfurtscheller & Lopes da Silva, 1999).

    Orden correcto:
      1. Filtrar en banda estrecha (bidireccional, fase nula).
      2. Elevar al cuadrado -> potencia instantánea por época.
      3. Suavizar con ventana móvil.
      4. Promediar potencia entre épocas -> P̄(t).
      5. Baseline escalar único: R = mean(P̄(t)) en la ventana baseline.
      6. Normalizar UNA vez: ERD%(t) = (P̄(t) - R) / R * 100.

    Para la banda de confianza, se normalizan también las épocas individuales
    con el MISMO R (no con el baseline de cada época), lo que elimina los
    picos extremos por baselines pequeños en épocas particulares.

    Retorna
    -------
    curva_promedio : array 1D, ERD%(t) promedio.
    curva_desvio   : array 1D, desvío entre épocas de las curvas ERD%_i(t).
    curvas_epoca   : array 2D (n_epocas, n_muestras), ERD%_i(t) por época
                     (útil para bootstrap o percentiles si se prefiere después).
    """
    # 1. Filtro angosto bidireccional
    sos_banda = diseno_filtro_banda_estrecha(banda, fs)
    epocas_filt = signal.sosfiltfilt(sos_banda, epocas, axis=-1)

    # 2. Potencia instantánea
    potencia_inst = epocas_filt ** 2

    # 3. Suavizado con ventana rectangular
    kernel = np.ones(n_ventana_suav) / n_ventana_suav
    potencia_suave = np.array([
        np.convolve(pot, kernel, mode='same') for pot in potencia_inst
    ])

    # 4. Promedio entre épocas -> curva de potencia promedio
    potencia_promedio = potencia_suave.mean(axis=0)

    # 5. Baseline escalar único
    baseline_medio = potencia_promedio[:n_baseline_muestras].mean()

    # 6. Curva ERD% promedio
    curva_promedio = (potencia_promedio - baseline_medio) / baseline_medio * 100

    # Curvas ERD% por época usando el MISMO baseline global
    curvas_epoca = (potencia_suave - baseline_medio) / baseline_medio * 100
    curva_desvio = curvas_epoca.std(axis=0)

    return curva_promedio, curva_desvio, curvas_epoca


def calcular_curvas_por_sujetos(sujetos, condicion, clase, canal_idx,
                                  banda_nombre, fs, n_baseline_muestras, n_ventana_suav):
    """
    Calcula la curva ERD promedio para cada sujeto y luego promedia entre
    sujetos para obtener la curva final con desvío interindividual.

    A diferencia de la versión anterior, ahora recibe el NOMBRE de la banda
    ('Mu' o 'Beta') en lugar del rango, para poder usar bandas específicas
    por sujeto en el caso de Mu.
    """
    curvas_por_sujeto = []
    for suj in sujetos:
        # Selección de banda: específica por sujeto para Mu, canónica para Beta
        if banda_nombre == 'Mu':
            banda = bandas_mu_por_sujeto[suj]
        else:
            banda = BANDA_BETA

        epocas = epocas_todas[suj][condicion][clase][:, canal_idx, :] * 1e6
        curva_suj, _, _ = calcular_curva_erd(
            epocas, fs, banda, n_baseline_muestras, n_ventana_suav
            )
        curvas_por_sujeto.append(curva_suj)

        curvas_por_sujeto = np.array(curvas_por_sujeto)
        curva_media = np.median(curvas_por_sujeto, axis=0)
        q25 = np.percentile(curvas_por_sujeto, 25, axis=0)
        q75 = np.percentile(curvas_por_sujeto, 75, axis=0)
        curva_std = (q75 - q25) / 2  # semi-rango intercuartil
    
        return curva_media, curva_std


# Cálculo de curvas para todas las combinaciones
# Estructura: {(canal, banda, condicion, clase): (media, std)}
curvas_erd = {}

print("\nCalculando curvas ERD/ERS por Band Power...")
for canal_idx, canal_nombre in enumerate(['C3', 'C4']):
    for banda_nombre in ['Mu', 'Beta']:
        for condicion in ['movimiento_real', 'imaginacion']:
            for clase in ['izquierda', 'derecha']:
                media, std = calcular_curvas_por_sujetos(
                    SUJETOS, condicion, clase, canal_idx,
                    banda_nombre, FS, n_baseline, n_ventana_suavizado
                )
                curvas_erd[(canal_nombre, banda_nombre, condicion, clase)] = (media, std)
print(f"  {len(curvas_erd)} curvas calculadas")

# Vector de tiempo común
n_muestras_epoca = curvas_erd[('C3', 'Mu', 'movimiento_real', 'izquierda')][0].shape[0]
tiempo_curva = np.arange(n_muestras_epoca) / FS + TMIN

# Recortar bordes para eliminar transients del filtro + edge effects del suavizado
MARGEN_BORDE_S = 0.5  # 500 ms a cada extremo
n_margen = int(MARGEN_BORDE_S * FS)

tiempo_curva_recortado = tiempo_curva[n_margen:-n_margen]

curvas_erd_recortadas = {}
for clave, (media, std) in curvas_erd.items():
    curvas_erd_recortadas[clave] = (media[n_margen:-n_margen],
                                     std[n_margen:-n_margen])

# ---- Visualización estilo Pfurtscheller ----
# En lugar de un panel 2x2, generamos dos figuras separadas (una por banda)
# para mejorar la legibilidad al insertarlas en el informe.

estilos = {
    ('movimiento_real', 'izquierda'):
        {'color': '#1976d2', 'linestyle': '-', 'label': 'Real - Izquierda'},
    ('movimiento_real', 'derecha'):
        {'color': '#e34948', 'linestyle': '-', 'label': 'Real - Derecha'},
    ('imaginacion', 'izquierda'):
        {'color': '#1976d2', 'linestyle': '--', 'label': 'Imaginación - Izquierda'},
    ('imaginacion', 'derecha'):
        {'color': '#e34948', 'linestyle': '--', 'label': 'Imaginación - Derecha'},
}


def graficar_curvas_por_banda(banda_nombre, banda_rango, nombre_archivo):
    """Genera una figura con paneles C3 y C4 lado a lado para una banda dada."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    for i_col, canal_nombre in enumerate(['C3', 'C4']):
        ax = axes[i_col]

        for (condicion, clase), estilo in estilos.items():
            media, std = curvas_erd[(canal_nombre, banda_nombre, condicion, clase)]
            ax.plot(tiempo_curva, media, **estilo, linewidth=1.8)
            ax.fill_between(tiempo_curva,
                             media - std, media + std,
                             color=estilo['color'], alpha=0.12)

        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(4, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(f'Canal {canal_nombre}', fontsize=13)
        ax.set_xlabel('Tiempo respecto al estímulo [s]', fontsize=11)
        if i_col == 0:
            ax.set_ylabel('ERD/ERS [%]', fontsize=11)
        ax.grid(True, alpha=0.3)

    # Leyenda única fuera de los subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=10)

    titulo_banda = f'{banda_nombre} ({banda_rango[0]}-{banda_rango[1]} Hz)'
    plt.suptitle(f'Curvas ERD/ERS vs tiempo - Banda {titulo_banda}\n'
                  f'(promedio entre {len(SUJETOS)} sujetos, banda de confianza = ±1 desvío)',
                  fontsize=13, y=1.09)
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Figura guardada: {nombre_archivo}")


# Generamos las dos figuras separadas
# Para la banda Mu usamos el rango medio de las bandas específicas
graficar_curvas_por_banda('Mu', (8, 15), 'fig_06b_curvas_ERD_tiempo_Mu.png')
graficar_curvas_por_banda('Beta', BANDA_BETA, 'fig_06b_curvas_ERD_tiempo_Beta.png')

# -----------------------------------------------------------------------------
# BLOQUE 4.8: Visualización del filtrado por bandas Mu y Beta
# -----------------------------------------------------------------------------
#
# OBJETIVO:
#
# Ilustrar visualmente el efecto del filtrado angosto por bandas Mu y Beta
# sobre una época representativa. Esta figura demuestra dos conceptos clave:
#
# 1) La utilidad del filtrado por banda estrecha: aísla oscilaciones
#    específicas que están mezcladas en la señal ancha 8-30 Hz.
#
# 2) La manifestación temporal del ERD: durante la ventana de tarea
#    (después del estímulo), la amplitud de las oscilaciones en la banda
#    Mu y/o Beta se reduce visiblemente respecto al baseline, lo cual es
#    la manifestación temporal directa del fenómeno de desincronización.

# Elegimos un sujeto con ERD claro y una época representativa
sujeto_demo_filtro = 42  # el sujeto representativo del análisis anterior

# Extraemos una época de movimiento real, mano derecha (donde ERD es más marcado en C3)
epocas_demo = epocas_todas[sujeto_demo_filtro]['movimiento_real']['derecha']
# Tomamos la primera época, canal C3 (índice 0), y convertimos a μV
epoca_C3 = epocas_demo[0, 0, :] * 1e6

# Banda Mu específica de este sujeto
banda_mu_demo = bandas_mu_por_sujeto[sujeto_demo_filtro]

# Diseñamos filtros angostos para cada banda
sos_mu = diseno_filtro_banda_estrecha(banda_mu_demo, FS)
sos_beta = diseno_filtro_banda_estrecha(BANDA_BETA, FS)

# Aplicamos filtrado bidireccional
senal_mu = signal.sosfiltfilt(sos_mu, epoca_C3)
senal_beta = signal.sosfiltfilt(sos_beta, epoca_C3)

# Vector de tiempo (la época va de TMIN a TMAX)
tiempo_demo = np.arange(len(epoca_C3)) / FS + TMIN

# Visualización
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# Panel 1: Señal cruda (con Laplaciano y filtro amplio 8-30 Hz ya aplicados)
axes[0].plot(tiempo_demo, epoca_C3, color='#424242', linewidth=0.9)
axes[0].axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7,
                label='Inicio estímulo')

axes[0].axvspan(TMIN, 0, alpha=0.08, color='blue', label='Baseline')
axes[0].axvspan(0.5, 3.5, alpha=0.08, color='orange', label='Ventana de tarea')
axes[0].set_ylabel('Amplitud [μV]')
axes[0].set_title(f'Señal filtrada 8-30 Hz (banda amplia) - C3 - Sujeto S{sujeto_demo_filtro:03d}')
axes[0].legend(loc='upper right', fontsize=9, ncol=4)
axes[0].grid(True, alpha=0.3)

# Panel 2: Filtrado en banda Mu específica del sujeto
axes[1].plot(tiempo_demo, senal_mu, color='#2e7d32', linewidth=1.2)
axes[1].axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].axvspan(TMIN, 0, alpha=0.08, color='blue')
axes[1].axvspan(0.5, 3.5, alpha=0.08, color='orange')
axes[1].set_ylabel('Amplitud [μV]')
axes[1].set_title(f'Filtrado en banda Mu específica del sujeto '
                   f'({banda_mu_demo[0]:.0f}-{banda_mu_demo[1]:.0f} Hz)')
axes[1].grid(True, alpha=0.3)

# Panel 3: Filtrado en banda Beta canónica
axes[2].plot(tiempo_demo, senal_beta, color='#e65100', linewidth=1.2)
axes[2].axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[2].axvspan(TMIN, 0, alpha=0.08, color='blue')
axes[2].axvspan(0.5, 3.5, alpha=0.08, color='orange')
axes[2].set_ylabel('Amplitud [μV]')
axes[2].set_xlabel('Tiempo respecto al estímulo [s]')
axes[2].set_title(f'Filtrado en banda Beta ({BANDA_BETA[0]}-{BANDA_BETA[1]} Hz)')
axes[2].grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('fig_12_filtrado_bandas.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_12_filtrado_bandas.png")

# -----------------------------------------------------------------------------
# BLOQUE 6.7: Curvas ERD/ERS de un sujeto representativo
# -----------------------------------------------------------------------------
#
# Como hacen Pfurtscheller y Neuper en sus papers ("single representative
# subject"), mostramos el análisis de un sujeto individual con buena
# calidad de datos para visualizar el ERD/ERS con menor ruido que en el
# promedio interindividual.

# Identificamos el sujeto con ERD más marcado en Beta C3 mano derecha
# (criterio: mediana más negativa en movimiento real)
ranking = df_erd[
    (df_erd['canal'] == 'C3') &
    (df_erd['banda'] == 'Beta') &
    (df_erd['clase'] == 'derecha') &
    (df_erd['condicion'] == 'movimiento_real')
].groupby('sujeto')['ERD_pct'].median().sort_values()

sujeto_repr = int(ranking.index[0])
print(f"\nSujeto representativo seleccionado: S{sujeto_repr:03d}")
print(f"  Mediana ERD Beta C3 mano derecha (movimiento real): {ranking.iloc[0]:.1f}%")

# Cálculo de curvas para el sujeto representativo (sin promedio entre sujetos)
def calcular_curva_erd_sujeto(sujeto, condicion, clase, canal_idx,
                                banda_nombre, fs, n_baseline_muestras, n_ventana_suav):
    """Curva ERD para un único sujeto (promedio entre sus épocas).
    Usa la banda Mu específica del sujeto o Beta canónica según corresponda."""
    if banda_nombre == 'Mu':
        banda = bandas_mu_por_sujeto[sujeto]
    else:
        banda = BANDA_BETA

    epocas = epocas_todas[sujeto][condicion][clase][:, canal_idx, :] * 1e6
    curva, desvio, _ = calcular_curva_erd(epocas, fs, banda,
                                       n_baseline_muestras, n_ventana_suav)
    return curva, desvio

# ---- Visualización estilo Pfurtscheller ----
# Igual que la Figura 6b, generamos dos figuras separadas (una por banda)
# para mejorar la legibilidad en el informe.

estilos_repr = {
    ('movimiento_real', 'izquierda'):
        {'color': '#1976d2', 'linestyle': '-', 'label': 'Real - Izquierda'},
    ('movimiento_real', 'derecha'):
        {'color': '#e34948', 'linestyle': '-', 'label': 'Real - Derecha'},
    ('imaginacion', 'izquierda'):
        {'color': '#1976d2', 'linestyle': '--', 'label': 'Imaginación - Izquierda'},
    ('imaginacion', 'derecha'):
        {'color': '#e34948', 'linestyle': '--', 'label': 'Imaginación - Derecha'},
}


def graficar_curvas_sujeto_por_banda(banda_nombre, banda_rango, nombre_archivo):
    """Genera una figura con paneles C3 y C4 para una banda dada, sujeto individual."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    for i_col, canal_nombre in enumerate(['C3', 'C4']):
        ax = axes[i_col]

        for (condicion, clase), estilo in estilos_repr.items():
            canal_idx = ['C3', 'C4'].index(canal_nombre)
            curva, desvio = calcular_curva_erd_sujeto(
                sujeto_repr, condicion, clase, canal_idx,
                banda_nombre, FS, n_baseline, n_ventana_suavizado
            )
            t_local = np.arange(len(curva)) / FS + TMIN
            ax.plot(t_local, curva, **estilo, linewidth=1.8)
            ax.fill_between(t_local, curva - desvio, curva + desvio,
                             color=estilo['color'], alpha=0.10)

        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(4, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Sombreado de la ventana de ERS (post-estímulo)
        ax.axvspan(4.5, 6.0, alpha=0.08, color='green')

        ax.set_title(f'Canal {canal_nombre}', fontsize=13)
        ax.set_xlabel('Tiempo respecto al estímulo [s]', fontsize=11)
        if i_col == 0:
            ax.set_ylabel('ERD/ERS [%]', fontsize=11)
        # Calcular límites basados solo en las curvas medias
        curvas_medias_panel = []
        for (condicion, clase), estilo in estilos_repr.items():
            canal_idx = ['C3', 'C4'].index(canal_nombre)
            curva, _ = calcular_curva_erd_sujeto(
                sujeto_repr, condicion, clase, canal_idx,
                banda_nombre, FS, n_baseline, n_ventana_suavizado
            )
            curvas_medias_panel.append(curva)

        min_y = min(c.min() for c in curvas_medias_panel) * 1.3
        max_y = max(c.max() for c in curvas_medias_panel) * 1.3
        ax.set_ylim(min_y, max_y)
        ax.grid(True, alpha=0.3)

    # Leyenda única fuera de los subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=10)

    titulo_banda = f'{banda_nombre} ({banda_rango[0]}-{banda_rango[1]} Hz)'
    plt.suptitle(f'Curvas ERD/ERS del sujeto representativo S{sujeto_repr:03d} - '
                  f'Banda {titulo_banda}\n(zona verde = ventana ERS post-estímulo)',
                  fontsize=13, y=1.09)
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Figura guardada: {nombre_archivo}")


# Generamos las dos figuras separadas
graficar_curvas_sujeto_por_banda('Mu', bandas_mu_por_sujeto[sujeto_repr],
                                    'fig_06c_curvas_ERD_sujeto_Mu.png')
graficar_curvas_sujeto_por_banda('Beta', BANDA_BETA,
                                    'fig_06c_curvas_ERD_sujeto_Beta.png')

# -----------------------------------------------------------------------------
# BLOQUE 6.8: Análisis comparativo - Bandas canónicas vs específicas por sujeto
# -----------------------------------------------------------------------------
#
# METODOLOGÍA:
#
# Como validación empírica de la decisión metodológica de usar bandas Mu
# específicas por sujeto (Pfurtscheller et al., 2006), se recalcula el
# ERD% usando la banda canónica fija (8-12 Hz) y se compara directamente
# con los valores obtenidos usando bandas individuales.
#
# La hipótesis es que las bandas específicas producen ERDs más marcados
# y con menor variabilidad interindividual, porque están centradas en el
# pico de reactividad real de cada sujeto.

# Banda canónica Mu clásica de la literatura
BANDA_MU_CANONICA = (8, 12)

# Recalculamos ERD% con banda canónica para poder comparar
registros_erd_canonica = []

print("\nRecalculando ERD% con banda Mu canónica (8-12 Hz)...")
for suj in SUJETOS:
    for condicion in ['movimiento_real', 'imaginacion']:
        for clase in ['izquierda', 'derecha']:
            epocas = epocas_todas[suj][condicion][clase]
            if epocas.shape[0] == 0:
                continue
            n_epocas = epocas.shape[0]

            for i_epoca in range(n_epocas):
                for i_canal, nombre_canal in enumerate(['C3', 'C4']):
                    senal_canal = epocas[i_epoca, i_canal, :] * 1e6
                    erd, _ = calcular_erd_ers_epoca(
                        senal_canal, FS, BANDA_MU_CANONICA,
                        n_baseline, n_tarea_inicio, n_tarea_fin,
                        n_ers_inicio, n_ers_fin,
                        NPERSEG_WELCH, NOVERLAP_WELCH
                    )
                    registros_erd_canonica.append({
                        'sujeto': suj,
                        'condicion': condicion,
                        'clase': clase,
                        'canal': nombre_canal,
                        'ERD_pct_canonica': erd
                    })

df_canonica = pd.DataFrame(registros_erd_canonica)

# Merge con los datos originales (banda específica)
df_mu_especifica = df_erd[df_erd['banda']=='Mu'][
    ['sujeto', 'condicion', 'clase', 'canal', 'epoca', 'ERD_pct']
].rename(columns={'ERD_pct': 'ERD_pct_especifica'})

# Comparación agregada por sujeto (mediana por sujeto)
df_comparacion = df_mu_especifica.groupby(
    ['sujeto', 'condicion', 'clase', 'canal']
)['ERD_pct_especifica'].median().reset_index()

df_comp_canonica = df_canonica.groupby(
    ['sujeto', 'condicion', 'clase', 'canal']
)['ERD_pct_canonica'].median().reset_index()

df_comp_final = df_comparacion.merge(
    df_comp_canonica,
    on=['sujeto', 'condicion', 'clase', 'canal']
)

# Impresión de estadísticas
print("\n" + "="*70)
print("COMPARACIÓN: Banda Mu canónica (8-12 Hz) vs específica por sujeto")
print("="*70)

for cond in ['movimiento_real', 'imaginacion']:
    for canal in ['C3', 'C4']:
        subset = df_comp_final[
            (df_comp_final['condicion']==cond) &
            (df_comp_final['canal']==canal)
        ]
        med_can = subset['ERD_pct_canonica'].median()
        med_esp = subset['ERD_pct_especifica'].median()
        std_can = subset['ERD_pct_canonica'].std()
        std_esp = subset['ERD_pct_especifica'].std()
        etiqueta_cond = 'Real' if cond=='movimiento_real' else 'Imag'
        print(f"  {etiqueta_cond} {canal}: canónica {med_can:>7.2f} ± {std_can:>5.2f}% | "
              f"específica {med_esp:>7.2f} ± {std_esp:>5.2f}%")


# --- Visualización comparativa ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Filtramos solo movimiento real, mano derecha, canal C3 (donde el ERD es más claro)
subset_C3 = df_comp_final[
    (df_comp_final['condicion']=='movimiento_real') &
    (df_comp_final['clase']=='derecha') &
    (df_comp_final['canal']=='C3')
]

# Panel 1: Scatter de comparación por sujeto
axes[0].scatter(subset_C3['ERD_pct_canonica'], subset_C3['ERD_pct_especifica'],
                s=80, alpha=0.7, color='#1976d2', edgecolor='black')

# Línea de identidad y=x (para comparación visual)
lim_min = min(subset_C3['ERD_pct_canonica'].min(),
              subset_C3['ERD_pct_especifica'].min()) - 5
lim_max = max(subset_C3['ERD_pct_canonica'].max(),
              subset_C3['ERD_pct_especifica'].max()) + 5
axes[0].plot([lim_min, lim_max], [lim_min, lim_max],
             'k--', alpha=0.5, label='y = x')
axes[0].axhline(0, color='gray', linestyle=':', alpha=0.5)
axes[0].axvline(0, color='gray', linestyle=':', alpha=0.5)

# Anotar cada punto con el sujeto
for _, row in subset_C3.iterrows():
    axes[0].annotate(f"S{int(row['sujeto']):02d}",
                     (row['ERD_pct_canonica'], row['ERD_pct_especifica']),
                     fontsize=8, alpha=0.8,
                     xytext=(5, 5), textcoords='offset points')

axes[0].set_xlabel('ERD% con banda canónica Mu (8-12 Hz)')
axes[0].set_ylabel('ERD% con banda específica por sujeto')
axes[0].set_title('Comparación por sujeto - C3 - Movimiento real - Mano derecha')
axes[0].legend()

# Panel 2: Boxplot comparativo
df_largo_comp = pd.melt(
    df_comp_final[
        (df_comp_final['condicion']=='movimiento_real') &
        (df_comp_final['canal']=='C3')
    ],
    id_vars=['sujeto', 'clase'],
    value_vars=['ERD_pct_canonica', 'ERD_pct_especifica'],
    var_name='metodo', value_name='ERD_pct'
)
df_largo_comp['metodo'] = df_largo_comp['metodo'].map({
    'ERD_pct_canonica': 'Canónica (8-12 Hz)',
    'ERD_pct_especifica': 'Específica por sujeto'
})

sns.boxplot(data=df_largo_comp, x='metodo', y='ERD_pct',
            ax=axes[1],
            palette={'Canónica (8-12 Hz)': '#ff9800',
                     'Específica por sujeto': '#1976d2'})
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.7)
axes[1].set_ylabel('ERD [%] - mediana por sujeto')
axes[1].set_xlabel('Método de selección de banda Mu')
axes[1].set_title('Distribución entre sujetos - C3 - Movimiento real')

plt.suptitle('Impacto de la selección de banda: canónica vs específica por sujeto',
             fontsize=13)
plt.tight_layout()
plt.savefig('fig_09_comparacion_bandas.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Figura guardada: fig_09_comparacion_bandas.png")
# -----------------------------------------------------------------------------
# BLOQUE 6.9: Índice de lateralidad hemisférica y clasificación T1 vs T2
# -----------------------------------------------------------------------------
#
# METODOLOGÍA:
#
# El índice de lateralidad hemisférica es una feature clásica de BCI que
# cuantifica la diferencia de reactividad entre los hemisferios contralateral
# e ipsilateral al movimiento imaginado.
#
# Se calcula como:
#     LI = ERD_C3 - ERD_C4
#
# Interpretación:
#     LI < 0 : ERD más marcado en C3 (contralateral a mano DERECHA)
#              -> el sujeto probablemente imagina mover la mano derecha
#     LI > 0 : ERD más marcado en C4 (contralateral a mano IZQUIERDA)
#              -> el sujeto probablemente imagina mover la mano izquierda
#     LI ≈ 0 : sin diferencia hemisférica clara (baja separabilidad)
#
# Este índice se compara con las anotaciones reales del dataset (T1 = mano
# izquierda, T2 = mano derecha) para evaluar si podría servir como señal
# de control confiable en una BCI.
#
# La curva ROC (Receiver Operating Characteristic) muestra el compromiso
# entre sensibilidad y especificidad para todos los umbrales posibles de
# clasificación. El área bajo la curva (AUC) es una medida global de
# separabilidad: 0.5 = azar, 1.0 = clasificación perfecta.

from sklearn.metrics import roc_curve, auc

# Cálculo del índice de lateralidad por época
# LI = ERD_C3 - ERD_C4 para cada época en la banda Mu específica del sujeto

def calcular_indice_lateralidad(sujeto, condicion, banda_nombre):
    """
    Calcula el índice de lateralidad (ERD_C3 - ERD_C4) por época
    para una condición y banda dadas, junto con la clase verdadera.

    Retorna
    -------
    li_valores : array
        Índices de lateralidad por época
    clases_verdaderas : array
        1 = mano derecha (T2), 0 = mano izquierda (T1)
    """
    li_valores = []
    clases_verdaderas = []

    for clase in ['izquierda', 'derecha']:
        # Extraer ERD por época para C3 y C4 de esta clase
        datos_C3 = df_erd[
            (df_erd['sujeto']==sujeto) &
            (df_erd['condicion']==condicion) &
            (df_erd['clase']==clase) &
            (df_erd['canal']=='C3') &
            (df_erd['banda']==banda_nombre)
        ].sort_values('epoca')['ERD_pct'].values

        datos_C4 = df_erd[
            (df_erd['sujeto']==sujeto) &
            (df_erd['condicion']==condicion) &
            (df_erd['clase']==clase) &
            (df_erd['canal']=='C4') &
            (df_erd['banda']==banda_nombre)
        ].sort_values('epoca')['ERD_pct'].values

        # LI por época
        li_epocas = datos_C3 - datos_C4

        # Clase verdadera: 1 si es derecha (T2), 0 si es izquierda (T1)
        etiqueta = 1 if clase == 'derecha' else 0

        li_valores.extend(li_epocas)
        clases_verdaderas.extend([etiqueta] * len(li_epocas))

    return np.array(li_valores), np.array(clases_verdaderas)


# Calculamos el índice de lateralidad para todas las condiciones y bandas
# y evaluamos la separabilidad mediante curva ROC

resultados_lateralidad = {}

print("\n" + "="*60)
print("ÍNDICE DE LATERALIDAD Y CLASIFICACIÓN T1 vs T2")
print("="*60)

for condicion in ['movimiento_real', 'imaginacion']:
    for banda_nombre in ['Mu', 'Beta']:
        # Agregamos LI de todos los sujetos
        li_todos = []
        clases_todos = []

        for suj in SUJETOS:
            li, clases = calcular_indice_lateralidad(suj, condicion, banda_nombre)
            li_todos.extend(li)
            clases_todos.extend(clases)

        li_todos = np.array(li_todos)
        clases_todos = np.array(clases_todos)

        # Curva ROC: usamos -LI como score (porque LI negativo predice clase 1)
        fpr, tpr, umbrales = roc_curve(clases_todos, -li_todos)
        auc_valor = auc(fpr, tpr)

        # Umbral óptimo por criterio de Youden (maximiza TPR - FPR)
        idx_optimo = np.argmax(tpr - fpr)
        umbral_optimo = -umbrales[idx_optimo]  # revertimos el signo

        resultados_lateralidad[(condicion, banda_nombre)] = {
            'fpr': fpr, 'tpr': tpr, 'auc': auc_valor,
            'umbral_optimo': umbral_optimo,
            'li_todos': li_todos, 'clases_todos': clases_todos
        }

        etiqueta_cond = 'Real' if condicion=='movimiento_real' else 'Imaginación'
        print(f"  {etiqueta_cond} - {banda_nombre}: AUC = {auc_valor:.3f}, "
              f"umbral óptimo LI = {umbral_optimo:.2f}%")


# --- Visualización ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel superior izquierdo: Distribución de LI por clase (Beta, movimiento real)
ax = axes[0, 0]
res = resultados_lateralidad[('movimiento_real', 'Beta')]
li_izq = res['li_todos'][res['clases_todos']==0]
li_der = res['li_todos'][res['clases_todos']==1]

ax.hist(li_izq, bins=25, alpha=0.6, color='#2a78d6',
        label=f'Mano IZQ (T1, n={len(li_izq)})', density=True)
ax.hist(li_der, bins=25, alpha=0.6, color='#e34948',
        label=f'Mano DER (T2, n={len(li_der)})', density=True)
ax.axvline(res['umbral_optimo'], color='black', linestyle='--',
           label=f'Umbral óptimo: {res["umbral_optimo"]:.1f}%')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Índice de Lateralidad LI = ERD_C3 - ERD_C4 [%]')
ax.set_ylabel('Densidad')
ax.set_title('Distribución del LI - Movimiento real - Banda Beta')
ax.legend(fontsize=9)

# Panel superior derecho: Curvas ROC comparativas
ax = axes[0, 1]
colores = {'movimiento_real': {'Mu': '#1976d2', 'Beta': '#0d47a1'},
           'imaginacion': {'Mu': '#ff9800', 'Beta': '#e65100'}}
etiquetas = {'movimiento_real': 'Real', 'imaginacion': 'Imaginación'}

for cond in ['movimiento_real', 'imaginacion']:
    for banda in ['Mu', 'Beta']:
        res = resultados_lateralidad[(cond, banda)]
        ax.plot(res['fpr'], res['tpr'],
                color=colores[cond][banda], linewidth=2,
                label=f'{etiquetas[cond]} {banda} (AUC={res["auc"]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Azar (AUC=0.5)')
ax.set_xlabel('Tasa de falsos positivos (1-Especificidad)')
ax.set_ylabel('Tasa de verdaderos positivos (Sensibilidad)')
ax.set_title('Curvas ROC: capacidad de decodificar mano IZQ vs DER')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel inferior izquierdo: Distribución de LI para imaginación (comparativa)
ax = axes[1, 0]
res = resultados_lateralidad[('imaginacion', 'Beta')]
li_izq = res['li_todos'][res['clases_todos']==0]
li_der = res['li_todos'][res['clases_todos']==1]

ax.hist(li_izq, bins=25, alpha=0.6, color='#2a78d6',
        label=f'Mano IZQ (T1, n={len(li_izq)})', density=True)
ax.hist(li_der, bins=25, alpha=0.6, color='#e34948',
        label=f'Mano DER (T2, n={len(li_der)})', density=True)
ax.axvline(res['umbral_optimo'], color='black', linestyle='--',
           label=f'Umbral óptimo: {res["umbral_optimo"]:.1f}%')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Índice de Lateralidad LI = ERD_C3 - ERD_C4 [%]')
ax.set_ylabel('Densidad')
ax.set_title('Distribución del LI - IMAGINACIÓN - Banda Beta')
ax.legend(fontsize=9)

# Panel inferior derecho: Tabla resumen de precisión
ax = axes[1, 1]
ax.axis('off')

tabla_texto = "Resumen de clasificación T1 (izq) vs T2 (der)\n"
tabla_texto += "usando índice de lateralidad hemisférica\n\n"
tabla_texto += f"{'Condición':<18}{'Banda':<8}{'AUC':<8}{'Interpretación':<25}\n"
tabla_texto += "-"*60 + "\n"

for cond in ['movimiento_real', 'imaginacion']:
    for banda in ['Mu', 'Beta']:
        res = resultados_lateralidad[(cond, banda)]
        auc_v = res['auc']

        if auc_v >= 0.8:
            interp = "Buena separabilidad"
        elif auc_v >= 0.65:
            interp = "Separabilidad moderada"
        elif auc_v >= 0.55:
            interp = "Separabilidad marginal"
        else:
            interp = "Sin separabilidad clara"

        etiqueta = etiquetas[cond]
        tabla_texto += f"{etiqueta:<18}{banda:<8}{auc_v:<8.3f}{interp:<25}\n"

tabla_texto += "\n\nInterpretación clínica:\n"
tabla_texto += "AUC > 0.8: viable para BCI con calibración simple\n"
tabla_texto += "AUC 0.65-0.8: requiere entrenamiento adicional del sujeto\n"
tabla_texto += "AUC < 0.65: no discriminable sin features adicionales"

ax.text(0.05, 0.95, tabla_texto, transform=ax.transAxes,
        fontsize=10, family='monospace', verticalalignment='top')

plt.suptitle('Análisis de decodificación BCI: predicción de mano imaginada '
             'a partir del índice de lateralidad',
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig('fig_10_lateralidad_ROC.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_10_lateralidad_ROC.png")

# -----------------------------------------------------------------------------
# BLOQUE 6.9b: Análisis de lateralidad por sujeto individual
# -----------------------------------------------------------------------------
#
# Recalculamos el AUC para cada sujeto por separado, para identificar
# variabilidad interindividual en la capacidad de decodificación.

print("\n" + "="*70)
print("AUC de clasificación LI por sujeto individual (banda Beta)")
print("="*70)
print(f"{'Sujeto':<10}{'Real AUC':<15}{'Imag AUC':<15}{'Interp real':<20}")
print("-"*70)

aucs_por_sujeto = {'movimiento_real': [], 'imaginacion': []}

for suj in SUJETOS:
    aucs_sujeto = {}
    for condicion in ['movimiento_real', 'imaginacion']:
        li_suj, clases_suj = calcular_indice_lateralidad(suj, condicion, 'Beta')

        # Solo calculamos AUC si hay ambas clases con suficientes muestras
        if len(np.unique(clases_suj)) < 2 or len(li_suj) < 10:
            aucs_sujeto[condicion] = np.nan
            continue

        fpr, tpr, _ = roc_curve(clases_suj, -li_suj)
        aucs_sujeto[condicion] = auc(fpr, tpr)
        aucs_por_sujeto[condicion].append(aucs_sujeto[condicion])

    # Interpretación
    auc_real = aucs_sujeto['movimiento_real']
    if np.isnan(auc_real):
        interp = "Datos insuficientes"
    elif auc_real >= 0.75:
        interp = "Buena"
    elif auc_real >= 0.60:
        interp = "Moderada"
    else:
        interp = "Baja/Azar"

    print(f"S{suj:03d}      {aucs_sujeto['movimiento_real']:<15.3f}"
          f"{aucs_sujeto['imaginacion']:<15.3f}{interp:<20}")

# Estadísticas agregadas
print("\nEstadísticas de AUC entre sujetos (banda Beta):")
print(f"  Movimiento real: media = {np.mean(aucs_por_sujeto['movimiento_real']):.3f}, "
      f"desvío = {np.std(aucs_por_sujeto['movimiento_real']):.3f}")
print(f"  Imaginación:     media = {np.mean(aucs_por_sujeto['imaginacion']):.3f}, "
      f"desvío = {np.std(aucs_por_sujeto['imaginacion']):.3f}")

# Cantidad de sujetos con AUC "utilizable" (>=0.65)
n_utilizables_real = sum(1 for a in aucs_por_sujeto['movimiento_real'] if a >= 0.65)
n_utilizables_imag = sum(1 for a in aucs_por_sujeto['imaginacion'] if a >= 0.65)
print(f"\n  Sujetos con AUC≥0.65 en movimiento real: {n_utilizables_real}/{len(SUJETOS)}")
print(f"  Sujetos con AUC≥0.65 en imaginación:     {n_utilizables_imag}/{len(SUJETOS)}")

# -----------------------------------------------------------------------------
# BLOQUE 6.10: Método de Varianza Intertrial (ITV) para separar ERD del VEP
# -----------------------------------------------------------------------------
#
# METODOLOGÍA:
#
# El método clásico de Band Power (elevación al cuadrado) captura tanto:
#   - Cambios de potencia phase-locked (el VEP del estímulo visual)
#   - Cambios de potencia no phase-locked (el ERD real de imaginación motora)
#
# Esto genera el pico grande visible en t=0 en las curvas ERD/ERS, que
# no es ERD real sino contaminación por el potencial evocado visual.
#
# Kalcher & Pfurtscheller (1995) proponen el método de VARIANZA INTERTRIAL
# como alternativa: calcular la varianza punto a punto entre trials en
# lugar de elevar al cuadrado la señal individual.
#
# Justificación matemática:
#   - Componentes phase-locked (VEP): idénticas entre trials -> varianza = 0
#   - Componentes no phase-locked (ERD): varían entre trials -> varianza > 0
#
# Referencia:
# Kalcher, J. & Pfurtscheller, G. (1995). Discrimination between phase-locked
# and non-phase-locked event-related EEG activity. Electroenceph. Clin.
# Neurophysiol., 94(5), 381-384.

def calcular_curva_erd_ITV(epocas, fs, banda, n_baseline_muestras,
                            n_ventana_suav):
    """
    Calcula la curva ERD%(t) usando el método de Varianza Intertrial (ITV).

    A diferencia del método clásico (elevación al cuadrado), este método
    elimina la contribución de componentes phase-locked como el Visual
    Evoked Potential.

    Pasos:
    1. Filtrado bidireccional por banda estrecha (igual al método clásico).
    2. Cálculo del promedio entre trials en cada instante.
    3. Cálculo de la varianza punto a punto: (x_trial - x_promedio)².
    4. Promedio de las varianzas entre trials.
    5. Suavizado por ventana móvil.
    6. Normalización porcentual respecto al baseline.

    Parámetros idénticos a calcular_curva_erd.

    Retorna
    -------
    curva_promedio : array 1D
        Curva ERD%(t) basada en varianza intertrial
    curva_desvio : array 1D
        Desvío estándar (para banda de confianza)
    """
    # Filtro angosto en la banda de interés
    sos_banda = diseno_filtro_banda_estrecha(banda, fs)

    # Filtrado bidireccional de todas las épocas
    epocas_filtradas = signal.sosfiltfilt(sos_banda, epocas, axis=-1)

    # DIFERENCIA CLAVE con el método clásico:
    # Calculamos el promedio entre trials en cada instante temporal
    promedio_entre_trials = epocas_filtradas.mean(axis=0)

    # Desviación de cada trial respecto al promedio phase-locked
    desviaciones = epocas_filtradas - promedio_entre_trials

    # Elevamos al cuadrado esa desviación (no la señal original)
    # Esto es la contribución no phase-locked de cada trial
    potencia_no_phase_locked = desviaciones ** 2

    # Suavizado por convolución con ventana rectangular normalizada
    kernel = np.ones(n_ventana_suav) / n_ventana_suav
    potencia_suave = np.array([
        np.convolve(pot, kernel, mode='same') for pot in potencia_no_phase_locked
    ])

    # Promedio entre trials de la potencia no phase-locked
    potencia_promedio = potencia_suave.mean(axis=0)
    potencia_desvio = potencia_suave.std(axis=0)

    # Baseline promedio (escalar) sobre la señal promediada
    baseline_medio = potencia_promedio[:n_baseline_muestras].mean()

    # Normalización porcentual sobre la curva promedio
    curva_promedio = (potencia_promedio - baseline_medio) / baseline_medio * 100
    curva_desvio = potencia_desvio / baseline_medio * 100

    return curva_promedio, curva_desvio


# Cálculo comparativo: método clásico vs ITV
# Elegimos un sujeto con buena calidad para la comparación
sujeto_comparacion = SUJETOS[0]  # Sujeto 1 o cambia por otro con buenos datos

print(f"\nCalculando comparación clásico vs ITV para sujeto {sujeto_comparacion}...")

# Configuración: mano derecha, canal C3 (contralateral, ERD esperado más marcado)
canal_idx = 0  # C3
clase = 'derecha'
banda_mu_suj = bandas_mu_por_sujeto[sujeto_comparacion]

# Movimiento real
epocas_real = epocas_todas[sujeto_comparacion]['movimiento_real'][clase][:, canal_idx, :] * 1e6

resultado = calcular_curva_erd(epocas_real, FS, banda_mu_suj, n_baseline, n_ventana_suavizado)
curva_clasica_real_mu = resultado[0]

resultado = calcular_curva_erd_ITV(epocas_real, FS, banda_mu_suj, n_baseline, n_ventana_suavizado)
curva_itv_real_mu = resultado[0]

resultado = calcular_curva_erd(epocas_real, FS, BANDA_BETA, n_baseline, n_ventana_suavizado)
curva_clasica_real_beta = resultado[0]

resultado = calcular_curva_erd_ITV(epocas_real, FS, BANDA_BETA, n_baseline, n_ventana_suavizado)
curva_itv_real_beta = resultado[0]

# Imaginación
epocas_imag = epocas_todas[sujeto_comparacion]['imaginacion'][clase][:, canal_idx, :] * 1e6

resultado = calcular_curva_erd(epocas_imag, FS, banda_mu_suj, n_baseline, n_ventana_suavizado)
curva_clasica_imag_mu = resultado[0]

resultado = calcular_curva_erd_ITV(epocas_imag, FS, banda_mu_suj, n_baseline, n_ventana_suavizado)
curva_itv_imag_mu = resultado[0]

resultado = calcular_curva_erd(epocas_imag, FS, BANDA_BETA, n_baseline, n_ventana_suavizado)
curva_clasica_imag_beta = resultado[0]

resultado = calcular_curva_erd_ITV(epocas_imag, FS, BANDA_BETA, n_baseline, n_ventana_suavizado)
curva_itv_imag_beta = resultado[0]

# Vector de tiempo
n_muestras_epoca = len(curva_clasica_real_mu)
tiempo_curva = np.arange(n_muestras_epoca) / FS + TMIN


# --- Visualización comparativa ---
fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)

# Panel superior izquierdo: Mu - Movimiento real
axes[0, 0].plot(tiempo_curva, curva_clasica_real_mu,
                color='#1976d2', linewidth=2, label='Método clásico (Band Power)')
axes[0, 0].plot(tiempo_curva, curva_itv_real_mu,
                color='#e65100', linewidth=2, linestyle='--',
                label='Método ITV (varianza intertrial)')
axes[0, 0].axhline(0, color='gray', linestyle=':', alpha=0.7)
axes[0, 0].axvline(0, color='black', linestyle='--', alpha=0.7)
axes[0, 0].axvline(4, color='black', linestyle='--', alpha=0.5)
axes[0, 0].set_title(f'Banda Mu - Movimiento Real - C3 (S{sujeto_comparacion:03d})')
axes[0, 0].set_ylabel('ERD/ERS [%]')
axes[0, 0].legend(loc='best', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Panel superior derecho: Beta - Movimiento real
axes[0, 1].plot(tiempo_curva, curva_clasica_real_beta,
                color='#1976d2', linewidth=2, label='Método clásico (Band Power)')
axes[0, 1].plot(tiempo_curva, curva_itv_real_beta,
                color='#e65100', linewidth=2, linestyle='--',
                label='Método ITV (varianza intertrial)')
axes[0, 1].axhline(0, color='gray', linestyle=':', alpha=0.7)
axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.7)
axes[0, 1].axvline(4, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_title(f'Banda Beta - Movimiento Real - C3 (S{sujeto_comparacion:03d})')
axes[0, 1].set_ylabel('ERD/ERS [%]')
axes[0, 1].legend(loc='best', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Panel inferior izquierdo: Mu - Imaginación
axes[1, 0].plot(tiempo_curva, curva_clasica_imag_mu,
                color='#1976d2', linewidth=2, label='Método clásico (Band Power)')
axes[1, 0].plot(tiempo_curva, curva_itv_imag_mu,
                color='#e65100', linewidth=2, linestyle='--',
                label='Método ITV (varianza intertrial)')
axes[1, 0].axhline(0, color='gray', linestyle=':', alpha=0.7)
axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.7)
axes[1, 0].axvline(4, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_title(f'Banda Mu - Imaginación - C3 (S{sujeto_comparacion:03d})')
axes[1, 0].set_ylabel('ERD/ERS [%]')
axes[1, 0].set_xlabel('Tiempo respecto al estímulo [s]')
axes[1, 0].legend(loc='best', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Panel inferior derecho: Beta - Imaginación
axes[1, 1].plot(tiempo_curva, curva_clasica_imag_beta,
                color='#1976d2', linewidth=2, label='Método clásico (Band Power)')
axes[1, 1].plot(tiempo_curva, curva_itv_imag_beta,
                color='#e65100', linewidth=2, linestyle='--',
                label='Método ITV (varianza intertrial)')
axes[1, 1].axhline(0, color='gray', linestyle=':', alpha=0.7)
axes[1, 1].axvline(0, color='black', linestyle='--', alpha=0.7)
axes[1, 1].axvline(4, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title(f'Banda Beta - Imaginación - C3 (S{sujeto_comparacion:03d})')
axes[1, 1].set_ylabel('ERD/ERS [%]')
axes[1, 1].set_xlabel('Tiempo respecto al estímulo [s]')
axes[1, 1].legend(loc='best', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Comparación método clásico vs ITV para separar ERD del VEP\n'
             f'(Sujeto S{sujeto_comparacion:03d} - Mano derecha)',
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig('fig_11_comparacion_ITV.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_11_comparacion_ITV.png")

# -----------------------------------------------------------------------------
# BLOQUE 6.11: Contraste "mejor caso" vs "caso difícil" - Separabilidad LI
# -----------------------------------------------------------------------------
#
# OBJETIVO:
#
# Ilustrar visualmente la variabilidad interindividual en la capacidad
# de decodificar la mano imaginada a partir del índice de lateralidad.
#
# S042 representa un usuario ideal de BCI (AUC = 0.86 en imaginación):
#   los histogramas de LI están claramente separados entre clases,
#   permitiendo una decodificación confiable con un umbral simple.
#
# S014 representa un caso de "BCI illiteracy" (AUC = 0.28 en real):
#   los histogramas de LI se solapan casi completamente, imposibilitando
#   la separación de clases con features univariadas.
#
# Este contraste ilustra por qué el desarrollo de prótesis controladas
# por EEG requiere calibración individualizada y estrategias de manejo
# de la variabilidad interindividual.

SUJETO_MEJOR = 42
SUJETO_DIFICIL = 14

# Extraemos los índices de lateralidad para cada caso (banda Beta,
# donde el ERD suele ser más consistente)
li_mejor, cl_mejor = calcular_indice_lateralidad(SUJETO_MEJOR, 'imaginacion', 'Beta')
li_dificil, cl_dificil = calcular_indice_lateralidad(SUJETO_DIFICIL, 'imaginacion', 'Beta')

# Cálculo de AUC para mostrar en el título
fpr_m, tpr_m, _ = roc_curve(cl_mejor, -li_mejor)
auc_mejor = auc(fpr_m, tpr_m)
fpr_d, tpr_d, _ = roc_curve(cl_dificil, -li_dificil)
auc_dificil = auc(fpr_d, tpr_d)


# --- Figura ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: mejor caso (S042)
li_izq_m = li_mejor[cl_mejor==0]
li_der_m = li_mejor[cl_mejor==1]

axes[0].hist(li_izq_m, bins=15, alpha=0.65, color='#2a78d6',
              label=f'Mano IZQ (T1, n={len(li_izq_m)})',
              density=True, edgecolor='black', linewidth=0.5)
axes[0].hist(li_der_m, bins=15, alpha=0.65, color='#e34948',
              label=f'Mano DER (T2, n={len(li_der_m)})',
              density=True, edgecolor='black', linewidth=0.5)
axes[0].axvline(0, color='gray', linestyle=':', alpha=0.7)
axes[0].set_xlabel('Índice de Lateralidad LI = ERD_C3 - ERD_C4 [%]', fontsize=11)
axes[0].set_ylabel('Densidad', fontsize=11)
axes[0].set_title(f'Sujeto S{SUJETO_MEJOR:03d} - AUC = {auc_mejor:.3f}',
                   fontsize=12)
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Panel derecho: caso difícil (S014)
li_izq_d = li_dificil[cl_dificil==0]
li_der_d = li_dificil[cl_dificil==1]

axes[1].hist(li_izq_d, bins=15, alpha=0.65, color='#2a78d6',
              label=f'Mano IZQ (T1, n={len(li_izq_d)})',
              density=True, edgecolor='black', linewidth=0.5)
axes[1].hist(li_der_d, bins=15, alpha=0.65, color='#e34948',
              label=f'Mano DER (T2, n={len(li_der_d)})',
              density=True, edgecolor='black', linewidth=0.5)
axes[1].axvline(0, color='gray', linestyle=':', alpha=0.7)
axes[1].set_xlabel('Índice de Lateralidad LI = ERD_C3 - ERD_C4 [%]', fontsize=11)
axes[1].set_ylabel('Densidad', fontsize=11)
axes[1].set_title(f'Sujeto S{SUJETO_DIFICIL:03d} - AUC = {auc_dificil:.3f}',
                   fontsize=12)
axes[1].legend(fontsize=10, loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Contraste de separabilidad de clases entre sujetos - Imaginación motora - Banda Beta',
             fontsize=13, y=0.95)
plt.tight_layout()
plt.savefig('fig_13_contraste_sujetos.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_13_contraste_sujetos.png")
# -----------------------------------------------------------------------------
# BLOQUE 7: Visualizaciones comparativas principales
# -----------------------------------------------------------------------------

# --- Figura 6: Boxplots comparativos por banda y electrodo ---
#
# Cada subplot muestra los valores de ERD% por época para las 4 combinaciones
# condición × clase. Permite visualizar la separabilidad entre clases y la
# diferencia entre movimiento real e imaginado.

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
combinaciones = [
    ('C3', 'Mu'), ('C4', 'Mu'),
    ('C3', 'Beta'), ('C4', 'Beta')
]

for ax, (canal, banda) in zip(axes.flat, combinaciones):
    datos_filtrados = df_erd[(df_erd['canal']==canal) & (df_erd['banda']==banda)]

    sns.boxplot(data=datos_filtrados, x='condicion', y='ERD_pct',
                hue='clase', ax=ax,
                palette={'izquierda': '#2a78d6', 'derecha': '#e34948'},
                showfliers=False)  # oculta outliers extremos

    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_title(f'Canal {canal} - Banda {banda}')
    ax.set_xlabel('Condición')
    ax.set_ylabel('ERD [%]')
    ax.set_ylim(-100, 100)  # rango razonable para visualización
    ax.legend(title='Mano', loc='best')

plt.suptitle('ERD% por condición, clase, banda y electrodo',
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig('fig_06_boxplots_ERD.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_06_boxplots_ERD.png")


# --- Figura 7: Comparación por sujeto del ERD% promedio ---
#
# Muestra la variabilidad interindividual del ERD promediado por sujeto.

# Cambiamos a 2 filas, 1 columna y ajustamos el tamaño de la figura (ej. 12x10)
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Promedio de ERD% por sujeto, condición y banda en C3 mano derecha
# (contralateral, donde esperamos el ERD más marcado)
datos_C3_der = df_erd[(df_erd['canal']=='C3') & (df_erd['clase']=='derecha')]
promedios_C3 = datos_C3_der.groupby(
    ['sujeto', 'condicion', 'banda'])['ERD_pct'].mean().reset_index()

# Subplot superior (axes[0])
sns.barplot(data=promedios_C3[promedios_C3['banda']=='Mu'],
            x='sujeto', y='ERD_pct', hue='condicion', ax=axes[0],
            palette={'movimiento_real': '#1976d2', 'imaginacion': '#ff9800'})
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.7)
axes[0].set_title('ERD% promedio en banda Mu - C3 - Mano derecha')
axes[0].set_xlabel('Sujeto')
axes[0].set_ylabel('ERD [%]')

# Subplot inferior (axes[1])
sns.barplot(data=promedios_C3[promedios_C3['banda']=='Beta'],
            x='sujeto', y='ERD_pct', hue='condicion', ax=axes[1],
            palette={'movimiento_real': '#1976d2', 'imaginacion': '#ff9800'})
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.7)
axes[1].set_title('ERD% promedio en banda Beta - C3 - Mano derecha')
axes[1].set_xlabel('Sujeto')
axes[1].set_ylabel('ERD [%]')

plt.tight_layout()
plt.savefig('fig_07_ERD_por_sujeto.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_07_ERD_por_sujeto.png")

# --- Figura nueva: Comparación ERD (tarea) vs ERS (post-estímulo) ---

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ERD en C3 mano derecha (contralateral)
datos_erd = df_erd[(df_erd['canal']=='C3') & (df_erd['clase']=='derecha')].copy()

# Reorganizamos para tener ERD y ERS en filas separadas para el boxplot
df_largo = pd.melt(datos_erd,
                    id_vars=['sujeto', 'condicion', 'banda'],
                    value_vars=['ERD_pct', 'ERS_pct'],
                    var_name='medida', value_name='valor_pct')
df_largo['medida'] = df_largo['medida'].map(
    {'ERD_pct': 'ERD (0.5-3.5 s)', 'ERS_pct': 'ERS (4.5-6 s)'}
)

# Panel izquierdo: banda Mu
datos_mu = df_largo[df_largo['banda']=='Mu']
sns.boxplot(data=datos_mu, x='condicion', y='valor_pct', hue='medida',
            ax=axes[0], showfliers=False,
            palette={'ERD (0.5-3.5 s)': '#1976d2', 'ERS (4.5-6 s)': '#4caf50'})
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.7)
axes[0].set_title('Banda Mu - C3 - Mano derecha')
axes[0].set_ylabel('Cambio de potencia [%]')
axes[0].set_ylim(-80, 80)

# Panel derecho: banda Beta
datos_beta = df_largo[df_largo['banda']=='Beta']
sns.boxplot(data=datos_beta, x='condicion', y='valor_pct', hue='medida',
            ax=axes[1], showfliers=False,
            palette={'ERD (0.5-3.5 s)': '#1976d2', 'ERS (4.5-6 s)': '#4caf50'})
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.7)
axes[1].set_title('Banda Beta - C3 - Mano derecha')
axes[1].set_ylabel('Cambio de potencia [%]')
axes[1].set_ylim(-80, 80)

plt.suptitle('Contraste ERD (durante estímulo) vs ERS (post-estímulo) - '
             'todos los sujetos', fontsize=13)
plt.tight_layout()
plt.savefig('fig_08_ERD_vs_ERS.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Figura guardada: fig_08_ERD_vs_ERS.png")

# --- Tabla resumen exportable a CSV ---

tabla_resumen = df_erd.groupby(
    ['sujeto', 'condicion', 'canal', 'banda', 'clase']
)['ERD_pct'].agg(['mean', 'std', 'count']).round(2).reset_index()

tabla_resumen.to_csv('tabla_resumen_ERD.csv', index=False)
print("\n✓ Tabla exportada: tabla_resumen_ERD.csv")


# -----------------------------------------------------------------------------
# BLOQUE 8: Guardado de resultados para el Script 3 (Laplaciano opcional)
# -----------------------------------------------------------------------------

resultados = {
    'df_erd': df_erd,
    'epocas_todas': epocas_todas,
    'datos_procesados': datos_procesados,
    'sos_filtro': sos,
    'contador_rechazos': contador_rechazos,
    'config': {
        'FS': FS,
        'TMIN': TMIN,
        'TMAX': TMAX,
        'n_baseline': n_baseline,
        'n_tarea_inicio': n_tarea_inicio,
        'n_tarea_fin': n_tarea_fin,
        'BANDA_MU': BANDA_MU,
        'BANDA_BETA': BANDA_BETA,
        'FRECUENCIA_PASO': FRECUENCIA_PASO,
        'FRECUENCIA_STOP': FRECUENCIA_STOP,
        'orden_filtro': orden_filtro,
        'UMBRAL_ARTEFACTO_uV': UMBRAL_ARTEFACTO_uV
    }
}

with open('resultados_principales.pkl', 'wb') as f:
    pickle.dump(resultados, f)

print("\n" + "="*60)
print("SCRIPT 2 COMPLETADO")
print("="*60)
print("Archivos generados:")
print("  - fig_03_diseno_filtro.png")
print("  - fig_04_comparacion_cruda_filtrada.png")
print("  - fig_05_scalogramas_CWT.png")
print("  - fig_06_boxplots_ERD.png")
print("  - fig_06b_curvas_ERD_tiempo.png")
print("  - fig_07_ERD_por_sujeto.png")
print("  - tabla_resumen_ERD.csv")
print("  - resultados_principales.pkl (usado por el Script 3)")
print("\nProceder con el Script 3 (opcional): 03_bonus_laplaciano.py")