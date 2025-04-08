# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:10:48 2025

@author: Tobi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:20:15 2025

@author: Tobi
"""

#%% módulos y funciones a importar

import numpy as np
import matplotlib.pyplot as plt

# B = bits del ADC
# kn = escala para la potencia del ruido de cuantización
def cuantizacion_ADC (B,kn):
    
    def mi_funcion_sen (vmax, dc, f0, ph, N, fs):
        #fs frecuencia de muestreo (Hz)
        #N cantidad de muestras
        
        ts = 1/fs # tiempo de muestreo o periodo
        tt=np.linspace (0, (N-1)*ts, N) #vector de tiempo
        
        #generacion de la señal senoidal
        xx= dc + vmax*np.sin(2*np.pi*f0*tt + ph)
        #la señal debe generarse con la formula: x(t)=DC+Vmax*sen(2pift+fase)
        
        return tt, xx
    
    
    
    #%% Datos de la simulación
    
    fs =  1000 # frecuencia de muestreo (Hz)
    N = 1000 # cantidad de muestras
    # con 1000 para cada una normalizamos la resolucion espectral
    
    # Datos del ADC
    Vf = 2 # rango simétrico de +/- Vf Volts 
    q = 2*Vf/(2**B)# paso de cuantización de q Volts
    
    ##1 de ganancia, fijarte el ancho de banda, y la potencia del radio 50 al cuadrado
    # datos del ruido (potencia de la señal normalizada, es decir 1 W)
    pot_ruido_cuant = q**2/12 # Watts 
    pot_ruido_analog = pot_ruido_cuant * kn # 
    
    
    df = fs/N # resolución espectral
    f0 = df
    
    
    #%% Experimento: 
    """
       Se desea simular el efecto de la cuantización sobre una señal senoidal de 
       frecuencia 1 Hz. La señal "analógica" podría tener añadida una cantidad de 
       ruido gausiano e incorrelado.
       
       Se pide analizar el efecto del muestreo y cuantización sobre la señal 
       analógica. Para ello se proponen una serie de gráficas que tendrá que ayudar
       a construir para luego analizar los resultados.
       
    """
    # Genero señal senoidal
    tt, xx = mi_funcion_sen(1.4, 0,f0, 0, N, fs)
    # Potencia normalizada:
    xn=xx/np.std(xx)
    
    
    # np.random.normal
    # np.random.uniform
    
    
    # Señales
    
    analog_sig = xn # señal analógica sin ruido
    
    nn=np.random.normal(0,np.sqrt(pot_ruido_analog),N) # señal de ruido analogico
    
    sr = xn+nn # señal analógica de entrada al ADC (con ruido analógico)
    
    
    srq = np.round(sr/q)*q# señal cuantizada, (señal divida la cantidad total de bits)
    
    nq =  srq-sr# señal de ruido de cuantización
    
    
    
    
    #%% Visualización de resultados
    
    # cierro ventanas anteriores
    plt.close('all')
    
    ##################
    # Señal temporal
    ##################
    
    plt.figure(4)
    
    plt.plot(tt, srq, lw=1, linestyle='-', color='blue', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$ s_{RQ} = Q_{B,V_F}\{s_R\}$ (ADC out)')
    plt.plot(tt, sr, lw=1, color='green', marker='o', markersize='2', ls='dotted', label='$ s_R = s + n $ (ADC in)')
    plt.plot(tt, analog_sig, lw=1, color='yellow', ls='--', label='$ s $ (analog)')
    
    plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
    plt.xlabel('tiempo [segundos]')
    plt.ylabel('Amplitud [V]')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.show()
    
    
    ###########
    # Espectro
    ###########
    
    plt.figure(5)
    ft_SR = 1/N*np.fft.fft( sr)
    ft_Srq = 1/N*np.fft.fft( srq)
    ft_As = 1/N*np.fft.fft( analog_sig)
    ft_Nq = 1/N*np.fft.fft( nq)
    ft_Nn = 1/N*np.fft.fft( nn)
    
    # grilla de sampleo frecuencial
    ff = np.linspace(0, (N-1)*df, N)
    
    bfrec = ff <= fs/2
    
    Nnq_mean = np.mean(np.abs(ft_Nq)**2)
    nNn_mean = np.mean(np.abs(ft_Nn)**2)
    
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $ (ADC in)' )
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_{RQ} = Q_{B,V_F}\{s_R\}$ (ADC out)' )
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
    
    plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
    plt.ylabel('Densidad de Potencia [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylim(-80, 5)
    axes_hdl = plt.gca()
    axes_hdl.legend()
    
    #############
    # Histograma
    #############
    
    plt.figure(6)
    bins = 10
    plt.hist(nq.flatten()/(q), bins=bins)
    plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
    plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
    return

cuantizacion_ADC(B=4, kn=1/10)
