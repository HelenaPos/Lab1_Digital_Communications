# %% Librerías
import numpy as np
import sys
import time
from pathlib import Path

# Calculamos ruta del directorio padre (en el que se encuentra la librería)
ruta = Path(__file__).resolve().parent.parent
# Añadimos al path
sys.path.append(str(ruta))
# Librería propia
import Binary as bn
import Plot as plt
import Equalization as eq
import Filter as ft
import Channels as ch
import Symbols as sy
import Pulse as ps
import TransferFunctions as tf

# %% Inicio
print("")
print(" Sistema Syst_Equalizer")
print("")
# Inicio del contador de tiempo
start_time = time.time()
# %% Parametros
#
# Parámetros del pulso
#
# Frecuencia de símbolo (Hz)
symbol_freq = 10e3
# Factor de sobremuestreo
oversampling = 4
# Tipo de pulso
pulse_type = "RRC"  # "rectangular"  #
# Duración del pulso en periodos de símbolo
long = 10
# Exceso de ancho de banda
alpha = 0.8
#
# Parámetros de la constelación
#
# Número de símbolos de la constelación
nsymbols_const = 4
# Número de bits por símbolo
nbits_symbol = int(np.log2(nsymbols_const))
# Indicador de uso de codificación Gray
Gray = True
# Potencia de la constelación
energy_const = 1
#
# Parámetros generales de la simulación
#
# Frecuencia de símbolo (Hz)
symbol_freq = 10e3
# Número de símbolos tx
nsymbols_tx = 100000
# Número de símbolos de entrenamiento
nsymbols_training = 50000
# Respuesta del canal (g(t))
gain_cont = np.load("Channel_Cable.npy")
# Es/N0 en decibelios
Es_N0dB = 10
# Es/N0 en escala lineal
Es_N0 = 10 ** (Es_N0dB / 10)
# Potencia del ruido
N0 = energy_const / Es_N0
# Tipo de filtro adaptado ('tx' al pulso tx, 'rx' al pulso rx)
adapt_filter_type = "rx"
#
# Parámetros de los igualadores
#
# Número de coeficientes de los igualadores LE
ncoeffs_eq = 24
# Número de coeficientes de los igualadores DFE
ncoeffs_ff = int(ncoeffs_eq / 2)
ncoeffs_fb = int(ncoeffs_eq / 2)
# Retardo que incluye el igualador (LE:1 y DFE:2)
delaye1 = int(ncoeffs_eq / 2)
delaye2 = int(ncoeffs_eq / 2) - 1
# Escalado de la constante de adaptación
mu = 0.5
# %% Cálculos y constructores
# Régimen binario
binary_rate = symbol_freq * nbits_symbol #Número de símbolos transmitidos por
#segundo x Número de bits por símbolo
# Cálculo del pulso tx, rx y filtrado (por el filtro adaptado)
# Pulso tx
match pulse_type:
    case "rectangular":
        pulse_tx = np.ones(oversampling) / np.sqrt(oversampling)
    case "RRC":
        pulse_tx = ps.RRC_pulse(alpha, long, oversampling)
# Pulso rx: pulso_tx(t)*g(t)
pulse_rx = np.convolve(pulse_tx, gain_cont)
# Filtro adaptado (al pulso rx o tx)
match adapt_filter_type: #Filtro adapatado al pulso transmitido o recibido
    case "tx":
        filt_adapt = np.conj(pulse_tx[::-1])
    case "rx":
        filt_adapt = np.conj(pulse_rx[::-1])
# Pulso filtrado (pulso tx filtrado por el filtro adaptado): (p(t)*g(t))*h(t)
pulse_filt = np.convolve(pulse_rx, filt_adapt)
# Fase de muestreo del pulso filtrado
# (para máxima energía de la componente principal)
phase0 = np.argmax(np.abs(pulse_filt)) % oversampling
# Respuesta del canal equivalente discreto equivalente(hD[n]=convolución de )
disc_response = disc_response = ps.sampling(pulse_filt, oversampling, phase0%oversampling)
# Autocorrelación del ruido:Rw(t)=Nodelta(t), al pasar por el filtro adaptado 
# h(t), el ruido es w(t)*h(t) y su correlación Rw(t)*h(t)*h*(-t) y la 
#muestreamos en nTs
noise_autocorr = N0 * ps.sampling(
    np.convolve(filt_adapt, np.conj(filt_adapt[::-1])),
    oversampling
)


# Potencia de los símbolos recibidos (antes de la igualación): Ps de a[n]*hD[n]
#Ps=Es sumatorio del módulo al cuadrado del filtro
rx_symbols_power = energy_const * np.sum(np.abs(disc_response)**2)
# Potencia del ruido en los símbolos recibidos (antes de la igualación): La 
# potencia de una señal aleatoria es su autocorrelación en el retardo cero
rx_noise_power = np.real(noise_autocorr[len(noise_autocorr) // 2])
# Potencia total de la señal recibida
rx_power = rx_symbols_power + rx_noise_power
# Retardo del canal discreto equivalente
delayd = np.argmax(np.abs(disc_response))
# Retardo acumulado (canal equivalente discreto más igualador- LE y DFE)
delay = delayd + delaye1 
delay2 = delayd + delaye2  
# Respuestas diseñadas para LE_ZF,LE_MMSE, DFE_ZF y DFE_MMSE
h_LE_ZF = eq.LE_design(disc_response, noise_autocorr, energy_const,
    ncoeffs_eq, delay, False
)


h_LE_MMSE = eq.LE_design(disc_response, noise_autocorr, energy_const,
    ncoeffs_eq, delay, True
)


h_DFE_ZF_ff, h_DFE_ZF_fb = eq.DFE_design(disc_response, noise_autocorr,
    energy_const, ncoeffs_ff, ncoeffs_fb, delay2, False
)


h_DFE_MMSE_ff, h_DFE_MMSE_fb = eq.DFE_design(disc_response, noise_autocorr,
    energy_const, ncoeffs_ff, ncoeffs_fb, delay2, True
)


# Diseño ideal DFE-MMSE
freq_axis_DFE, DFE_ideal = eq.ideal_DFE_design(
    disc_response, noise_autocorr, energy_const
)
# Espectro del ruido
freq_axis_noise, noise_spectrum = tf.frequency_response(noise_autocorr, [1])
# Cálculos de las respuestas equivalenteas (tras igualación directa)
# LE
h_equiv_LE_ZF = np.convolve(disc_response, h_LE_ZF)
h_equiv_LE_MMSE = np.convolve(disc_response, h_LE_MMSE)
# DFE
h_equiv_DFE_MMSE = np.convolve(disc_response, h_DFE_MMSE_ff)
h_equiv_DFE_ZF = np.convolve(disc_response, h_DFE_ZF_ff)
#
## Constructrores
#
# Constelación
const = sy.QAM(nsymbols_const, energy_const, Gray)
# Filtro adaptado
adapt_filter = ft.Filter(filt_adapt)

#Paras las adaptadas
#mu'= mu/(Número constantesxPotenciamediadela señal que entra al filtro)
mu_LE = mu / (ncoeffs_eq * rx_power)
mu_DFE_ff = mu / (ncoeffs_ff * rx_power)
mu_DFE_fb = mu / (ncoeffs_fb * energy_const)

# DFE ZF
LE_ZF = eq.LE(const, h_LE_ZF)
# DFE MMSE
LE_MMSE = eq.LE(const, h_LE_MMSE)
# LE adaptativo
LE_adapt = eq.LE(const,np.zeros(ncoeffs_eq, dtype=complex),mu_LE)


# DFE ZF
DFE_ZF = eq.DFE(const,h_DFE_ZF_ff,h_DFE_ZF_fb)
# DFE MMSE
DFE_MMSE = eq.DFE(const,h_DFE_MMSE_ff,h_DFE_MMSE_fb)
# DFE adaptativo
DFE_adapt = eq.DFE(const, np.zeros(ncoeffs_ff, dtype=complex),
    np.zeros(ncoeffs_fb, dtype=complex), mu_DFE_ff, mu_DFE_fb
) 


# %% Sistema
###################
### Transmisor  ###
###################
# Símbolos de sincronismo: Generamos bits que sean 0 o 1
tx_bits =   np.random.randint(0, 2, nsymbols_tx * nbits_symbol)
tx_symbols = const.mapping(tx_bits)
# Señal pulsada transmitida: a[n]*p(t)
tx_signal = ps.shaping(tx_symbols, oversampling, pulse_tx)
############
## Canal ##
############
# Señal recibida
rx_signal = ch.selective_channel(tx_signal, gain_cont, N0) #Obtenemos
#tx_signal*g(t) + w(t)
################
### Receptor ###
################
# Filtro adaptado
rx_matched = adapt_filter.filtering_signal(rx_signal) #rx_signal*h(t)
# Diezmado
rx_samples = ps.sampling(rx_matched, oversampling, phase0)
###############################################################
## No igualación, igualación LE y DFE (ZF, MMSE y adaptativa)
#
## No igualación
#
# Símbolos decididos sin igualador
decided_symbols_plain = const.decision_symbol_seq(
    rx_samples[delayd : delayd + nsymbols_tx]
)
#
## LE
#
# rx_samples*hZF[n]
eq_symbols_LE_ZF, decided_symbols_LE_ZF = LE_ZF.equalization_decision(
    rx_samples, nsymbols_tx, delay
)


eq_symbols_LE_MMSE, decided_symbols_LE_MMSE = LE_MMSE.equalization_decision(
    rx_samples, nsymbols_tx, delay
)


eq_symbols_LE_adapt, decided_symbols_LE_adapt = LE_adapt.adaptive_equalization_decision(
    rx_samples, nsymbols_tx, delay, tx_symbols[:nsymbols_training]
)


#
## DFE
#
eq_symbols_DFE_ZF, decided_symbols_DFE_ZF = DFE_ZF.equalization_decision(
    rx_samples, nsymbols_tx, delay2
)

eq_symbols_DFE_MMSE, decided_symbols_DFE_MMSE = DFE_MMSE.equalization_decision(
    rx_samples, nsymbols_tx, delay2
)


eq_symbols_DFE_adapt, decided_symbols_DFE_adapt = DFE_adapt.adaptive_equalization_decision(
    rx_samples, nsymbols_tx, delay2, tx_symbols[:nsymbols_training]
)


#%% Medidas
#
## Respuestas impulsivas del los igualadores adaptativos
#
h_LE_adapt = LE_adapt.response()
h_DFE_adapt_ff, h_DFE_adapt_fb = DFE_adapt.response()
print("Coeficientes DFE:", h_DFE_adapt_ff)
print("¿Son finitos?", np.all(np.isfinite(h_DFE_adapt_ff)))
print("Máximo:", np.max(np.abs(h_DFE_adapt_ff)))
# Respuestas equivalentes tras la igualación
# (en el caso del DFE hasta la igualación directa)
h_equiv_LE_adapt = np.convolve(disc_response, h_LE_adapt)
h_equiv_DFE_adapt = np.convolve(disc_response, h_DFE_adapt_ff)
# Cálculo de SER
SER_plain = bn.error_rate(tx_symbols, decided_symbols_plain)
SER_LE_ZF = bn.error_rate(tx_symbols, decided_symbols_LE_ZF)
SER_LE_MMSE = bn.error_rate(tx_symbols, decided_symbols_LE_MMSE)
SER_LE_adapt = bn.error_rate(
    tx_symbols[nsymbols_training:], decided_symbols_LE_adapt
)
SER_DFE_ZF = bn.error_rate(tx_symbols, decided_symbols_DFE_ZF)
SER_DFE_MMSE = bn.error_rate(tx_symbols, decided_symbols_DFE_MMSE)
SER_DFE_adapt = bn.error_rate(
    tx_symbols[nsymbols_training:], decided_symbols_DFE_adapt
)
# Cálculo SINR
SINR_plain = -10 * np.log10(
    np.mean(
        (
            np.abs(
                tx_symbols[nsymbols_training:]
                - decided_symbols_plain[nsymbols_training:]
            )
            ** 2
        )
        / energy_const
    )
)
SINR_LE_ZF = -10 * np.log10(
    np.mean(
        (
            np.abs(
                tx_symbols[nsymbols_training:]
                - eq_symbols_LE_ZF[nsymbols_training:]
            )
            ** 2
        )
        / energy_const
    )
)
SINR_LE_MMSE = -10 * np.log10(
    np.mean(
        (
            np.abs(
                tx_symbols[nsymbols_training:]
                - eq_symbols_LE_MMSE[nsymbols_training:]
            )
            ** 2
        )
        / energy_const
    )
)
SINR_LE_adapt = -10 * np.log10(
    np.mean(
        (
            np.abs(
                tx_symbols[nsymbols_training:]
                - eq_symbols_LE_adapt[nsymbols_training:]
            )
            ** 2
        )
        / energy_const
    )
)
SINR_DFE_ZF = -10 * np.log10(
    np.mean(
        (
            np.abs(
                tx_symbols[nsymbols_training:]
                - eq_symbols_DFE_ZF[nsymbols_training:]
            )
            ** 2
        )
        / energy_const
    )
)
SINR_DFE_MMSE = -10 * np.log10(
    np.mean(
        (
            np.abs(
                tx_symbols[nsymbols_training:]
                - eq_symbols_DFE_MMSE[nsymbols_training:]
            )
            ** 2
        )
        / energy_const
    )
)
SINR_DFE_adapt = -10 * np.log10(
    np.mean(
        (
            np.abs(
                tx_symbols[nsymbols_training:]
                - eq_symbols_DFE_adapt[nsymbols_training:]
            )
            ** 2
        )
        / energy_const
    )
)

# %% Resultados
# Parámetros de la simulación
print("  Frec. de símbolo   : " + str(symbol_freq))
print("  Régimen binario    : " + str(binary_rate))
print("  Tipo de pulso      : " + pulse_type)
print("  Filtro adaptado    : " + adapt_filter_type)
print("  Es/N0              : " + str(Es_N0dB))
# Medidas de SINR
print("")
print("  SINR (No Eq)        : " + str(SINR_plain) + " dB")
print("  SINR (LE-ZF)        : " + str(SINR_LE_ZF) + " dB")
print("  SINR (LE-MMSE)      : " + str(SINR_LE_MMSE) + " dB")
print("  SINR (LE-adapt)     : " + str(SINR_LE_adapt) + " dB")
print("  SINR (DFE-ZF)       : " + str(SINR_DFE_ZF) + " dB")
print("  SINR (DFE-MMSE)     : " + str(SINR_DFE_MMSE) + " dB")
print("  SINR (DFE-adapt)    : " + str(SINR_DFE_adapt) + " dB")
print("")
# Medidas de SER
print("  SER (No Eq)         : " + str(SER_plain))
print("  SER (LE-ZF)         : " + str(SER_LE_ZF))
print("  SER (LE_MMSE)       : " + str(SER_LE_MMSE))
print("  SER (LE_adapt)      : " + str(SER_LE_adapt))
print("  SER (DFE-ZF)        : " + str(SER_DFE_ZF))
print("  SER (DFE_MMSE)      : " + str(SER_DFE_MMSE))
print("  SER (DFE_adapt)     : " + str(SER_DFE_adapt))
print("")
# Medida tiempo de ejecución
time = time.time() - start_time
print("")
print("  Tiempo de ejecución    : " + str(time) + " sec")
# %% Figuras
# Inicio figuras
plt.close()
#
## Representación de la respuesta del canal equivalente
# (incluido pulso tx y filtro adaptado)
# continua y discreta (tras el muestreo en rx)
#
plt.figure(1, suptitle="Canal continuo y discreto equivalente", maximize=True)
tn = np.arange(pulse_filt.size) / oversampling
plt.continuous_signal(
    pulse_filt, tn, styles=["b--", "m--"], legend="Canal continuo equivalente"
)
n = np.arange(disc_response.size) + (phase0 % oversampling) / oversampling
plt.discrete_signal(
    disc_response,
    n,
    colors=["r", "g"],
    legend="Canal discreto equivalente",
    title="Canal continuo/discreto equivalente",
)
#
## Diagrama de ceros de la respuesta discreta equivalente
#
plt.figure(
    2,
    suptitle=" Diagrama de ceros del canal discreto equivalente",
    maximize=True,
)
zeros = eq.roots(disc_response)
plt.poles_zeros(zeros, [], title="Diagrama de ceros")
#
## Respuestas de la etapa directa de los igualadores
#
plt.figure(
    3, suptitle="Respuestas impulsivas de los igualadores", maximize=True
)
plt.subplot(231)
plt.discrete_signal(h_LE_ZF, legend="LE-ZF")
plt.subplot(232)
plt.discrete_signal(h_LE_MMSE, legend="LE-MMSE")
plt.subplot(233)
plt.discrete_signal(h_LE_adapt, legend="LE-adapt")
plt.subplot(234)
plt.discrete_signal(h_DFE_ZF_ff, legend="DFE-ZF")
plt.subplot(235)
plt.discrete_signal(h_DFE_MMSE_ff, legend="DFE-MMSE")
plt.subplot(236)
plt.discrete_signal(h_DFE_adapt_ff, legend="DFE-adapt")
#
## Respuestas equivalentes tras la igualación
# (en el caso del DFE tras la etapa sin realimnsymbols_synchroación)
#
plt.figure(
    4,
    suptitle="Canales equivalentes tras la igualación (directa)",
    maximize=True,
)
plt.subplot(231)
plt.discrete_signal(
    h_equiv_LE_ZF,
    np.arange(h_equiv_LE_ZF.size) - delay,
    legend="LE_ZF",
)
plt.subplot(232)
plt.discrete_signal(
    h_equiv_LE_MMSE,
    np.arange(h_equiv_LE_MMSE.size) - delay,
    legend="LE_MMSE",
)
plt.subplot(233)
plt.discrete_signal(
    h_equiv_LE_adapt,
    np.arange(h_equiv_LE_adapt.size) - delay,
    legend="LE_adapt",
)
plt.subplot(234)
plt.discrete_signal(
    h_equiv_DFE_ZF,
    np.arange(h_equiv_DFE_ZF.size) - delay2,
    legend="DFE_ZF",
)
plt.discrete_signal(
    h_DFE_ZF_fb,
    np.arange(h_DFE_ZF_fb.size) + 1,
    colors=["g", "m"],
    legend="FB",
)
plt.subplot(235)
plt.discrete_signal(
    h_equiv_DFE_MMSE,
    np.arange(h_equiv_DFE_MMSE.size) - delay2,
    legend="DFE_MMSE",
)
plt.discrete_signal(
    h_DFE_MMSE_fb,
    np.arange(h_DFE_MMSE_fb.size) + 1,
    colors=["g", "m"],
    legend="FB",
)

plt.subplot(236)
plt.discrete_signal(
    h_equiv_DFE_adapt,
    np.arange(h_equiv_DFE_adapt.size) - delay2,
    legend="DFE_adapt",
)
plt.discrete_signal(
    h_DFE_adapt_fb,
    np.arange(h_DFE_adapt_fb.size) + 1,
    colors=["g", "m"],
    legend="FB",
)
#
## Respuestas en frecuencia del canal y los igualadores
# (en el caso del DFE de la etapa sin realimentación)
#
plt.figure(5, suptitle="Respuestas en frecuencia", maximize=True)
# Espectro del ruido
plt.spectrum(
    noise_spectrum,
    freq_axis_noise,
    style="k",
    legend="espectro ruido",
    log=True,
    power=True,
)
# Respuestas de los igualadores
plt.frequency_response(h_LE_ZF, style="g", legend="LE-ZF", log=True)
plt.frequency_response(h_LE_MMSE, style="g--", legend="LE-MMSE", log=True)
plt.frequency_response(h_LE_adapt, style="g:", legend="LE-adapt", log=True)
plt.frequency_response(h_DFE_ZF_ff, style="b", legend="DFE-ZF", log=True)
plt.frequency_response(h_DFE_MMSE_ff, style="b--", legend="DFE-MMSE", log=True)
plt.frequency_response(
    h_DFE_adapt_ff, style="b:", legend="DFE-adapt", log=True
)
# Respuesta del canal
plt.frequency_response(
    disc_response,
    title="Respuestas en frecuencia",
    style="c",
    legend="canal",
    log=True,
)
# Respuesta del DFE-MMSE ideal
plt.spectrum(
    DFE_ideal,
    freq_axis_DFE,
    style="m-.",
    legend="DFE-MMSE ideal",
    log=True,
    power=True,
)
#
## Constelaciones recibidas
#
plt.figure(
    6, suptitle="Constelaciones recibida", maximize=True, layout="constrained"
)
plt.subplot(241)
const.plot_symbols(rx_samples, title="No Eq")
plt.subplot(242)
const.plot_symbols(eq_symbols_LE_ZF, title="LE-ZF")
plt.subplot(243)
const.plot_symbols(eq_symbols_LE_MMSE, title="LE-MMSE")
plt.subplot(244)
const.plot_symbols(eq_symbols_LE_adapt[nsymbols_training:], title="LE-adapt")
plt.subplot(246)
const.plot_symbols(eq_symbols_DFE_ZF, title="DFE-ZF")
plt.subplot(247)
const.plot_symbols(eq_symbols_DFE_MMSE, title="DFE-MMSE")
plt.subplot(248)
const.plot_symbols(eq_symbols_DFE_adapt[nsymbols_training:], title="DFE-adapt")
#
## Error de adaptación
#
plt.figure(num=7, suptitle="Error de adaptación", maximize=True)
plt.subplot(121)
plt.continuous_signal(
    20 * np.log10(np.abs(eq_symbols_LE_adapt - tx_symbols) ** 2),
    ylabel="dB",
    title="Error del LE_adapt",
)
plt.subplot(122)
plt.continuous_signal(
    20 * np.log10(np.abs(eq_symbols_DFE_adapt - tx_symbols) ** 2),
    ylabel="dB",
    title="Error del DFE_adapt",
)
