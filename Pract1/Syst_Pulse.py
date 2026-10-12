# %% Librerías
import numpy as np
import sys
import time
from pathlib import Path

# Calculamos ruta del directorio padre (en el que se encuentra la librería)
ruta = Path(__file__).resolve().parent.parent
# Añadimos al path
sys.path.append(str(ruta))
# Librería propias
import Plot as plt
import Binary as bn
import Channels as ch
import Symbols as sy
import Pulse as ps
import Filter as ft

# %% Inicio
print("")
print(" Sistema Syst_Pulse ")
print("")
# Inicio del contador de tiempo
start_time = time.time()
# %% Parametros
#
## Parámetros de la constelación
#
# Número de símbolos de la constelación
nsymbols_const = 16
# Potencia de la constelación
energy_symbols = 1
# Uso de codificación Gray
Gray = True
#
## Parámetros del pulso
#
# Tipo de pulso ('rectangular' o 'RRC')
pulse_type = "RRC"  # "RRC"  #
# Frecuencia de símbolo
symbol_freq = 10e3
# Duración en periodos de símbolo
long = 50
# Factor de sobremuestreo del pulso
oversampling = 32
# Exceso de ancho de banda
alpha = 0.1
#
## Parámetros generales de la simulación
#
# Número de símbolos tx
nsymbols_tx = 1000000
# Número de bits tx
nbits_tx = nsymbols_tx * int(np.log2(nsymbols_const))
# SNR en decibelios
Es_N0_dB = 12
# SNR en escala lineal
Es_N0 = 10 ** (Es_N0_dB / 10)
## Parámetros del canal
# Potencia del ruido
N0 = energy_symbols / Es_N0
# Respuesta del canal (plano)
g = np.array([1])
# Frecuencia de muestreo
sampling_freq = symbol_freq * oversampling #Tm=MTs
# %% Cálculos y constructores
# Vector de muestras del pulso
match pulse_type:
    case "rectangular":  # Pulso rectangular
        pulse = np.block([np.ones(oversampling) / np.sqrt(oversampling), 0])
    case "RRC":  # Pulso RRC
        pulse = ps.RRC_pulse(alpha, long, oversampling)
# Autocorrelación del pulso
Phi = np.convolve(pulse, np.conj(pulse[::-1]))
Phi_sampled = ps.sampling(Phi, oversampling)
# Constelación
const = sy.QAM(nsymbols_const, energy_symbols, Gray) #const contains the rules 
# of the modulation 16-QAM
# Filtro
adapt_filt = ft.Filter(np.conj(pulse[::-1])) # Adaptative filter: h[k]=p*[-k]
# %% Sistema
###################
### Transmisor  ###
###################
# Generación de bits tx
tx_bits = bn.binrand(nbits_tx)
# Generación de símbolos QAM tx
tx_symbols = const.mapping(tx_bits)
# Conformación de pulsos
tx_signal = ps.shaping(tx_symbols, oversampling, pulse) #We get the 
# amplitude-modulated signal
###################
###   Canal     ###
###################
rx_signal = ch.selective_channel(tx_signal,g,N0) # Signal + noise
################
### Receptor ###
################
# Filtro adaptado
rx_matched = adapt_filt.filtering_signal(rx_signal)
# Muestreo
rx_symbols = ps.sampling(
    rx_matched, oversampling, pulse.size - 1, tx_symbols.size
)
# Decisor
decided_symbols = const.decision_symbol_seq(rx_symbols)
# Demapeo de bits
rx_bits = const.demapping(decided_symbols)
################
### Medidas  ###
################
# Tasa de errores de la simulación
SER = bn.error_rate(tx_symbols, decided_symbols)
# Cálculo teórico de la tasa de errores de la simulación
SERt = const.SERt(Es_N0)
# Tasa de errores de la simulación
BER = bn.error_rate(tx_bits, rx_bits)
# %% Resultados
# Parámetros
print(" Parámetros:")
print("")
# frecuencia de símbolo
print("  Pulso                      : " + pulse_type)
if pulse_type == "RRC":
    print("    alpha                    : " + str(alpha))
    print("    duración                 : " + str(long) + " (x Ts)")
print("  Frecuencia de símbolo      : " + str(symbol_freq / 1e3) + " kHz")
print("  Es/N0                      : " + str(Es_N0_dB) + " dB")
# Signal to Interference and Noise Ratio
SINR = -10 * np.log10(
    np.mean(np.abs(tx_symbols - rx_symbols) ** 2) / energy_symbols
)
print("")
print(" Medidas:")
print("")
print("  SINR                       : " + str(SINR) + " dB")
# Symbol/Bit Error Rate
print("  SER                        : " + str(SER))
print("  SERt                       : " + str(SERt))
print("  BER                        : " + str(BER))
print("")

# Tiempo de ejecución
time = time.time() - start_time
print("")
print(" Tiempo de ejecución         : " + str(time) + " sec")
# %% Figuras
# Inicio figuras
plt.close()
#
## Pulso transmitido
#
plt.figure(1, suptitle="Pulso transmitido")
# Extensión con ceros del pulso
pulse_ext = np.block(
    [np.zeros(2 * oversampling), pulse, np.zeros(2 * oversampling)]
)
plt.continuous_signal(
    pulse_ext,
    (np.arange(pulse_ext.size) - 2 * oversampling) / oversampling,
    title="Pulso tx ",
    time_unit="muestras",
)
#
## Autocorrelación de los pulsos
#
plt.figure(2, suptitle="Autocorrelación del pulso")
Phi, Phid, n = ps.autocorrelation(pulse, oversampling)
plt.continuous_signal(
    Phi,
    n,
    styles=["b", "b"],
    ylabel="autocorrelación",
    legend="autocorrelación",
)
plt.discrete_signal(
    Phid,
    n[(n % 1) == 0],
    colors=["r", "r"],
    ylabel="autocorrelación",
    title="Autocorrelación del pulso (antes y después del muestreo)",
    legend="autocorrelación muestreada",
    time_unit="x $T_S$",
)
#
## Espectro de la señal transmitida
#
plt.figure(3, suptitle="Espectro de la señal transmitida ")
freq_axis, spectrum_tx = ps.spectrum_estimation(tx_signal, pulse.size * 100)
plt.spectrum(
    spectrum_tx / np.max(spectrum_tx),
    freq_axis / 1e3,
    style="r",
    fm=sampling_freq,
    freq_unit="kHz",
    xlim=[-3.5 * symbol_freq / 1e3, 3.5 * symbol_freq / 1e3],
    ylim=[-90, 5],
    title="Espectro de la señal transmitida",
    power=True,
)
#
## Símbolos recibidos
#
plt.figure(4, suptitle="Símbolos recibidos")
const.plot_symbols(rx_symbols, title="Símbolos rx")
#
## Fragmento de la señal transmitida
#
plt.figure(
    5, suptitle="Fragmento del señal transmitida (parte real)", 
)
ps.plot_pulsed_signal(tx_signal, pulse, oversampling, tx_symbols, 20, "tx")
#
## Fragmento de la señal recibida
#
plt.figure(
    6, suptitle="Fragmento del señal recibida (parte real)", 
)
ps.plot_pulsed_signal(rx_signal, pulse, oversampling, rx_symbols, 20, "rx")
