# %% Librerías
import numpy as np
import sys
from pathlib import Path

# Calculamos ruta del directorio padre (en el que se encunsymbols_synchrora la librería)
ruta = Path(__file__).resolve().parent.parent
# Añadimos al path
sys.path.append(str(ruta))
# Librerías propias
import Binary as bn
import Equalization as eq
import Symbols as sy
import Channels as ch
import Plot as plt

# %% Inicio
print("")
print(" Sistema LE_Test ")
print("")
# Inicio de figuras
plt.close()
# %% Parámetros
# Número de símbolos de la constelación
nsymbols_const = 16
# Energía media de la constelación de símbolos
energy_const = 1
# Uso de codificación Gray
Gray = True
# Número de símbolos
nsymbols_tx = 40000
# Símbolos de entrenamiento
nsymbols_training = 8000
# Número de bits tx
nbits_tx = nsymbols_tx * int(np.log2(nsymbols_const))
# Es/N0 en decibelios
Es_N0dB = 20
# Es/N0 en escala lineal
Es_N0 = 10 ** (Es_N0dB / 10)
# Ruido
N0 = energy_const / Es_N0
# Número de coeficientes del igualador
ncoeffs = 20
# %% Constructores y cálculos
# Constelación
const = sy.QAM(nsymbols_const, energy_const, Gray)
# Respuesta impulsiva del canal
channel_impulse_response = np.array([1, 0.2], dtype=complex)
# Constante de adaptación
adapt_const = 0.1 / (ncoeffs * np.sum(np.abs(channel_impulse_response) ** 2))
# Equalizador fijo
LE_fixed = eq.LE(const, (-0.2) ** np.arange(ncoeffs))
# Equalizador adaptativo
LE_adaptive = eq.LE(const, np.zeros(ncoeffs, dtype=complex), adapt_const)
# %% Sistema
tx_bits = bn.binrand(nbits_tx, seed=1)
# Símbolos transmitidos
tx_symbols = const.mapping(tx_bits)
# Símbolos recibidos
rx_symbols = ch.selective_channel(
    tx_symbols, channel_impulse_response, N0, seed=1
)
# Igualación
eq_fixed_symbols, decided_fixed_symbol = LE_fixed.equalization_decision(
    rx_symbols, nsymbols_tx, 0
)
eq_adaptive_symbols, decided_adaptive_symbol = (
    LE_adaptive.adaptive_equalization_decision(
        rx_symbols, nsymbols_tx, 0, tx_symbols[:nsymbols_training]
    )
)
# %% Resultados
# Medida de la tasa de error
SER_fixed = bn.error_rate(decided_fixed_symbol, tx_symbols)
SER_adaptive = bn.error_rate(
    decided_adaptive_symbol, tx_symbols[nsymbols_training:]
)
# Tasas de error
print("")
print("  SER (fijo)        :  " + str(SER_fixed))
print("  SER (adaptativo)  :  " + str(SER_adaptive))
# %% Figuras
plt.figure(num=1, suptitle="LE test ", figsize=[14, 7.5], layout="constrained")
plt.subplot(231)
const.plot_symbols(eq_fixed_symbols, title="Símbolos igualados (fijo)")
plt.subplot(232)
const.plot_symbols(
    eq_adaptive_symbols[nsymbols_training:],
    title="Símbolos igualados (adaptativo)",
)
#
## Coeficientes
#
plt.subplot(233)
plt.discrete_signal(LE_fixed.Eq.coeffs, colors=["r"], legend="Fijos")
plt.discrete_signal(
    LE_adaptive.Eq.coeffs.real,
    colors=["b"],
    legend="Adapt.",
    title="Coeficientes del igualador",
)
#
## Respuesta equivalente
#
extended_impulse_response = np.block([np.array([1, 0.2]), np.zeros(20)])
ht = np.convolve(extended_impulse_response, LE_fixed.Eq.coeffs)
plt.subplot(234)
plt.discrete_signal(
    ht,
    np.arange(ht.size) - np.argmax(np.abs(ht)),
    title="Respuesta equivalente tras el igualador (fijo)",
)
#
## Error de adaptación
#
error = np.abs(eq_adaptive_symbols - tx_symbols) ** 2
plt.subplot(235)
plt.continuous_signal(
    10 * np.log10(np.abs(error[:nsymbols_training])),
    ylabel="dB",
    title="Error cuadrático de adaptación",
)
#
## Respuestas en frecuencia
#
plt.subplot(236)
plt.frequency_response(
    channel_impulse_response, style="b", log=True, legend="Canal"
)
plt.frequency_response(
    LE_fixed.Eq.coeffs, style="r", log=True, legend="LE_fijo"
)
plt.frequency_response(
    LE_adaptive.Eq.coeffs, style="g--", log=True, legend="LE_adapt."
)
