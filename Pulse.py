##################################
# Pulse
##################################
# Librerías
import numpy as np
import scipy.signal.windows as wd
import Plot as plt


###################################
# Funciones
###################################
def root_raised_cos(alpha, t):
    """
    Función de raíz de coseno alzado.

    Parameters
    ----------
        alpha : float
            exceso de ancho de banda (parámetro de forma de los pulsos RRC)
        t : array_like
            eje de tiempo normalizado al periodo de símbolo (x T_S)
    Returns
    -------
        pulse : ndarray (dtype:float)
            vector de muestras del pulso
    """
    # Array_like a ndarray
    t = np.asarray(t, dtype=float)
    # Eje temporal del pulso
    pulse = np.zeros(t.size, dtype=float)
    # 1ª Condición de división por cero
    index_a = t == 0
    pulse[index_a] = 1 - alpha + 4 * alpha / np.pi
    # 2ª Condición de división por cero
    index_b = np.abs(4 * alpha * t) == 1
    pulse[index_b] = (
        alpha
        / np.sqrt(2)
        * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
        )
    )
    # Resto de instantes de tiempo
    index_c = np.logical_and(np.logical_not(index_a), np.logical_not(index_b))
    pulse[index_c] = (
        np.sin(np.pi * t[index_c] * (1 - alpha))
        + 4 * alpha * t[index_c] * np.cos(np.pi * t[index_c] * (1 + alpha))
    ) / (np.pi * t[index_c] * (1 - (4 * alpha * t[index_c]) ** 2))

    return pulse


def RRC_pulse(alpha, long, oversampling):
    """
    Función que devuelve pulsos raíz de coseno alzado (RRC).

    Parameters
    ----------
        alpha : float
            exceso de ancho de banda (parámetro de forma de los pulsos
            RRC), definido en el intervalo [0,1]
        long : int
            duración en periodos de símbolo (número par)
        oversampling : int
            factor de sobremuestreo de los pulsos
    Returns
    -------
        pulse : ndarray (dtype:float)
            vector de muestras del pulso
    """
    # Eje temporal del pulso
    axis = (
        np.arange(-long / 2 * oversampling, long / 2 * oversampling + 1)
        / oversampling
    )
    pulse = root_raised_cos(alpha, axis)
    # Normalización en energía
    pulse = pulse / np.sqrt(np.sum(np.abs(pulse) ** 2))

    return pulse


def raised_cos(alpha, t):
    """
    Función que devuelve la función de coseno alzado (RC),
    autocorrelación de la función raíz de coseno alzado.
                        .
    Parameters
    ----------

        alpha : float
            factor de exceso de ancho de banda (parámetro de forma de los
            pulsos RC), definido en el intervalo [0,1]
        t : array_like
            vector de instantes de tiempo

    Returns
    -------
        pulse : ndarray (dtype: float)
            pulso de coseno alzado
    """
    # Eje de tiempo como array
    t = np.asarray(t, dtype=float)
    # Fórmula general
    num = np.sinc(t) * np.cos(np.pi * alpha * t)
    den = 1 - (2 * alpha * t) ** 2
    # Expresión del pulso
    pulse = num / den
    # Caso t próximo a cero
    index0 = np.isclose(t, 0.0)
    pulse[index0] = 1.0
    # Caso t = ±1/(2*alpha)
    if alpha != 0:
        t_asynt = 1 / (2 * alpha)
        index_t_asynt = np.isclose(np.abs(t), t_asynt)
        pulse[index_t_asynt] = (alpha / 2) * np.sin(np.pi / (2 * alpha))

    return pulse


def autocorrelation(pulse, oversampling):
    """
    Función que calcula la autocorrelación de un pulso y calcula la
    versión diezmada a frecuencia de símbolo para comprobar como cumple el
    criterio de Nyquist.

    Parameters
    ----------
        pulse : array_like
            vector de muestras del pulso
        oversampling : int (positive)
            factor de sobremuestreo del pulso (con respecto a la
            frecuencia de símbolo).
    Returns
    -------
        Phi : ndarray (dtype:float/complex)
            vector de muestras de la autocorrelación
        Phi_d :  ndarray (dtype:float/complex)
            vector de muestras de la autocorrelación diezmada a frecuencia
            de símbolo.
        n : ndarray (dtype:int)
            vector de índices de las muestras de la autocorrelación
    """
    # Array_like a ndarray
    pulse = np.asarray(pulse)
    # Cálculo de la autocorrelación
    Phi = np.convolve(pulse, np.conj(pulse[::-1]))
    # Indice del origen de la autocorrelación
    ini_index = np.argmax(np.abs(Phi))
    # Eje de tiempos de la autocorrelación
    n = (np.arange(Phi.size) - int((Phi.size - 1) / 2)) / oversampling
    # Autocorrelación diezmada
    Phi_d = sampling(Phi, oversampling, ini_index % oversampling)

    return Phi, Phi_d, n


def shaping(symbols, oversampling, pulse):
    """
    Función que conforma una señal formada por una secuencia de pulsos,
    cada uno modulado en amplitud por una amplitud de un símbolo.

    Parameters
    ----------
        symbols : ndarray (dtype:float/complex)
            vector de amplitudes de los símbolos
        oversampling : int
            separación en muestras entre pulsos (Ts=MTm)
        pulse : ndarray (dtype:float/complex)
            vector de muestras del pulso
    Returns
    -------
        samples_vector : ndarray (dtype:float/complex)
            vector de muestras de la señal pulsada
    """
    # Conformador de pulsos (Pulse shaper)
    symbols = np.asarray(symbols)
    pulse = np.asarray(pulse)
    symbols_upsampled = np.zeros(symbols.size*oversampling, dtype=symbols.dtype)
    #Just in case it is a complex coefficient, because if we do not specify, 
    #we might get floats
    symbols_upsampled[::oversampling]=symbols #::x takes positions starting 
    # at 0, advancing in steps of M, so we will get M-1 zeros between samplings
    samples_vector = np.convolve(symbols_upsampled, pulse) # It gives N+L-1 
    # samples
    samples_vector=samples_vector[:symbols_upsampled.size] # We onl
    return samples_vector
    



def sampling(samples_vector_in, oversampling, ini=None, noutputs=None):
    """
    Función que muestrea un vector de muestras de una señal.

    Parameters
    ----------
        samples_vector_in : array_like
            vector de muestras de la señal de entrada
        oversampling :  int
            factor por el que se va a reducir la frecuencia de muestreo
            de la señal de entrada
        ini :  int (optional - default:None)
            muestra inicial para el muestreo periódico. Si no se proporciona   
            se usa la fase de muestreo que permita inculir la muestra que 
            contenga el valor máximo del vector de muestras
        noutputs : int (optional - default:None)
            número de muestras de salida
    Returns
    -------
        samples_vector_out : ndarray (dtype:float/complex)
            vector de muestras de la señal muestreada de tamaño
            noutputs
    """
    # Array_like a ndarray
    samples_vector_in = np.asarray(samples_vector_in)
    # Muestreo
    if ini is None:
        ini = np.argmax(np.abs(samples_vector_in))% oversampling
    samples_vector = samples_vector_in[ini::oversampling]
    if noutputs is not None:
        samples_vector_out = samples_vector[:noutputs]
    else:
        samples_vector_out = samples_vector
    return samples_vector_out


def spectrum_estimation(signal, window_length):
    """
    Función que estima el espectro de una señal estacionaria.

    Estima el espectro promediando el modulo al cuadrado de la
    transformada de Fourier de segmentos consecutivos de la señal.
    Gracias a ese promediado se reduce la fluctuación que tiene el
    espectro de la señal (error debido a la estimación con segmentos
    finitos). La estimación será mejor cuanto mayor es la duración de
    la realización de la señal que se proporciona.

    En cuanto a la longitud de la ventana de promediado hay un compromiso
    entre fluctuación y resolución espectral: si la longitud de la ventana
    es mayor la resolución espectral es mayor (mayor número de puntos de
    la DFT) pero la varianza tambien es mayor porque se promedian menos
    segmentos; si la longitud de la ventana es menor la varianza se reduce
    al promediar más segmentos pero la resolución espectral disminuye
    (menor número de puntos de la DFT).

    Parameters
    ----------
        signal : array_like (float or complex)
            vector de muestras de la señal
        window_lenght : int (positive)
            longitud de la ventana que se va a usar para promediar el
            espectro (potencia de 2)
    Returns
    -------
        freq : ndarray (dtype:float)
            vector del eje de frecuencia entre [-0.5,0.5)
        spectrum :  ndarray (dtype:float)
            vector de muestras de la estimación de la densidad espectral
            de potencia

    """
    # Array_like a ndarray
    signal = np.asarray(signal)
    nframes = int(signal.size / window_length)
    # División de la señal en fragmentos de longitud Long
    matrix = signal[: nframes * window_length].reshape(-1, window_length)
    # Enventanado de Hanning
    matrix_windowed = matrix * wd.hann(window_length)
    # Cálculo del espectro
    spectrum = np.mean(
        np.fft.fftshift(
            np.abs(np.fft.fft(matrix_windowed, axis=1)) ** 2, axes=1
        ),
        axis=0,
    )
    # Cálculo del eje de frecuencia
    freq = np.arange(spectrum.size) / spectrum.size - 0.5

    return freq, spectrum


def plot_pulsed_signal(
    pulsed_signal, pulse, oversampling, symbols, nsymbols_rep, title=''
):
    """
    Función que representa una señal pulsada y su descomposición en una
    secuencia de pulsos modulados por los símbolos (sólo parte real).

    Parameters
    ----------
        pulsed_signal : array_like
            vector de muestras de la señal pulsada continua
        pulse: array_like
            vector de muestras del pulso
        oversampling: int
            factor de sobremuestreo del pulso
        symbols : array_like
            vector de símbolos
        nsymbols_rep: int
            número de símbolos que se van a representar
        title : str (optional,default='')
            cadena de caracteres que identifica la señal representada

    """
    # Array_like a ndarray
    pulsed_signal = np.asarray(pulsed_signal)
    pulse = np.asarray(pulse)
    symbols = np.asarray(symbols)
    # Duración del pulso
    long = int((pulse.size - 1) / oversampling)
    # Índice de retardo inicial
    dt0 = 2 * long * oversampling
    n0 = 2 * long
    # Eje de tiempo
    t = np.arange(nsymbols_rep * oversampling)
    # Fragmento de la señal pulsada
    plt.continuous_signal(
        np.real(pulsed_signal[dt0 : dt0 + nsymbols_rep * oversampling]),
        (t + dt0) / oversampling,
        styles=["b", "b"],
        xlim=(np.array([t[0], t[-1]]) + dt0) / oversampling,
        title="Fragmento de la señal "+title,
    )
    # Secuencia de pulsos que forman la señal pulsada
    t_pulse = np.arange(pulse.size)
    for k in np.arange(-long, nsymbols_rep + long):
        plt.continuous_signal(
            np.real(symbols[n0 + k]) * pulse,
            (t_pulse + k * oversampling + dt0) / oversampling,
            styles=["g--", "g--"],
        )
