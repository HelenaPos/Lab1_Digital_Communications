##################################
# Channels
##################################
"""
    Módulo para la simulación de canales selectivos en frecuencia.
"""
##################################
# Librerías
##################################
import numpy as np


##################
# Funciones
##################
def selective_channel(tx_samples, continuous_response, N0, seed=1):
    """
    Función que simula un canal selectivo en frecuencia con ruido AWGN

    Parameters
    ----------
        tx_samples: array_like
            vector de muestras de la señal 'continua'
        continuous_response : array_like
            vector de muestras de la respuesta  al impulso equivalente
            'continua' del canal
        N0 : float
            potencia del ruido en la banda
        seed : int
            semilla del generador aleatorio del ruido
    Returns
    -------
        rx_samples : ndarray (dtype : float/complex)
            vector de muestras de salida 'continua'
    """
    tx_samples = np.asarray(tx_samples)
    continuous_response = np.asarray(continuous_response)
    # Generador aleatorio para el ruido
    rng = np.random.default_rng(seed)
    # Convolución de la señal de entrada con la respuesta al impulso del
    # canal
    rx_samples = np.convolve(continuous_response, tx_samples)
    # Si es complejo suma ruido complejo y si es real ruido real
    if rx_samples.dtype == complex:
        rx_samples = rx_samples + np.sqrt(N0 / 2) * (
            rng.normal(0, 1, rx_samples.size)
            + 1j * rng.normal(0, 1, rx_samples.size)
        )
    else:
        rx_samples = rx_samples + np.sqrt(N0) * rng.normal(
            0, 1, rx_samples.size
        )

    return rx_samples


def MIMO_flat_channel(tx_samples, matrix_response, N0, seed=1):
    """
    Función que simula un canal selectivo en frecuencia con ruido AWGN.

    Parameters
    ----------
        tx_samples: array_like
            vector de muestras de la señal 'continua'
        matrix_response : array_like
            matriz de ganancias del canal MIMO
            'continua' del canal
        N0 : float
            potencia del ruido en la banda
        seed : int
            semilla del generador aleatorio del ruido
    Returns
    -------
        rx_samples : ndarray (dtype : float/complex)
            vector de muestras de salida 'continua'
    """
    tx_samples = np.atleast_2d(tx_samples)
    nsamples,Nt=tx_samples.shape
    matrix_response = np.asarray(matrix_response)
    Nt,Nr=matrix_response.shape
    # Generador aleatorio para el ruido
    rng = np.random.default_rng(seed)
    # Ruido en el receptor
    noise=np.sqrt(N0 / 2) * (
        rng.normal(0, 1, (nsamples,Nr))
        + 1j * rng.normal(0, 1, (nsamples,Nr))
    )
    # Muestras recibidas
    rx_samples=(tx_samples @ matrix_response)+noise

    return rx_samples


def binary_channel(tx_bits,pe,seed=None):
    """
    Función que simula el comportamiento de un canal binario simétrico.

    Parameters
    ----------
        tx_bits : array_like
            vector de bits transmitidos
    
        pe : float
            probabilidad de error
        
        seed : int
            semilla del generador aleatorio

    Returns
    -------
        rx_bits : ndarray (dtype: np.byte)
            vector de bits recibidos

    """
    
    tx_bits=np.asarray(tx_bits,dtype=np.byte)
    
    rng=np.random.default_rng(seed)
    
    rx_bits=(tx_bits+rng.choice(np.array([0,1],dtype=np.byte),
                                size=tx_bits.shape,
                                p=[1-pe,pe])
             )%2
    
    return rx_bits
             
        
    