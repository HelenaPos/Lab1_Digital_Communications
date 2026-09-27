##################################
# Plot
##################################
# Librerías
import numpy as np
import scipy.signal as sci
import scipy.special as scie

##################################
# Constantes
##################################
N_FREQ_POINT = 16 * 1024


def impulse_response(B, A, Nsamples):
    """
    Función que calcula un fragmento de la respuesta al impulso
    de un sistema estable definido por su función de transferencia recional
    H(z)=B(z)/A(z).

    Si los polos están dentro del círculo unidad, la respuesta es causal,
    pero si no, la respuesta es no causal pero estable.

    Parameters
    ----------
        B : array_like (dtype:float/complex)
            polinomio del numerador en potencia decrecientes de z:
                B[0]+B[1]z^-1+B[2]z^-2+...+B[n]z^-n
        A : array_like (dtype:float/complex)
            polinomio del denominador en potencia decrecientes de z:
                A[0]+A[1]z^-1+A[2]z^-2+...+A[n]z^-n
        Nsamples : int
            número de muestras de la respuesta impulsiva (si no es causal,
            se reparten entre instantes de tiempo positivos y negativos)
    Returns
    -------
        h : ndarray (dtype:float/complex)
            vector de muestras de la respuesta impulsiva
        n : ndarray (dtype:int)
            vector de índices temporales de la respuesta impulsiva.
            (tiene la misma longitud que h)
    """
    # Tolerancia
    tol = 1e-3
    # Convierte en ndarrays
    A = np.atleast_1d(A)
    B = np.atleast_1d(B)
    # Error si el numerador es nulo
    if np.all(A == 0):
        raise ValueError("Denominador no puede se igual a cero")
    # Deflacción del numerador (elimina polos en el origen)
    delayn = np.flatnonzero(B)[0]
    Bd = B[delayn:]
    # Deflacción del denominador (elimina ceros en el origen)
    delayd = np.flatnonzero(A)[0]
    # Elimino ceros que se compensan con retardo posterior
    Ad = A[delayd:]
    # Retardo total
    delay = delayd - delayn
    # Indicador de si el sistema es complejo o real
    Complex = (Ad.dtype == complex) or (B.dtype == complex)
    # Cálcula los polos, residuos y términos directos
    [residues, poles, direct] = sci.residuez(Bd, Ad)
    # Calcula los polos y su multiplicidad
    [poles_unique, multiplicity] = sci.unique_roots(poles)
    # Comprueba si es estable
    if np.any(np.abs(np.abs(poles) - 1) < tol):
        raise ValueError(" Polos muy cerca del círculo unidad ")
    # Inicia el vector de índices de la respuesta, dependiendo de
    # si los polos son fuera o dentro del círculo
    if np.any(np.abs(poles) > 1) & (np.any(np.abs(poles) < 1)):
        n = np.arange(Nsamples, dtype=int) - int(Nsamples / 2) + delay
    elif np.all(np.abs(poles) > 1):
        n = np.arange(Nsamples, dtype=int) - (Nsamples - 1) + delay
    else:
        n = np.arange(Nsamples, dtype=int) + delay
    # Inicia el vector de la respuesta
    if Complex:
        h = np.zeros(n.size, dtype=complex)
    else:
        h = np.zeros(n.size, dtype=float)
    # Bucle para obtener la respuesta del impulso a partir de los
    # términos asociado a la descomposición en fracciones simples de los
    # polos
    pole_index = 0
    for k in np.arange(poles_unique.size):
        for order in np.arange(multiplicity[k]):
            if np.abs(poles_unique[k]) > 1:
                h_k = (
                    residues[pole_index]
                    * ((-1.0) ** (order + 1))
                    * scie.comb(-n - 1, order)
                    * (poles_unique[k] ** n)
                    * ((-n - 1) >= 0)
                )
            else:
                h_k = (
                    residues[pole_index]
                    * scie.comb(n + order, order)
                    * (poles_unique[k] ** n)
                    * (n >= 0)
                )
            if Complex:
                h = h + h_k
            else:
                h = h + np.real(h_k)
            pole_index += 1
    for k in np.arange(direct.size):
        h[n == k] = h[n == k] + direct[k]
    # Compensación del retardo de deflacción
    n = n - delay

    return h, n


def frequency_response(B, A):
    """
    Función que calcula la respuesta en frecuencia de un sistema lineal
    discreto en el tiempo definido por su función de transferencia H(z)

    Parameters
    ----------
        B : array_like
            vector de coeficientes del numerador de la función de transferencia
            H(z) del sistema
        A : array_like
            vector de coeficientes del denominador de la función de
            transferencia H(z) del sistema

    Returns
    -------
        f : ndarray (dtype:float)
            vector del eje de frecuencias normalizado (entre -0.5 y 0.5)
        H : ndarray (dtype:complex)
            vector de la respuesta en frecuencia del mismo tamaño que f
    """
    # Cálculo de la respuesta en frecuencia
    w, H = sci.freqz(B, A, N_FREQ_POINT, whole=True)
    # Rotación de la respuesta
    H = np.fft.fftshift(H)
    # Rotación y escalado del eje
    f = w / (2 * np.pi) - 0.5

    return f, H
