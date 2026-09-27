##################################
# Binary
##################################
"""
    Módulo para el cálculo de operaciones básicas con vectores/matrices 
    binarios.
"""
##################################
# Librerías
##################################
import numpy as np
import sympy as sp
import itertools as it


##################################
# Funciones
##################################
def binrand(size, seed=None):
    """
    Función que genera un vector de números binarios aleatorios con
    distribución equiprobable (misma probabilidad para el 0 y el 1).

    Parameters
    ----------
        size: int
            tamaño del vector que se va a generar
        seed: int -optional, default:None-
            semilla del generador pseudoaleatorio
    Returns
    ----------
        array_bin : ndarray -dtype np.byte-
                vector de binarios aleatorios de tamaño 'size'
    """

    # Generador aleatorio
    rng = np.random.default_rng(seed)
    # Generación de binarios equiprobables
    array_bin = rng.integers(0, 2, size, dtype=np.byte)
    return array_bin


def dec2bin(array_dec, outbits):
    """
    Función que convierte números enteros no negativos en vectores
    de números binarios.

    En cada vector binario, los bits están ordenados de mayor a menor
    peso.

    Parameters
    ----------
        array_dec : array_like -dtype int-
            array de números enteros no negativos.
        outbits : int -positive-
            número de bits para representar cada valor decimal
    Returns
    -------
        array_bin : ndarray -dtype np.byte-
            array formado por vectores de números binarios,
            cada uno de los cuales representan un valor de entrada.
    """

    # Array_like a ndarray y hace copia
    array_dec = np.asarray(array_dec, dtype=int).copy()
    # Bucle de cálculo de bits
    binvalues = 2 ** (np.arange(outbits - 1, -1, -1))
    array_bin = (
        np.floor_divide(
            array_dec.reshape(*array_dec.shape, 1), binvalues, dtype=int
        )
        % 2
    ).astype(np.byte)
    return array_bin


def bin2dec(array_bin):
    """
    Función que convierte vectores de números binarios en números enteros no
    negativos.

    El vector de entrada, o cada vector fila si la entrada es un array,
    se convierte en número entero no negativo, considerando que los bits
    en cada vector están ordenados en la fila de mayor a menor peso.

    Parameters
    ----------
        array_bin : array_like
            array de número binarios
    Returns
    ----------
        array_dec : ndarray -dtype int-
            array de enteros no negativos
    """

    # Array_like a ndarray 2D
    array_bin = np.asarray(array_bin)
    # Calcula pesos de los bits
    binvalues = 2 ** (np.arange(array_bin.shape[-1] - 1, -1, -1))
    # Computa valor dewcimal
    array_dec = np.sum(array_bin * binvalues, axis=-1)
    # Elimina dimensiones sobrantes
    array_dec = np.squeeze(array_dec)
    return array_dec


def Hammingdist(array_1, array_2):
    """
    Función que calcula la distancia de Hamming entre arrays, comparando sus
    vectores fila binarios.

    Parameters
    ----------
        array_1,array_2 : array_like
            array de binarios (array_1 y array_2 ó tienen el mismo tamaño ó se
            puede hacer broadcasting con ellos, pero en la última dimensión
            deben tener el mismo tamaño)
    Returns
    -------
        dist : ndarray, -dtype int-
            escalar/array de distancias de Hamming
    """
    # Array_like a ndarray
    array_1 = np.asarray(array_1, dtype=np.byte)
    array_2 = np.asarray(array_2, dtype=np.byte)
    # Suma modulo 2
    summod2 = (array_1 + array_2) % 2
    # Valores distintos = suma de 1's
    dist = np.sum(summod2, axis=-1)
    return dist


def minHammingdist(table):
    """
    Función que cálcula la mínima distancia de Hamming de los vectores de un
    tabla

    Parameters
    ----------
        table : array_like
            tabla de vectores, definida por un array cuyos vectores queremos
            comparar para calcular la mínima distancia de Hamming
    Returns
    -------
        min_dist : int -positive-
            distancia de Hamming mínima entre los vectores del array
    """
    # Array_like a ndarray
    table = np.asarray(table)
    # Calcula la distancia de Hamming de todas las combinaciones de palabras
    dist = Hammingdist(
        table.reshape((-1, table.shape[-1])),
        table.reshape((-1, 1, table.shape[-1])),
    )
    # Calcula distancia mínima
    min_dist = np.min(dist[dist > 0])
    return min_dist


def randomizer(array_in, seed=1):
    """
    Función que aleatoriza un array de números binarios.

    Se aleatoriza sumando módulo 2 una secuencia aleatoria de
    0's y 1's, generados con una semilla. Si se vuelve a aplicar la
    aleatorización con la misma semilla se recupera la secuencia original.

    Parameters
    ----------
        array_in : array_like
            array de números binarios
        seed: int -optional, default:1-
            semilla del generador pseudoaleatorio
    Returns
    ----------
        array_out: ndarray -dtype np.byte-
            array de números binarios aleatorizados
    """
    # Array_like a ndarray
    array_in = np.asarray(array_in, dtype=np.byte)
    # Generador aleatorio
    rng = np.random.default_rng(seed)
    # Generación del patrón aleatorizador
    rand_sec = rng.integers(0, 2, array_in.shape, dtype=np.byte)
    array_out = (array_in + rand_sec) % 2
    return array_out


def interleaver(bin_vector_in, seed):
    """
    Función que realiza el entrelazado de un vector binario permutando las
    posiciones de los bits.

    Parameters
    ----------
        bin_vector_in: array_like
            vector binario cuyas posiciones van a permutar
        seed: int
            semilla del generador aleatorio que genera el orden aleatorio
            de permutación
    Returns
    --------
        bin_vector_out : ndarray -dtype np.byte-
            vector binario de posiciones permutadas
    """

    bin_vector_in = np.asarray(bin_vector_in)
    # Generador aleatorio
    rng = np.random.default_rng(seed)
    # Orden de permutación
    sequence = np.arange(bin_vector_in.size)
    order = rng.permutation(sequence)
    # Permutación
    bin_vector_out = bin_vector_in[order]
    return bin_vector_out


def deinterleaver(bin_vector_in, seed):
    """
    Función que realiza el desentrelazado de un vector binario permutando las
    posiciones de los bits.

    Parameters
    ----------
        bin_vector_in: array_like
            vector binario cuyas posiciones van a permutar para volver a la
            posición original
        seed: int
            semilla del generador aleatorio que genera el orden aleatorio
            de permutación
    Returns
    --------
        bin_vector_out : ndarray -dtype np.byte-
            vector binario de posiciones permutadas para volver a las
            posiciones originales
    """
    bin_vector_in = np.asarray(bin_vector_in)
    # Generador aleatorio
    rng = np.random.default_rng(seed)
    # Orden para desahacer la permutación
    sequence = np.arange(bin_vector_in.size)
    order = rng.permutation(sequence)
    reorder = np.empty(bin_vector_in.size, dtype=int)
    reorder[order] = sequence
    # Permutación
    bin_vector_out = bin_vector_in[reorder]
    return bin_vector_out


def binnullspace(matrix):
    """
    Función que calcula la matriz del espacio nulo correspondiente a la matriz
    de entrada.

    Parameters
    ----------
        matrix : array_like
            matriz binaria de entrada
    Returns
    -------
        matrix_null : ndarray -dtype np.byte-
            matriz del espacio nulo

    """
    # Array_like a matriz
    matrix = np.atleast_2d(np.asarray(matrix))
    # Convertir la matriz a un formato de sympy
    matrix_sp = sp.Matrix(matrix)
    # Calcular el espacio nulo (null space) de la matriz
    matrix_null_sp = matrix_sp.nullspace()
    # Convertir el resultado a una matriz numpy para facilitar su uso
    if matrix_null_sp:
        matrix_null = np.asarray(matrix_null_sp, dtype=np.byte) % 2
        matrix_null = np.atleast_2d(np.squeeze(matrix_null))
        return matrix_null.T


def bincombinations(nbits_total, nbits_ones):
    """
    Función que devuelve una matriz, cuyas filas son todas las combinaciones
    posibles de vectores binarios de tamaño 'nbits_total' y 'nbits_ones' unos.

    Parameters
    ----------
        nbits_total : int -positive-
            tamaño de los vectores
        nbits_ones : int -positive-
            número de unos
    Returns
    -------
        matrix : ndarray -dtype np.byte-
            matriz cuyas filas son todas las combinaciones de vectores binarios
    """
    # Indices de valores a 1
    index = np.array(list(it.combinations(range(nbits_total), nbits_ones)))
    # Saltos de fila
    row_jumps = np.arange(
        0, nbits_total * (index.shape[0] - 1) + 1, nbits_total
    ).reshape(-1, 1)
    # Indice del vector plano
    flat_index = (index + row_jumps).ravel()
    # Vector plano con unos
    vector = np.zeros(nbits_total * index.shape[0], np.byte)
    vector[flat_index] = 1
    # Martriz final
    matrix = vector.reshape(-1, nbits_total)
    return matrix


def bin2str(array_bin):
    """
    Función que convierte un array de binarios en cadena de caracteres.

    Parameters
    ----------
        array_bin : array_like
            array de elementos binarios
    Returns
    -------
        string : str
            cadena de caracteres correspondiente al array binario
    Raises
    ------
        ValuesError
            La entrada contiene valores no binarios
    Examples
    --------
        .. jupyter-execute::
            :emphasize-lines: 3

            import digcomm as dc
            a=[1,0,1,1,0]
            string=dc.bin2str(a)
            print('String : '+string)

        .. jupyter-execute::
            :emphasize-lines: 2

            a=[[1,1],[1,0],[0,1],[0,0]]
            string=dc.bin2str(a)
            print('String : '+string)
    """
    # Array_like a ndarray vector
    array_bin = np.asarray(array_bin).ravel()
    # Bucle para incluir los elementos convertidos en caracteres
    string = ""
    for k in np.arange(array_bin.size):
        string = string + str(array_bin[k])
    return string


def error_rate(vector1, vector2):
    """
    Función que calcula la tasa de diferencias entre dos vectores
    (de bits, símbolos, etc.).

    Parameters
    ----------
        vector1, vector2 : ndarray (dtype:float/complex/int)
            vectores que comparamos
     Returns
     -------
         rate : float
             tasa de errores
    """
    if vector1.size != vector2.size:
        print([vector1.size, vector2.size])
        raise ValueError("Vectores de tamaño distinto")

    rate = np.sum(vector1 != vector2) / vector1.size
    return rate
