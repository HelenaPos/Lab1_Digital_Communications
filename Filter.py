##################################
# Filter
##################################
"""
Módulo para la implementación de filtros digitales fijos o adaptativos.
"""
##################################
# Librerías
##################################
import numpy as np
import scipy.linalg as scia

##################
# Clase Filter
##################
class Filter:
    """
    Clase que implementa la funcionalidad de un filtro FIR que
    puede ser de coeficientes fijos o adaptativos.
    """

    def __init__(self, coeffs, adapt_const=None):
        """
        Constructor de la clase Filter.

        Parameters
        ----------
            coeffs : array_like
                vector de coeficientes del filtro FIR
            adapt_const : float
                constante de adaptación
        """
        # Array_like a ndarray
        coeffs = np.asarray(coeffs)
        # Vector de coeficientes
        self.coeffs = coeffs
        # Número de coeficientes
        self.ncoeffs = coeffs.size
        # Potencia de 2 mayor y más cercana a ncoeffs
        self.buffer_size = int(2 ** np.ceil(np.log2(self.ncoeffs)))
        # Constante de adaptación
        self.adapt_const = adapt_const
        # Vector de estados internos del filtro
        self.circular_buffer = np.zeros(self.buffer_size + 1, dtype=complex)

        # Indice de la muestra de entrada
        self.index_in = 0
        # Vector de índices de los coeficientes precalculados
        self.index_vector = np.arange(self.ncoeffs)

    def filtering_signal(self, signal_in, nsymbols=None, delay=0):
        """
        Función que filtra el vector muestras de  la señal de entrada
        y saca el número de muestras especificado, compensando
        el retardo que se especifique.

        Parameters
        ----------
            signal_in : ndarray (dtype : float/complex)
                vector de muestras de entrada al igualador
            nsymbols : int (optional, defaul:None)
                número de símbolos que se deben calcular
            delay : int (optional,deafualt:0)
                retardo que compensa el igualador
        Returns
        -------
            signal_out : ndarray (dtype : float/complex)
                vector de muestras de salida igualadas (tamaño Nsymbols)
        """
        # convolución
        signal_out = np.convolve(signal_in, self.coeffs)
        # Compensa el retardo y recorta a nsymbols
        if nsymbols == None:
            return signal_out[delay:]
        else:
            return signal_out[delay : delay + nsymbols]

    def filtering_sample(self, sample_in):
        """
        Función que filtra una muestras de entrada.

        Parameters
        ----------
            sample_in : float/complex
                muestra de la señal de entrada que se va
                a igualar
        Returns
        -------
            sample_out : float/complex
                símbolo igualado
        """
        # Incremento el índice temporal (módulo el tamaño del
        # vector de estados internos del filtro)
        self.index_in = (self.index_in + 1) % self.buffer_size
        # Actualización de estados internos
        # Se crea un buffer circular en la variable z
        self.circular_buffer[self.index_in] = sample_in
        # Cálculo de la salida
        sample_out = np.sum(
            self.circular_buffer[
                (self.index_in - self.index_vector) % self.buffer_size
            ]
            * self.coeffs
        )

        return sample_out

    def filtering_decimate(self, samples_vector_in):
        """
        Función que almacena las muestras de entrada pero sólo calcula la
        salida correspondiente a la última muestra de entrada.

        Se puede usar para una estructura eficiente de filtrado +
        diezmado.

        Parameters
        ----------
            samples_vector_in : array_like
                vector de muestras que se van a almacenar en los estados
                internos del filtro
        Returns
        -------
            sample_out : float/complex
                muestra de salida
        """
        samples_vector_in = np.atleast_1d(samples_vector_in)
        for k in np.arange(samples_vector_in.size):
            # Incremento el índice temporal (módulo el tamaño del
            # vector de estados internos del filtro)
            self.index_in = (self.index_in + 1) % self.buffer_size
            # Actualización de estados internos
            # Se crea un buffer circular en zi
            self.circular_buffer[self.index_in] = samples_vector_in[k]
        # Cálculo de la salida a partir de últimas muestras y coeficientes
        sample_out = np.sum(
            self.circular_buffer[
                (self.index_in - self.index_vector) % self.buffer_size
            ]
            * self.coeffs
        )
        return sample_out

    def adaptation(self, error):
        """
        Función que adapta los coeficientes del filtro con el error
        estimado

        Parameters
        ----------
            error : float/complex
                muestra del error
        """
        



def Wiener_filter(autocorr_vector, crosscorr_vector):
    """
    Función que calcula los coeficientes del filtro óptimo de Wiener
    en función de la autocorrelación de la señal observada y la correlación
    de la señal deseada y observada.

    Parameters
    ----------
        autocorr_vector : array_like
            vector de muestras de la autocorrelación de la señal observada de
            0 a N-1 (de tamaño N igual al número de coeficientes que se desea
            calcular)
        crosscorr_vector :  array_like
            vector de muestras de la correlación de la señal deseada con la
            observada (de tamaño igual al número de coeficientes que se desea
            calcular)

    Returns
    -------
        coeffs_vector : ndarray (dtype:float/complex)
            vector de coeficientes del filtro
    """
    autocorr_vector = np.asarray(autocorr_vector) #Sea lo que sea que me hayas pasado, ahora lo trato como un vector de numpy
    crosscorr_vector = np.asarray(crosscorr_vector)
    # Nosotros sabemos que viéndolo como matriz h=R^-1r, con R la de correlación y r autocorrelación
    # 1. autocorr_vector=[Rx[0],Rx[1],...,Rx[N-1]], queremos pasarlo a matriz
    autocorr_matrix = scia.toeplitz(autocorr_vector,np.conj(autocorr_vector)) #Si aurocorr_vector=[5,2], me da [5,2;2,5], toeplitz(primera columna, primera fila),
    # queremos si tenemos [Rx[0],Rx[1],Rx[2]], una matriz [Rx[0],Rx[-1],Rx[-2]; Rx[1],Rx[0],Rx[-1]; Rx[2],Rx[1],Rx[0]], la conjugada es porque la autocorrelación cumple
    # Rx[-k]=Rx*[k]
    coeffs_vector = scia.inv(autocorr_matrix) @ crosscorr_vector # =R^-1r, @ es la multiplicación matricial, no elemento a elemento
    return coeffs_vector
    



def predictor_filter(autocorr_vector):
    # Caso particular del filtro óptimo de Wiener
    """
    Función que calcula los coeficientes del filtro óptimo de Wiener
    que predice la siguiente muestra de la señal observada.

    Parameters
    ----------
        autocorr_vector : array_like
            vector de muestras de la autocorrelación de la señal observada de
            0 a N (de tamaño N+1, una muestra más que el número de coeficientes
            N que se desea calcular)

    Returns
    -------
        coeffs_vector : ndarray (dtype:float/complex)
            vector de coeficientes del filtro predictor óptimo
    """
    

