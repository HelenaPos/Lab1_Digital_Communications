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

    def __init__(self, coeffs, adapt_const=None): #"None" implies that 
    # "adapt_const" is optional
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
        coeffs = np.asarray(coeffs) # We recieve the coefficients: 
        # h[n]=[h[0],...,h[N-1]]
        # Vector de coeficientes
        self.coeffs = coeffs
        # Número de coeficientes
        self.ncoeffs = coeffs.size
        # Potencia de 2 mayor y más cercana a ncoeffs
        self.buffer_size = int(2 ** np.ceil(np.log2(self.ncoeffs))) #If we have
        # 1 coefficient, we have a buffer size of 2^0, if we have 2, 2^1, 
        # if we have 3, 2^2, if we have 4, 2^2,.... We can do by doing 
        # log2(number of coefficients), we round this with ceil and 
        # then 2^rounded. We make int(float), beacuse ceil returns a float
        # Constante de adaptación
        self.adapt_const = adapt_const
        # Vector de estados internos del filtro
        self.circular_buffer = np.zeros(self.buffer_size + 1, dtype=complex) 
        # It will contain x[0],...,x[N-1], but rigth now, we do not have them,
        # so we will have it with 0s. dtype=complex indicates that we want to
        # store complex number
        # Indice de la muestra de entrada
        self.index_in = 0 # Where the most recent sample is
        # Vector de índices de los coeficientes precalculados
        self.index_vector = np.arange(self.ncoeffs) # It returns an array 
        # with this numbers [0,1,2,...N-1]

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
        self.index_in = (self.index_in + 1) % self.buffer_size # It indicates
        # where we are going to store this sample. We need it becase 
        # of how we are storing the samples, to know where to start
        # reading
        # Actualización de estados internos
        # Se crea un buffer circular en la variable z
        self.circular_buffer[self.index_in] = sample_in
        # Example: N=4, x[0] in circular_buffer[1], x[1] in circular_buffer[2]
        # x[2] in circular_buffer[3], x[3] in circular_buffer[0], so will have:
        # [x[3], x[0], x[1], x[2]], because of this we need index_in
        # Cálculo de la salida
        sample_out = np.sum(
            self.circular_buffer[
                (self.index_in - self.index_vector) % self.buffer_size
            ]
            * self.coeffs
        )
        # Example: N=3
        # self.index_vector =[0, 1, 2]
        # self.coeffs = [h[0], h[1], h[2]]
        # self.index_in = 1 (at the beginning)
        # self.index_in - self.index_vector = 1 -[0,1,2]=[1,0,-1]
        # (self.index_in - self.index_vector) % self.buffer_size = [1,0,-1]%4
        # = [1,0,3], which tell us where are x[n], x[n-1], x[n-2], so we will
        # get them with self.circular_buffer[1,0,3]
        #Then, we make np.sum(...*self.coeffs)=[x[n]h[n]+x[1]h[n-1]+x[2]h[n-2]]

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
        samples = self.circular_buffer[
            (self.index_in - self.index_vector) % self.buffer_size
        ]
        self.coeffs = (self.coeffs - self.adapt_const*error*np.conj(samples))



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
    # We recive parameters "array_like", so maybe will get a list instead of 
    # an array, so we may not be able to use them with NumPy operations. 
    # The solution is to make sure we are using arrays.
    autocorr_vector = np.asarray(autocorr_vector) 
    crosscorr_vector = np.asarray(crosscorr_vector)
    # We know that we can resolve it by applying h=R^-1r,
    # with R being the autocorrelation matrix and r "crosscorr_vector"
    # We have autocorr_vector=[Rx[0],Rx[1],...,Rx[N-1]], but we need it
    # as the autocorrelation matrix. To do that, we can use the following 
    # propertie: Rx[-m]=Rx*[m]
    autocorr_matrix = scia.toeplitz(autocorr_vector,np.conj(autocorr_vector)) 
    coeffs_vector = scia.inv(autocorr_matrix) @ crosscorr_vector # We are 
    # doing R^-1r. @ is for matrix multiplication, so we will not make
    # element x element
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
    # We know the autocorrelation of x[n], but to use "Wiener_filter", 
    # we must use the correlation between x[n] and the desired signal, in this 
    # problem: d[n]=x[n+1].
    # The correlation is rdx[m]=E[d[n]x*[n-m]]=E[x[n+1]x*[n-m]]=Rx[m+1]
    autocorr_vector = np.asarray(autocorr_vector) 
    # We have received samples from 0 to N (we will need the last one because 
    # of the Rx[m+1]), but to use the Wiener_filter, we only need from 0 to N-1
    # autocorr_vector[:-1]
    # For the correlation, we need from sample number 1 to N: 
    # autocorr_vector[1:]
    coeffs_vector = Wiener_filter(autocorr_vector[:-1], autocorr_vector[1:])
    return coeffs_vector
