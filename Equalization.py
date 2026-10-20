##################################
# Equalization
##################################
"""
Módulo de igualación
"""

############################
# Librerías
############################
import numpy as np
import Filter as ft
import TransferFunctions as tf
import scipy.linalg as scia


###################################
# Funciones
###################################
def LE_design(
    discrete_response, noise_autocorr, symbols_energy, ncoeffs, delay, MMSE
):
    #Calculamos la autororrealción de lo que entra al filtro ZF que es a[n]*hD[n]
    # ruido por h[n], donde tenemos que al ser independientes, nos va a dar que
    #la autocorrelación de entrada es la suma de las dos autocorrelaciones
    # individualmente
    #Para calcula la correlación entre lo que entra y la señal deseada
    """
    Función que diseña los coeficientes óptimos con criterio
    MMSE del igualador LE-MMSE.

    Parameters
    ----------
        discrete_response : ndarray (dtype:float/complex)
            vector de muestras de la respuesta impulsiva discreta
            equivalente del canal
        noise_autocorr : ndarray (dtype:float/complex)
            vector de muestras de la autocorrelación del ruido
        symbols_energy : float
            energía de los símbolos transmitidos
        ncoeffs : int
            número de coeficientes del igualador
        delay : int
            retardo de la señal igualada
        MMSE : bool
            criterio de diseño MMSE (True) o ZF (False)

    Returns
    -------
        coeffs : ndarray (dtype:float/complex)
            vector de coefcientes del igualador
    """
    
    discrete_response = np.asarray(discrete_response)
    noise_autocorr = np.asarray(noise_autocorr)

    # Autocorrelación de la señal recibida sin ruido (Matriz R, todavía no completa)
    signal_autocorr = np.convolve(discrete_response,
        np.conj(discrete_response[::-1])
    ) 

    # Vector de autocorrelación de entrada(Matriz R): Para el filtro de Wiener
    input_autocorr = np.zeros(ncoeffs, dtype=complex)

    for k in range(ncoeffs):
        if k < discrete_response.size:
            input_autocorr[k] = (symbols_energy *
                signal_autocorr[discrete_response.size - 1 + k]
            ) #Autocorrelación de a[n]*hD[n]

    # Añadir ruido únicamente para MMSE
    if MMSE:
        n0 = len(noise_autocorr) // 2
        for k in range(ncoeffs):
            if n0 + k < noise_autocorr.size:
                input_autocorr[k] += noise_autocorr[n0 + k]
    #En este ya sí tenemos la utocreelación real, pues incluye la de a[n]*hd[n]
    # más la del ruido, lo cual el la autocorrelación de lo que entraría al 
    #filtro

    # Correlación cruzada entre señal deseada y recibida 
    # (r=[Rxrecibidaa[0],Rxrecibidaa[1],...]), pues el ZF desea eliminar 
    # la ISI, así que necesitamos que nos dé 1/hd[n], es decir,
    #xrecibida es una señal deformada por el canal y con ruido, pero yo sólo 
    #que el zero´forcing me devuelva la señal sin ISI, así que lo deseado es a[n]
    # y para ello no necesito contar el ruido
    #Señal deseada a[n-delay]
    #Entrada al ZF=a[n]*hD[n]
    crosscorr_vector = np.zeros(ncoeffs, dtype=complex)

    for k in range(ncoeffs):
        indice = delay - k
        if 0 <= indice < discrete_response.size:
            crosscorr_vector[k] = (
                symbols_energy *
                np.conj(discrete_response[indice])
            )

    # Diseño mediante el filtro de Wiener
    coeffs = ft.Wiener_filter(input_autocorr, crosscorr_vector)


    return coeffs


def DFE_design(
    discrete_response,
    noise_autocorr,
    symbols_energy,
    ncoeffs_ff,
    ncoeffs_fb,
    delay,
    MMSE,
):
    """
    Función que diseña los coeficientes óptimos con criterio
    MMSE/ZF del igualador DFE, tanto la etapa feedforward como
    la etapa feedback.

    Parameters
    ----------
        discrete_response : array_like
            vector de muestras de la respuesta discreta equivalente
            del canal
        noise_autocorr : ndarray (dtype:float/complex)
            vector de muestras de la autocorrelación del ruido
        symbols_energy : float
            energía de los símbolos transmitidos
        ncoeffs_ff : int
            número de coeficientes de la etapa feedforward o directa
        ncoeffs_fb : int
            número de coeficientes de la etapa feedback o realimentada
        delay : int
            retardo de la señal igualada
        MMSE : bool
            criterio de diseño MMSE (True) o ZF (False)

    Returns
    -------
        coeff_ff_vector : ndarray (dtype:float/complex)
            vector de coefcientes de la etapa feedforward o directa
        coeff_fb_vector : ndarray (dtype:float/complex)
            vector de coefcientes de la etapa feedback o realimentada
    """
    if MMSE:
        # Diseño por el método de Wienner
        # Término de la autocorrelación de la señal
        signal_autocorr = np.convolve(
            discrete_response, np.conj(discrete_response[::-1])
        )
        Nx = np.min([discrete_response.size, ncoeffs_ff])
        input_autocorr = np.zeros(ncoeffs_ff, dtype=discrete_response.dtype)
        input_autocorr[:Nx] = (
            symbols_energy
            * signal_autocorr[
                discrete_response.size - 1 : discrete_response.size - 1 + Nx
            ]
        )
        # Término de la autocorrelación del ruido
        no = np.argmax(np.abs(noise_autocorr))
        nsamples = np.min([noise_autocorr[no:].size, ncoeffs_ff])
        input_autocorr[:nsamples] = (
            input_autocorr[:nsamples]
            + MMSE * noise_autocorr[no : no + nsamples]
        )
        # Matriz de autocorrelación de la entrada de la etapa directa
        direct_matrix = scia.toeplitz(input_autocorr)
        # Matriz de correlación de la entrada del precursor y las
        # decisiones anteriores
        feedback_matrix = np.zeros(
            (ncoeffs_ff, ncoeffs_fb), dtype=discrete_response.dtype
        )
        # Bucle de columnas (simbolos anteriores)
        for c in np.arange(ncoeffs_fb):
            # Bucle de filas (muestras de entrada del precursor)
            for f in np.arange(ncoeffs_ff):
                if ((delay + 1 + c - f) < discrete_response.size) & (
                    (delay + 1 + c - f) >= 0
                ):
                    feedback_matrix[f, c] = symbols_energy * np.conj(
                        discrete_response[delay + 1 + c - f]
                    )
        # Matriz de autocorrelación global
        autocorr_matrix = np.block(
            [
                [direct_matrix, feedback_matrix],
                [
                    np.conj(feedback_matrix.T),
                    symbols_energy * np.eye(ncoeffs_fb),
                ],
            ]
        )
        # Vector de correlación cruzada
        cross_correlation = np.zeros(
            ncoeffs_ff + ncoeffs_fb, dtype=discrete_response.dtype
        )
        for k in np.arange(ncoeffs_ff):
            if ((delay - k) < discrete_response.size) & ((delay - k) >= 0):
                cross_correlation[k] = symbols_energy * np.conj(
                    discrete_response[delay - k]
                )
        # Término de regularización
        autocorr_matrix = autocorr_matrix + 1e-14 * np.mean(
            np.abs(autocorr_matrix) ** 2
        ) * np.eye(ncoeffs_ff + ncoeffs_fb)
        # Cálculo de coeficientes
        coeffs = np.squeeze(
            np.linalg.inv(autocorr_matrix) @ cross_correlation.reshape(-1, 1)
        )
        # Separación de los coefciencies
        coeffs_ff = coeffs[0:ncoeffs_ff]
        coeffs_fb = -coeffs[-ncoeffs_fb:]
    else:
        # Array_like a ndarray
        discrete_response = np.asarray(discrete_response)
        noise_autocorr = np.asarray(noise_autocorr)
        # Diseño de la parte de igualación
        coeffs_Eq_vector = LE_design(
            discrete_response,
            noise_autocorr,
            symbols_energy,
            2 * ncoeffs_ff,
            delay,
            MMSE,
        )
        ## Diseño del filtro blanqueador
        # Cálculo de la autocorrelación del ruido tras el filtro ZF
        new_noise_autocorr = np.convolve(
            noise_autocorr,
            np.convolve(coeffs_Eq_vector, np.conj(coeffs_Eq_vector[::-1])),
        )
        # Término de la autocorrelación del ruido
        ini_index = np.argmax(np.abs(new_noise_autocorr))
        input_autocorr = np.zeros(
            ncoeffs_fb + 1, dtype=discrete_response.dtype
        )
        last = np.min([ini_index, ncoeffs_fb + 1])
        input_autocorr = new_noise_autocorr[ini_index : ini_index + last]
        coeffs_w = np.block([1, -ft.predictor_filter(input_autocorr)])
        ## Diseño de la etapa directa
        coeffs_ff = np.convolve(coeffs_Eq_vector, coeffs_w)
        # Recorte al número de muestras especificado
        coeffs_ff = coeffs_ff[:ncoeffs_fb]
        # Respuesta impulsiva equivalente
        h_eq1 = np.convolve(discrete_response, coeffs_Eq_vector)
        h_eq2 = np.convolve(discrete_response, coeffs_ff)
        ## Diseño de la etapa realimentada
        ini_index = np.argmax(np.abs(h_eq1)) + 1
        coeffs_fb = np.zeros(ncoeffs_fb, dtype=h_eq2.dtype)
        last = np.min([ini_index + ncoeffs_fb, h_eq2.size])
        coeffs_fb[: last - ini_index] = h_eq2[ini_index:last]

    return coeffs_ff, coeffs_fb


def ideal_DFE_design(discrete_response, noise_autocorr, energy_const):
    """
    Función que calcula el módulo al cuadrado de la respuesta en frecuencia
    de la etapa directa del igualador DFE con diseño MMSE ideal (sin 
    restricción en el número de coeficientes)

    Parameters
    ----------
        discrete_response : array_like
            vector de muestras de la respuesta discreta equivalente
            del canal
        noise_autocorr : ndarray (dtype:float/complex)
            vector de muestras de la autocorrelación del ruido
        energy_const : float
            energía media de la constelación transmitida

    Returns
    -------
        freq_axis :  ndarray (dtype:float)
            vector del eje de frecuencias de la respuesta en frecuencia
        freq_response :  ndarray (dtype:float)
            vector de muestras del módulo al cuadrado de la respuesta en
            frecuencia de la etapa directa del DFE MMSE ideal

    """
    # Autocorelación de la señal (sinruido)
    signal_autocorr = energy_const * np.convolve(
        discrete_response, np.conj(discrete_response[::-1])
    )
    # Espectro de la señal (sin ruido)
    freq_axis, signal_spectrum = tf.frequency_response(signal_autocorr,[1])
    # Espectro del ruido
    freq_axis, noise_spectrum = tf.frequency_response(noise_autocorr,[1])
    # Constantes de normalización de la factorización de la señal y el ruido
    Ks = np.exp(
        np.mean(np.log(np.abs(noise_spectrum) + np.abs(signal_spectrum)))
    )
    Kn = np.exp(np.mean(np.log(np.abs(noise_spectrum))))
    # Módulo de la respuesta en frecuencia
    freq_response = (
        energy_const
        * Kn
        / Ks
        * np.abs(signal_spectrum)
        / (np.abs(noise_spectrum) + np.abs(signal_spectrum))
        / np.abs(noise_spectrum)
    )

    return (
        freq_axis,
        freq_response,
    )


#################
# Clases
#################
class LE:
    """
    Clase que implementa un igualador lineal.
    """

    def __init__(self, QAMconst, coeffs_vector, adapt_const=None):
        """
        Constructor de la clase AdaptiveEq

        Parameters
        ----------
            QAMconst : object QAM
                objeto de la clase QAM
            coeffs : ndarray (dtype float/complex)
                vector de coeficientes del igualador
            adapt_const : float (optional - default None)
                constante de adaptación
        """
        self.QAMconst = QAMconst #Valores que puede tomar a[n] para 
        # comparar en el decisor
        self.Eq = ft.Filter(coeffs_vector, adapt_const) #Creamos el igualador lineal


    def equalization_decision(self, rx_samples, nsymbols, delay):
        """
        Método que iguala y decide un vector de muestras de la señal
        recibida.

        Parameters
        ----------
            rx_samples : array_like
                vector de muestras de entrada
            nsymbols : int
                número de símbolos que hay que igualar y decidir
            delay : int
                retardo que tiene que compensar en la igualación

        Returns
        -------
            eq_symbols : ndarray (dtype:float/complex)
                vector de símbolos igualados (tamaño = nsymbols)
            decided_symbols : ndarray (dtype:float/complex)
                vector de símbolos decididos (tamaño = nsymbols)

        """
        #Convolución de he[n] con los símbolos recibidos del canal(con ruido e
        #ISI), retardamos porque el filtro LE estará retardado
        #rx_samples=a[n]*hD[n]+w[n]*h[n]
        #Ahora haremos rx_samples*hLE[n], con hLE[n] cumpliendohD[n]*hLE[n]=delta[n-delay]
        #eq_symbols[n]=z[n+delay]
        eq_symbols = self.Eq.filtering_signal(rx_samples, nsymbols, delay)
        #Decidir los símbolos a partir de las muestras ya filtradas
        decided_symbols = self.QAMconst.decision_symbol_seq(eq_symbols)
        return eq_symbols, decided_symbols

    def adaptive_equalization_decision(
        self, rx_samples, nsymbols, delay, training_symbols
    ):
        """
        Método que iguala y decide un vector de muestras de la señal
        recibida, adaptando los coeficientes con el error calculado
        con los símbolos de entrenamiento y/o las decisiones de los
        símbolos igualados (cuando se acaban los de entrenamiento).

        Parameters
        ----------
            rx_samples : array_like
                vector de muestras de entrada
            nsymbols : int
                número de símbolos que hay que igualar y decidir
            delay : int
                retardo que tiene que compensar en la igualación
            training_symbols : array_like
                vector de símbolos de entrenamiento

        Returns
        -------
            eq_symbols : ndarray (dtype:float/complex)
                vector de símbolos igualados (tamaño = nsymbols)
            decided_symbols : ndarray (dtype:float/complex)
                vector de símbolos decididos
                (tamaño = nsymbols - training_symbols.size)
        """
        eq_symbols = np.zeros(nsymbols, dtype=complex) #Guarda  los símbolos a
        #la salida del igualador que esr¡tamos entrenando
        decided_symbols = np.zeros(nsymbols - len(training_symbols), dtype=complex) # Guarda los 
        #símbolos elegidos por el decisor
        # Además, la primera vez que hagamos la fase de entrenamiento, hay 
        # que dividir por 10 la mu, para adaptar mejor
        #self.Eq.adapt_const = self.Eq.adapt_const / 10
        for n in range(nsymbols + delay): #Porque si la salida tiene un 
        # retardo de n muestras, estas también hay que leerlas
            y = self.Eq.filtering_sample(rx_samples[n])
            if n >= delay: #Sólo guardamos muestras a partir de delay, el 
            #resto son 0
                k = n - delay
                eq_symbols[k] = y
                #Si todavía estamos usando los símbolos de entrenamiento
                if k < len(training_symbols):
                    
                    error = y - training_symbols[k]
                #Si ya se ha acabado la fase de entrenamiento
                else:
                    if k == len(training_symbols):
                        self.Eq.adapt_const /= 10
                    decided_symbols[k -  len(training_symbols)] = (
                        self.QAMconst.decision_symbol(y)
                    )
                    error = y - decided_symbols[k -  len(training_symbols)]
                self.Eq.adaptation(error)
        return eq_symbols,  decided_symbols


    def response(self):
        """
        Método que devuelve los coeficientes del igualador
        Returns
        -------
            coeffs : ndarray (dtype:float/complex)
                vector de coeficientes del igualador
        """
        return self.Eq.coeffs


class DFE:
    """
    Clase que implementa un igualador DFE.
    """

    def __init__(
        self,
        QAMconst,
        coeffs_ff,
        coeffs_fb,
        adapt_const_ff=None,
        adapt_const_fb=None,
    ):
        """
        Constructor de la clase DFE

        Parameters
        ----------
            QAMconst : QAM object
                objeto de la clase QAM
            coeffs_ff : array_like
                vector de coeficientes del igualador precursor
            coeffs_fb : array_like
                vector de coeficientes del igualador de realimentación
            adapt_const_ff : float (optional - default None)
                constante de adaptación del filtro directo
            adapt_const_ff : float (optional - default None)
                    constante de adaptación del filtro realimentado
        """

        self.QAMconst = QAMconst
        self.Eq_FF = ft.Filter(coeffs_ff, adapt_const=adapt_const_ff)
        self.Eq_FB = ft.Filter(coeffs_fb, adapt_const=adapt_const_fb)

    def equalization_decision(self, rx_samples, nsymbols, delay):
        """
        Función que implementa la igualación y la decisión.

        Parameters
        ----------
            rx_samples : array_like
                vector de muestras de la señal que se va a igualar
            nsymbols : int
                número de símbolos que hay que obtener a la salida
            delay : int
                retardo (en muestras) que hay que compensar
        Returns
        -------
            eq_symbols : ndarray (dtype:float/complex)
                vector de símbolos igualados
            decided_symbols :  ndarray (dtype:float/complex)
                vector de símbolos decididos
        """
        # Array_like
        rx_samples = np.asarray(rx_samples)
        # Inicia los vectores decididos e igualados
        filtered_samples = np.zeros(nsymbols, dtype=rx_samples.dtype)
        eq_symbols = np.zeros(nsymbols, dtype=rx_samples.dtype)
        decided_symbols = np.zeros(nsymbols, dtype=rx_samples.dtype)
        # Inicia la variable de realimentación
        ISI = 0
        # Rellena de ceros la entrada si faltan muestras
        if rx_samples.size < (nsymbols + delay):
            extend_rx_samples = np.zeros(
                nsymbols + delay, dtype=rx_samples.dtype
            )
            extend_rx_samples[: rx_samples.size] = rx_samples
        else:
            extend_rx_samples = rx_samples
        # Compensa el retardo (la salida no se usa)
        filtered_samples[0] = self.Eq_FF.filtering_decimate(
            extend_rx_samples[:delay]
        )
        # Bucle de filtrado y decisión
        for k in np.arange(nsymbols):
            filtered_samples[k] = self.Eq_FF.filtering_sample(
                extend_rx_samples[k + delay]
            )
            eq_symbols[k] = filtered_samples[k] - ISI
            decided_symbols[k] = self.QAMconst.decision_symbol(eq_symbols[k])
            ISI = self.Eq_FB.filtering_sample(decided_symbols[k])
        return eq_symbols, decided_symbols

    def adaptive_equalization_decision(
        self, rx_samples, nsymbols, delay, training_symbols
    ):
        """
        Método que iguala y decide un vector de muestras de la señal
        recibida, adaptando los coeficientes con el error calculado
        con los símbolos de entrenamiento y/o las decisiones de los
        símbolos igualados (cuando se acaban los de entrenamiento)

        Parameters
        ----------
            rx_samples : ndarray (dtype:float/complex)
                vector de muestras de la señal que se va a igualar
            nsymbols : int
                número de símbolos que hay que obtener a la salida
            delay : int
                retardo (en muestras) que hay que compensar
            training_symbols : ndarray (dtype:float/complex)
                vector de símbolos de entrenamiento
        Returns
        -------
            eq_symbols : ndarray (dtype:float/complex)
                vector de muestras de los símbolos igualados
                (tamaño Nsymbols)
            decided_symbols :  ndarray (dtype:float/complex)
                vector de muestras de los símbolos decididos
                (tamaño Nsymbols-a.size)
        """
        rx_samples = np.asarray(rx_samples)
        # Inicia los vectores de símbolos igualados y decididos
        filtered_samples = np.zeros(nsymbols, dtype=complex)
        eq_symbols = np.zeros(nsymbols, dtype=complex)
        decided_symbols = np.zeros(
            nsymbols - training_symbols.size, dtype=complex
        )
        # Inicia la variable de realimentación
        ISI = 0
        # Rellena de ceros la señal de entrada si no hay muestras suficientes
        if rx_samples.size < (nsymbols + delay):
            extend_rx_samples = np.zeros(
                nsymbols + delay, dtype=rx_samples.dtype
            )
            extend_rx_samples[: rx_samples.size] = rx_samples
        else:
            extend_rx_samples = rx_samples
        # Compensa el retardo
        filtered_samples[0] = self.Eq_FF.filtering_decimate(
            extend_rx_samples[:delay]
        )
        # Bucle de filtrado, decisisón y adaptación
        for k in np.arange(nsymbols):
            filtered_samples[k] = self.Eq_FF.filtering_sample(
                extend_rx_samples[k + delay]
            )
            eq_symbols[k] = filtered_samples[k] - ISI
            if k < training_symbols.size:
                self.Eq_FF.adaptation(eq_symbols[k] - training_symbols[k])
                self.Eq_FB.adaptation(-eq_symbols[k] + training_symbols[k])
                # Calcula ISI para la próxima iteración
                ISI = self.Eq_FB.filtering_sample(training_symbols[k])
            else:
                if k == training_symbols.size:
                    self.Eq_FF.adapt_const = self.Eq_FF.adapt_const / 10
                    self.Eq_FB.adapt_const = self.Eq_FB.adapt_const / 10
                decided_symbols[k - training_symbols.size] = (
                    self.QAMconst.decision_symbol(eq_symbols[k])
                )
                # Adaptación
                self.Eq_FF.adaptation(
                    eq_symbols[k] - decided_symbols[k - training_symbols.size]
                )
                self.Eq_FB.adaptation(
                    -eq_symbols[k] + decided_symbols[k - training_symbols.size]
                )
                # Calcula ISI para la próxima iteración
                ISI = self.Eq_FB.filtering_sample(
                    decided_symbols[k - training_symbols.size]
                )

        return eq_symbols, decided_symbols

    def response(self):
        """
        Método que devuelve los coeficientes del igualador.

        Returns
        -------
            hp,hf : ndarray (dtype:float/complex)
                vectores de coeficientes del igualador
        """
        return self.Eq_FF.coeffs, self.Eq_FB.coeffs


class FSE:
    """Clase que implementa un igualador FSE"""

    def __init__(
        self,
        QAMconst,
        coeffs_fse,
        coeffs_fb,
        oversampling,
        adapt_constant_fse=None,
        adapt_constant_fb=None,
    ):
        """
        Constructor de la clase FSE.

        Parameters
        ----------
        QAMconst : QAM object
            objeto de la clase QAM
        coeffs_fse : ndarray (dtype:float/complex)
            vector de coeficientes del igualador FSE
        coeffs_fb : ndarray (dtype:float/complex)
            vector de coeficientes del igualador realimentado
        oversampling : int
            factor de sobremuestreo de la señal de entrada al FSE
        adapt_constant_fse : float
            constante de adaptación del igualador fraccionario
        adapt_constant_fb : float
            constante de adaptación del igualador realimentado

        """
        # Constelación QAM
        self.QAMconst = QAMconst
        # Igualador FSE
        self.Eq_FSE = ft.Filter(coeffs_fse, adapt_constant=adapt_constant_fse)
        # Igualador realimentado
        self.Eq_FB = ft.Filter(coeffs_fb, adapt_constant=adapt_constant_fb)
        # Factor de sobremuestreo
        self.oversampling = oversampling

    def equalization_decision(self, rx_samples, nsymbols, delay):
        """
        Método que implementa la igualación FSE+DFE.

        Parameters
        ----------
            rx_samples : array_like
                vector de muestras de la señal recibida
            nsymbols : int
                número de símbolos que hay que igualar y decidir
            delay : int
                retardo que hay que comensar
        Returns
        -------
            eq_symbols : ndarray (dtype:float/complex)
                vector de símbolos igualados
            decided_symbols :  ndarray (dtype:float/complex)
                vector de símbolos decididos
        """

        # Inicia los vecrtores de símbolos igualados y decididos
        filtered_samples = np.zeros(nsymbols, dtype=complex)
        eq_symbols = np.zeros(nsymbols, dtype=complex)
        decided_symbols = np.zeros(nsymbols, dtype=complex)
        # Inicio de la variable de realimentación
        ISI = 0
        # Chequea si faltan muestras al final y rellena de ceros
        if rx_samples.size < (self.oversampling * nsymbols + delay):
            extend_rx_samples = np.zeros(
                self.oversampling * nsymbols + delay, dtype=rx_samples.dtype
            )
            extend_rx_samples[: rx_samples.size] = rx_samples
        else:
            extend_rx_samples = rx_samples
        # Compensa el retardo
        filtered_samples[0] = self.Eq_FSE.Filtered_Decimate(
            extend_rx_samples[: delay - self.oversampling + 1]
        )
        # bucle de igualación y decisión
        for k in np.arange(nsymbols):
            filtered_samples = self.Eq_FSE.Filtered_Decimate(
                extend_rx_samples[
                    self.oversampling * (k - 1)
                    + delay
                    + 1 : self.oversampling * k
                    + delay
                    + 1
                ]
            )
            eq_symbols[k] = filtered_samples[k] - ISI
            decided_symbols[k] = self.QAMconst.Decision_Symbol(eq_symbols[k])
            ISI = self.Eq_FB.filtering_sample(decided_symbols[k])
        return eq_symbols, decided_symbols

    def adaptive_equalization_decision(
        self, rx_samples, nsymbols, delay, training_symbols
    ):
        """
        Método que iguala y decide un vector de muestras de la señal
        recibida, adaptando los coeficientes con el error calculado
        con los símbolos de entrenamiento o las decisiones de los
        símbolos igualados (cuando se acaban los de entrenamiento).

        Parameters
        ----------
            rx_samples : ndarray (dtype:float/complex)
                vector de muestras de la señal recibida
            nsymbols : int
                número de símbolos que hay que igualar y decidir
            delay : int
                retardo que hay que comensar
            training_symbols : ndarray (dtype:float/complex)
                vector de símbolos de entrenamiento
        Returns
        -------
            eq_symbols : ndarray (dtype:float/complex)
                vector de muestras de los símbolos igualados
                (tamaño Nsymbols)
            decided_symbols :  ndarray (dtype:float/complex)
                vector de muestras de los símbolos decididos
                (tamaño Nsymbols-a.size)

        """
        # Inicia los vectores de símbolos igualados y decididos
        filtered_samples = np.zeros(nsymbols, dtype=complex)
        eq_symbols = np.zeros(nsymbols, dtype=complex)
        decided_symbols = np.zeros(
            nsymbols - training_symbols.size, dtype=complex
        )
        # Inicio de la variable de realimentación
        ISI = 0
        # Chequea que haya muestras suficientes y rellena de ceros
        if rx_samples.size < (self.oversampling * nsymbols + delay):
            extend_rx_samples = np.zeros(
                self.oversampling * nsymbols + delay, dtype=rx_samples.dtype
            )
            extend_rx_samples[: rx_samples.size] = rx_samples
        else:
            extend_rx_samples = rx_samples
        # Compensa el retardo (no se usa la salida)
        filtered_samples[0] = self.Eq_FSE.FilteringDecimate(
            extend_rx_samples[:delay]
        )
        # bucle de igualación, decisión y adaptación
        for k in np.arange(nsymbols):
            filtered_samples[k] = self.Eq_FSE.Filtered_Decimate(
                extend_rx_samples[
                    self.oversampling * (k - 1)
                    + delay
                    + 1 : self.oversampling * k
                    + delay
                    + 1
                ]
            )
            eq_symbols[k] = filtered_samples[k] - ISI
            # Adaptación con secuencia de entrenamiento
            if k < training_symbols.size:
                self.Eq_FSE.Adapting(eq_symbols[k] - training_symbols[k])
                self.Eq_FB.Adapting(-eq_symbols[k] + training_symbols[k])
                # Calcula ISI para la próxima iteración
                ISI = self.Eq_FB.filtering_sample(training_symbols[k])
            # Adaptación con símbolos decididos
            else:
                decided_symbols[k - training_symbols.size] = (
                    self.QAMconst.DecisorSymbol(filtered_samples[k])
                )
                self.Eq_FSE.Adaptation(
                    eq_symbols[k] - decided_symbols[k - training_symbols.size]
                )
                self.Eq_FB.Adaptation(
                    -eq_symbols[k] + decided_symbols[k - training_symbols.size]
                )
                # Calcula ISI para la próxima iteración
                ISI = self.Eq_FB.filtering_sample(
                    decided_symbols[k - training_symbols.size]
                )

        return eq_symbols, decided_symbols

    def response(self):
        """
        Método que devuelve los coefciientes del igualador.

        Returns
        -------
            coeffs_fse,coeffs_fb : ndarray (dtype:float/complex)
                vectores de coeficientes del igualador
        """
        return self.Eq_FSE.coeffs, self.Eq_FB.coeffs


def roots(poly):
    '''
    Función que calcula las raíces del polinomio.

    Parameters
    ----------
        poly : array_like 
            vector de coeficientes del polinomio

    Returns
    -------
        zeros : ndarray (dtype:float/complex)
            vector de raíces del polinomio
    '''
    # Array-like a ndarray
    poly=np.asarray(poly)
    # Reduce el rángo dinámico
    poly[np.abs(poly)<(1e-6*np.max(np.abs(poly)))]=0
    # Calcula las raíces
    zeros=np.roots(poly)
    
    return zeros
        
        