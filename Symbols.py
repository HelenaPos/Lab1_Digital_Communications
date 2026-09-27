##################################
# Symbols
##################################
"""
Módulo para la codificación/decodificación de modulaciones QAM cuadradas.
"""

##################################
# Librerías
##################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scie
import Binary as bn

############################
# Variables globales
############################
N_SYMBOLS_PLOT = 5000

############################
# Clases y funciones
############################


class QAM:
    """
    Clase que implementa las funciones básicas de un codificador y
    decodificador de una constelación QAM cuadrada consgtruida como la
    combinación de dos PAM, una para la parte real y otra para la parte
    imaginaria del símbolo.

    Esta clase incluye las siguientes funcionalidades: la decisión de los
    símbolos recibidos (hard y soft), mapeo/demapeo de bits, representación
    gráfica de símbolos recibidos y el cálculo de SER para canales AWGN.
    """

    def __init__(self, nsymbol_const, energy, Gray):
        """
        Método constructor de la clase QAM.

        Parameters
        ----------
            nsymbol_const : int
                número de símbolos de la constelación QAM cuadrada (debe
                ser potencia de 4)
            energy : float
                energía media de los símbolos de la constelación
            Gray : bool
                indicador de si se usa codificación Gray o no.
        Raises
        ------
            ValueError:
                Tamaño de la constelación incorrecto
        """
        if 2 ** (int(np.log2(nsymbol_const) / 2) * 2) != nsymbol_const:
            ValueError(" Tamaño de la constelación incorrecto ")
        # Número de símbolos
        self.nsymbols_const = nsymbol_const
        # Número de bits por símbolo y por parte real del símbolo
        self.nbits_symbol = int(np.log2(nsymbol_const))
        self.nbits_symbol_real = int(self.nbits_symbol / 2)
        # Número de símbolos de la parte real o imaginaria
        self.nsymbols_real = int(np.sqrt(nsymbol_const))
        # Energía media de los símbolos de la constelación
        self.energy = energy
        # Distancia mínima entre símbolos QAM
        self.dist = np.sqrt(6 * energy / (nsymbol_const - 1))
        # Valor máximos de la constelación (parte real o completo)
        self.maxsymbol_real = ((self.nsymbols_real - 1) / 2) * self.dist
        self.maxsymbol = self.maxsymbol_real * (1 + 1j)
        # Offset para convertir los símbolos normalizados y redondeados a
        # índices
        self.symbol2index_offset = int(self.nsymbols_real / 2)
        # Número máximo de símbolos QAM qyue se van a representar
        self.nsymbols_plot = N_SYMBOLS_PLOT
        self.bits_mapping, self.bits_demapping = QAM.labels_PAM(
            self.nbits_symbol_real, Gray
        )
        self.table_index_bit = QAM.table_soft_decision(self.bits_demapping)

    @staticmethod
    def labels_PAM(nbits_symbol, Gray):
        """
        Método que calcula las tabla de mapeo y demapeo de bits de una
        constelación PAM (pudiendo usar codificación Gray).

        Parameters
        ----------
            nbits_symbol : int
                número de bits por símbolo
            Gray : bool
                indicador de si se usa codificación Gray (True) o no (False).

        Returns
        -------
            bits_mapping : ndarray (dtype:np.byte)
                tabla que contiene un vector de bits para cada índice de símbolo
                de la constelación PAM
    
            bits_demapping : ndarray (dtype:np.byte)
                tabla que contiene un vector de bits para cada índice de símbolo
                de la constelación PAM
        """

        # Tablas de demapeo
        if Gray:
            # Método recursivo de generación de la tabla de demapping Gray
            # (comienza con el demapping para un único bit)
            bits_demapping = np.array([[0], [1]], dtype=np.byte)
            for k in np.arange(1, nbits_symbol):
                bits_demapping = np.block(
                    [
                        [
                            bits_demapping,
                            np.zeros((2**k, 1), dtype=np.byte),
                        ],
                        [
                            bits_demapping[::-1, :],
                            np.ones((2**k, 1), dtype=np.byte),
                        ],
                    ]
                )
        else:
            # Tabla de demapeo trivial
            bits_demapping = bn.dec2bin(
                np.arange(int(2**nbits_symbol)),
                nbits_symbol,
            )
        # Tabla de mapeo a partir de la de demapeo
        bits_mapping = np.zeros(2**nbits_symbol, dtype=np.byte)
        bits_mapping[bn.bin2dec(bits_demapping)] = np.arange(2**nbits_symbol)

        return bits_mapping, bits_demapping

    @staticmethod
    def table_soft_decision(bits_demapping):
        """
        Método para el cálculo de las tablas que identifican los símbolos que
        comparten cada uno de los bits a 0 o 1 usado para la decisisón soft.

        Parameters
        ----------
            bits_demapping:  ndarray (dtype:np.byte)
                tabla que asocia a cada índice de símbolo un vector fila de 
                bits

        Returns
        -------
            table_index_bits : ndarray (dtype:np.byte)
                tabla de índices de símbolos que comparten el bit n-ésimo a 
                0 ó 1 (valor_bit,índice del bit,indices de símbolo)

        """

        _, nbits_symbol = np.shape(bits_demapping)
        # Tablas de indices que comparten bit en cada posición
        table_index_bits = np.zeros(
            (
                2,
                nbits_symbol,
                2 ** (nbits_symbol - 1),
            ),
            dtype=np.byte,
        )
        for bit in np.arange(nbits_symbol):
            table_index_bits[0, bit, :] = np.where(
                bits_demapping[:, bit] == 0
            )[0]
            table_index_bits[1, bit, :] = np.where(
                bits_demapping[:, bit] == 1
            )[0]

        return table_index_bits

    def mapping(self, bits):
        """
        Método que mapea los bits en símbolos de la constelación.

        Parameters
        ----------
            bits :  array_like
                vector de bits que se van a codificar como símbolos

        Returns
        -------
            symbols : ndarray (dtype:complex)
                vector de símbolos mapeados
        Raises
        ------
            ValueError:
                Número de bits a mapear no es múltiplo del numero de bits
                por símbolo de la constelación
        """
        # Array_like a ndarray
        bits = np.asarray(bits, dtype=np.byte)
        # Avisa si el tamaño es erróneo
        if (bits.size % self.nbits_symbol) != 0:
            raise ValueError(
                "Número de bits a mapear no es múltiplo del "
                + "numero de bits por símbolo de la constelación"
            )
        # Matriz de bits
        bitsMatrix = bits.reshape(-1, self.nbits_symbol)
        # Cálcula los índices de la parte real e imaginaria
        index_real = self.bits_mapping[
            bn.bin2dec(bitsMatrix[:, : self.nbits_symbol_real])
        ]
        index_imag = self.bits_mapping[
            bn.bin2dec(bitsMatrix[:, self.nbits_symbol_real :])
        ]
        # cCodifica los símbolos
        symbols = (index_real + 1j * index_imag) * self.dist - self.maxsymbol
        return symbols

    def demapping(self, decided_symbols):
        """
        Método que demapea los símbolos decididos como bits.

        Parameters
        ----------
            decided_symbols : array_like
                vector de símbolos decididos

        Returns
        -------
            decided_bits : ndarray (dtype:np.byte)
                vector de bits decididos

        """
        # Array_like a ndarray
        decided_symbols = np.asarray(decided_symbols, dtype="complex")
        # Cálcula los índices decididos
        index = self.decision_index(decided_symbols)
        # Calcula la matriz de bits, una fila por cada símbolo (mitad de la
        # parte real y mitad de la parte imaginaria)
        bitsMatrix = np.zeros(
            (int(index.size / 2), self.nbits_symbol), dtype=np.byte
        )
        bitsMatrix[:, : self.nbits_symbol_real] = self.bits_demapping[
            index[::2]
        ]
        bitsMatrix[:, self.nbits_symbol_real :] = self.bits_demapping[
            index[1::2]
        ]

        return bitsMatrix.ravel()

    def decision_index(self, rx_symbols):
        """
        Método que decide los índices de las partes real e imaginaria de los
        símbolos recibidos más probable.

        Parameters
        ----------
            rx_symbols : array_like
                vector de símbolos recibidos

        Returns
        -------
            index : ndarray (dtype:np.byte)
                vector de índices de los símbolos, los pares corresponden a la
                parte real del símbolo, los impares a la parte imaginaria.
        """
        # Array_like a ndarray 1d
        rx_symbols = np.asarray(rx_symbols, dtype="complex")
        rx_symbols = np.atleast_1d(rx_symbols)
        # Inicio del vector de índices
        index = np.zeros(2 * rx_symbols.size, dtype=int)
        # Índices de símbolo de las partes reales e imaginarias
        index[::2] = (
            np.floor(rx_symbols.real / self.dist).astype(int)
            + self.symbol2index_offset
        )
        index[1::2] = (
            np.floor(rx_symbols.imag / self.dist).astype(int)
            + self.symbol2index_offset
        )
        # Cambia los que se salen de los límites inferior o superior
        index[index >= self.nsymbols_real] = self.nsymbols_real - 1
        index[index < 0] = 0

        return index

    def decision_symbol_seq(self, rx_symbols):
        """
        Método que decide la secuencia de símbolos transmitidos más probable
        a partir de una secuencia de símbolos recibidos

        Parameters
        ----------
            rx_symbols : array_like
                vector de símbolos recibidos

        Returns
        -------
            decided_symbols : ndarray (dtype : complex)
                vector de símbolos decididos
        """

        # Array_like a ndarray 1d
        rx_symbols = np.asarray(rx_symbols)
        rx_symbols = np.atleast_1d(rx_symbols)
        # Cálculo de los índices de los símbolos decididos
        index = self.decision_index(rx_symbols)
        # Cálculo de los símbolos decididos a partir de los índices
        decided_symbols = (
            index[::2] + 1j * index[1::2]
        ) * self.dist - self.maxsymbol

        return decided_symbols

    def decision_symbol(self, rx_symbol):
        """
        Función que decide el vector de símbolos transmitidos más probable
        a partir de un vector de símbolos recibidos

        Parameters
        ----------
            rx_symbol : complex
                símbolos recibido

        Returns
        -------
            decided_symbol : complex
                símbolos decididos
        """

        # Índices de símbolo de las partes reales e imaginarias
        index_real = (
            np.floor(rx_symbol.real / self.dist).astype(int)
            + self.symbol2index_offset
        )
        index_imag = (
            np.floor(rx_symbol.imag / self.dist).astype(int)
            + self.symbol2index_offset
        )
        # Cambia los que se salen de los límites inferior o superior
        if np.abs(index_real) >= self.nsymbols_real:
            index_real = self.nsymbols_real - 1
        if np.abs(index_imag) < 0:
            index_imag = 0
        # Símbolos decididos
        decided_symbol = (
            index_real + 1j * index_imag
        ) * self.dist - self.maxsymbol

        return decided_symbol

    def soft_decision(self, rx_symbols):
        """
        Función que decide el vector de símbolos transmitidos más probable
        a partir de un vector de símbolos recibidos

        Parameters
        ----------
            rx_symbols : array_like
                vector de símbolos recibidos

        Returns
        -------
            decided_symbol : ndarray (dtype : complex)
                vector de símbolos decididos
            soft_decisions : ndarray (dtype : int)
                vector de decisiones soft
        """

        # Array_like a ndarray 1d
        rx_symbols = np.asarray(rx_symbols, dtype=complex)
        rx_symbols = np.atleast_1d(rx_symbols)
        # Inicio de la matriz de decisiones soft
        soft_decisions = np.zeros(
            (rx_symbols.size, self.nbits_symbol), dtype=float
        )

        for k in np.arange(int(self.nbits_symbol_real)):
            # Distancia mínima de la parte real del símbolo a los
            # símbolos de la constelación que tienen en la posición k
            # un bit a 0
            dist_0r = np.min(
                (
                    np.real(rx_symbols.reshape(-1, 1))
                    - self.table_index_bit[0, k, :] * self.dist
                    + self.maxsymbol_real
                )
                ** 2,
                axis=-1,
            )
            # Distancia mínima de la parte real del símbolo a los
            # símbolos de la constelación que tienen en la posición k
            # un bit a 1
            dist_1r = np.min(
                (
                    np.real(rx_symbols.reshape(-1, 1))
                    - self.table_index_bit[1, k, :] * self.dist
                    + self.maxsymbol_real
                )
                ** 2,
                axis=-1,
            )
            # Decisión soft del bit k de la parte real
            soft_decisions[:, k] = dist_0r - dist_1r
            # Distancia mínima de la parte imag del símbolo a los
            # símbolos de la constelación que tienen en la posición k
            # un bit a 0
            dist_0i = np.min(
                (
                    np.imag(rx_symbols.reshape(-1, 1))
                    - self.table_index_bit[0, k, :] * self.dist
                    + self.maxsymbol_real
                )
                ** 2,
                axis=-1,
            )
            # Distancia mínima de la parte imag del símbolo a los
            # símbolos de la constelación que tienen en la posición k
            # un bit a 1
            dist_1i = np.min(
                (
                    np.imag(rx_symbols.reshape(-1, 1))
                    - self.table_index_bit[1, k, :] * self.dist
                    + self.maxsymbol_real
                )
                ** 2,
                axis=-1,
            )
            # Decisión soft del bit k de la parte imaginaria
            soft_decisions[:, k + self.nbits_symbol_real] = dist_0i - dist_1i

        return soft_decisions.ravel()

    def plot_symbols(self, rx_symbol, title=None):
        """
        Método que representa la constelación de símbolos recibidos.

        Si se especifica el índice de la figura la crea y si no, o usa la
        actual o crea una nueva si no hay ninguna.

        Parameters
        ----------
            rx_symbol : array_like
                vector de símbolos recibidos
            title : str
                cadena de caracteres del título de la figura
        """
        # Array_like a ndarray
        rx_symbol=np.asarray(rx_symbol)
        # Número total de símbolos que se van a representar
        nsymbols = np.min([rx_symbol.size, self.nsymbols_plot])
        # Dibuja los símbolos recibidos como puntos en el plano complejo
        plt.plot(
            rx_symbol[:nsymbols].real,
            rx_symbol[:nsymbols].imag,
            "r.",
            label="símbolos",
        )
        # Array para dibujar los límites de las regiones de decisión de
        # los símbolos
        line = 1.25 * (self.nsymbols_real / 2) * self.dist * np.array([-1, 1])
        ones = np.ones(2)
        # Bucle para dibujar los límites de las regiones de decisión de
        # los símbolos
        for k in np.arange(1, self.nsymbols_real):
            if k == 1:
                label = "límites"
            else:
                label = None
            plt.plot(
                (k - self.symbol2index_offset) * self.dist * ones,
                line,
                "b--",
                label=label,
            )
            plt.plot(
                line, (k - self.symbol2index_offset) * self.dist * ones, "b--"
            )
        # Etiquetas de los ejes
        plt.xlabel(" parte real ")
        plt.ylabel(" parte imag ")
        plt.title(title)
        # relación de aspecto igual a 1
        plt.axis("square")
        # Rejilla
        plt.grid("on")
        plt.legend()

    def SERt(self, SNR):
        """
        Método que calcula la SER (Symbol Error Rate) de la constelación
        QAM cuadrada en un canal AWGN.

        Parameters
        ----------
            SNR : float or array_like
                vector de símbolos recibidos
        Returns
        -------
            SER : float
                Symbol Error Rate en canal AWGN con la SNR especificada
        """
        SNR = np.asarray(SNR, dtype=float)
        # Cálculo del término de SER con la función Q()
        Q_term = 0.5 * (
            1
            - scie.erf(
                np.sqrt(3 / (self.nsymbols_const - 1) * SNR) / np.sqrt(2)
            )
        )
        # Cálculo de la SER
        SER = (
            4 * (1 - 1 / self.nsymbols_real) * Q_term
            - (2 * (1 - 1 / self.nsymbols_real) * Q_term) ** 2
        )

        return SER
