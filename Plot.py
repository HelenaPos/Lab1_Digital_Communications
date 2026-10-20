##################################
# Plot
##################################
"""
Módulo para la representación gráfica de señales, espectros, etc.
"""

##################################
# Librerías
##################################
import numpy as np
import matplotlib.pyplot as plt
import TransferFunctions as tf

############################
# Funciones
############################


def close():
    """
    Función que cierra todas las figuras
    """
    plt.close("all")


def figure(num=None, figsize=None, suptitle=None, maximize=False, layout=None):
    """
    Función que crea una figura.

    Parameters
    ----------
        num : int (optional, default:None)
            índice de la figura
        figsize : array_like (optional, default:None)
            vector de anchura y altura de la figura
        suptitle: str
            título de la figura
        maximize : bool (optional, default:False)
            indica si se maximiza la ventana o no.
        layout: str (optional: default:None)
            tipo de layout ('constrained')
    """
    figure = plt.figure(num=num, figsize=figsize, layout=layout)
    if suptitle is not None:
        figure.canvas.manager.set_window_title(str(num) + ". " + suptitle)
    if maximize:
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "window"):
            manager.window.showMaximized()


def subplot(num, polar=False):
    """
    Función que crea una subfigura dentro de otra figura.

    Parameters
    ----------
        num : int
            entero de tres cifras:
                - número de filas en que se divide la figura principal,
                - número de columnas en que se divide la figura principal,
                - posición (se numera primero por filas)
        polar :  bool
            indicador de si la subfigura es un diagrama polar o no

    """
    plt.subplot(num, polar=polar)


def discrete_signal(
    amplitudes,
    time_axis=None,
    title=None,
    colors="br",
    legend="",
    xlim=None,
    ylim=None,
    time_unit=None,
    ylabel="",
):
    """
    Función que representa una señal compleja discreta y finita (pulso,
    respuesta al impulso, etc).

    Parameters
    ----------
        amplitudes : array_like (float/complex)
            vector de muestras de la respuesta al impulso
        time_axis : array_like
            eje de tiempo
        title : str (optional,default:None)
            título de la figura
        colors : str (optional, default:'br')
            cadena de dos caracteres con los colores de la representación
            (parte real e imaginaria si la hubiera)
        legend : str (optional, default:"")
            cadena de caracteres para la legenda
        xlim :  array_like (optional, default:None)
            limites del eje x
        ylim :  array_like (optional, default:None)
            limites del eje y
        time_unit : str (optional, default:None)
            unidades de tiempo
        ylabel : str (optional, default:"")
            etiqueta del eje y
    """
    add_legend = False
    # Array_like a ndarray
    amplitudes = np.asarray(amplitudes)
    # Eje de tiempo
    if time_axis is None:
        time_axis = np.arange(amplitudes.size)
    # Añade legenda si hay cadena de caracteres
    if legend != "":
        add_legend = True
    # Representación
    if (amplitudes.dtype == "float") | (
        np.mean(amplitudes.imag**2) < 1e-3 * np.mean(amplitudes.real**2)
    ):
        # Representación señal real
        plt.stem(
            time_axis,
            amplitudes.real,
            linefmt=colors[0],
            markerfmt=colors[0],
            basefmt="k",
            label=legend,
        )
    elif amplitudes.dtype == "complex":
        add_legend = True
        if legend == "":
            labels = ["real", "imag"]
        else:
            labels = [legend + " (real)", legend + " (imag)"]
        # Representación señal compleja
        plt.stem(
            time_axis,
            amplitudes.real,
            linefmt=colors[0],
            markerfmt=colors[0],
            basefmt="k",
            label=labels[0],
        )
        plt.stem(
            time_axis,
            amplitudes.imag,
            linefmt=colors[1],
            markerfmt=colors[1],
            basefmt="k",
            label=labels[1],
        )
    else:
        raise ValueError("El tipo de datos de la señal discreta no es válido")
    # Ejes
    # Xlabel
    if time_unit is not None:
        plt.xlabel("tiempo (" + time_unit + ")")
    else:
        plt.xlabel("tiempo")
    # Ylabel
    if ylabel == "":
        plt.ylabel("amplitud")
    else:
        plt.ylabel(ylabel)
    # Grid
    plt.grid("on")
    # Título
    if title is not None:
        plt.title(title)
    # Legenda
    if add_legend:
        plt.legend()
    # Límites de los ejes
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


def continuous_signal(
    amplitudes,
    time_axis=None,
    title=None,
    styles=["b-", "r-"],
    legend="",
    fm=1,
    time_unit="",
    xlim=None,
    ylim=None,
    ylabel="",
):
    """
    Función que representa una señal 'continua' compleja finita (pulso,
    respuesta al impulso, etc.)

    Parameters
    ----------
        amplitudes : array_like (float/complex)
            vector de muestras de la respuesta al impulso
        time_axis : ndarray (dtype:int, optional-default:None)
            eje de tiempo
        title : str (optional,default:None)
            título de la figura
        styles : array_like - dtype str (optional, default:['b-','r-'])
            vector de estilos de la curva (parte real e imaginaria)
        legend : str (optional, default:"")
            cadena de caracteres para la legenda
        fm : float (optional, default:1)
            frecuencia de muestreo
        time_unit : str (optional, default:"")
            unidad de tiempo de representación
        xlim :  array_like (optional, default:None)
            limites del eje x
        ylim :  array_like (optional, default:None)
            limites del eje y
        ylabel : str (optional, default:"")
            cadena de caracteres para el eje y

    """
    # Indicador de que hay que añadir legenda
    add_legend = False
    # Cadena de caracteres de la legenda
    labels = legend
    # Array_like a ndarray
    amplitudes = np.asarray(amplitudes)
    # Eje de tiempo
    if time_axis is None:
        time_axis = np.arange(amplitudes.size)
    # Señal real o compleja
    if legend != "":
        add_legend = True
    if (amplitudes.dtype == "float") | (np.all(amplitudes.imag == 0)):
        # Representación real
        plt.plot(time_axis / fm, amplitudes.real, styles[0], label=labels)
    elif amplitudes.dtype == "complex":
        add_legend = True
        if legend == "":
            labels = ["real", "imag"]
        else:
            labels = [legend + " (real)", legend + " (imag)"]
        # Representación compleja
        plt.plot(time_axis / fm, amplitudes.real, styles[0], label=labels[0])
        plt.plot(time_axis / fm, amplitudes.imag, styles[1], label=labels[1])
    else:
        raise ValueError("El tipo de datos de la señal discreta no es válido")
    # Etiquetas Ejes
    # Xlabel
    if fm == 1:
        plt.xlabel("tiempo (x $T_S$)")
    else:
        plt.xlabel("tiempo")
    if time_unit != "":
        plt.xlabel("tiempo (" + time_unit + ")")
    # Ylabel
    if ylabel != "":
        plt.ylabel(ylabel)
    else:
        plt.ylabel("amplitud")
    # Grid
    plt.grid("on")
    # Título
    if title is not None:
        plt.title(title)
    # Legenda
    if add_legend:
        plt.legend()
    # Limites de los ejes
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


def spectrum(
    freq_response,
    freq_axis,
    style="r-",
    legend="",
    title=None,
    log=True,
    xlim=None,
    ylim=None,
    ylabel="",
    fm=1,
    freq_unit="",
    power=False
):
    """
    Función que representa el módulo de una transformada de Fourier (respuesta
    en frecuencia de un sistema, espectro de una señal, etc.)

    Parameters
    ----------
        freq_response :array_like
            vector de muestras de la respuesta en frecuencia / densidad 
            espectral de potencia
        freq_axis : array_like
            vector de frecuencia de la respuesta en frecuencia / raíz de la
            densidad espectral de potencia
        style: str (optional, default:"r-")
            cadena con el estilo de la figura (color y tipo de línea)
        legend : str (optional, default:"")
            etiqueta de la curva en la legenda
        title : str (optional, default:None)
            título de la figura
        log : bool (optional, default:True)
            indicador de si se representa en escala logarítmica (True) o
            no (False)
        xlim : array_like (optional, default:None)
            vector [xmin,xmax] de limites en el eje x
        ylim : array_like (optional, default:None)
            vector [ymin,ymax] de límites en el eje y
        ylabel : str (optional, default:"")
            cadena de caracteres de la etiqueta del eje y
        fm : float (optional, default:1)
            frecuencia de muestreo
        freq_unit: str (optional, default:None)
            unidades de frecuencia ('kHz' o 'MHz')
        power : bool (optional, default:False)
            indica si la magnitud de la respuesta es en unidades de potencia 
            o no                                   
    """
    # Representación logarítmica o decimal

    if log:
        freq_response_dB = (10+10*(1-power))* np.log10(
            abs(freq_response) + np.finfo(float).eps
        )
        plt.plot(freq_axis * fm, freq_response_dB, style, label=legend)
        plt.ylabel("dB")
    else:
        plt.plot(freq_axis * fm, np.abs(freq_response), style, label=legend)
        if power:
            plt.ylabel("módulo al cuadrado")
        else:
            plt.ylabel("módulo")
    # Etiqueta y eje x
    if freq_unit != "":
        plt.xlabel("frecuencia (" + freq_unit + ")")
    elif fm == 1:
        plt.xlabel("frecuencia (normalizada)")
    else:
        plt.xlabel("frecuencia (Hz)")
    if ylabel!="":
        plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    # Parrilla
    plt.grid("on")
    if legend != "":
        plt.legend()
    if title is not None:
        plt.title(title)


def frequency_response(
    B,
    A=[1],
    style="r-",
    legend="",
    title=None,
    log=True,
    xlim=None,
    ylim=None,
    ylabel="",
    fm=1,
    freq_unit="",
):
    """
    Función que representa la respuesta en frecuencia de un sistema

    Parameters
    ----------
        B : array_like 
            vector de coeficientes del polinomio del numerador de la 
            función de transferencia
        A : array_like (optional, default:[1])
            vector de coeficientes del polinomio del denominador de la 
            función de transferencia            
        style: str (optional, default:"r-")
            cadena con el estilo de la figura (color y tipo de línea)
        legend : str (optional, default:"")
            etiqueta de la curva en la legenda
        title : str (optional, default:None)
            título de la figura
        log : bool (optional, default:True)
            indicador de si se representa en escala logarítmica (True) o
            no (False)
        xlim : array_like (optional, default:None)
            vector [xmin,xmax] de limites en el eje x
        ylim : array_like (optional, default:None)
            vector [ymin,ymax] de límites en el eje y
        ylabel : str (optional, default:"")
            cadena de caracteres de la etiqueta del eje y
        fm : float (optional, default:1)
            frecuencia de muestreo
        freq_unit: str (optional, default:None)
            unidades de frecuencia ('kHz' o 'MHz')
                                 
    """
    [freq_axis,freq_response]=tf.frequency_response(B,A)
    
    spectrum(
        freq_response,
        freq_axis,
        style=style,
        legend=legend,
        title=title,
        log=log,
        xlim=xlim,
        ylim=ylim,
        ylabel=ylabel,
        fm=fm,
        freq_unit=freq_unit,
        power=False
    )
    

def poles_zeros(zeros, poles,title=""):
    """
    Función que representa el diagrama polo-zero de una función de
    transferencia H(z) racional.

    Parameters
    ----------
        zeros : array_like (dtype float or complex)
            vector de ceros del polinomio del numerador de la
            función de transferencia H(z).
        poles : array_like (dtype float or complex, optional - default [1])
            vector de ceros del polinomio del denominador de la
            función de transferencia H(z).
        title : str (optional, defaul:None)
            título de la figura
    """
    #Array_like a ndarray
    zeros=np.asarray(zeros)
    poles=np.asarray(poles)
    
    # Valor máximo de los ejes X e Y
    XYmax = np.max(
        np.block(
            [
                np.abs(zeros.real),
                np.abs(zeros.imag),
                np.abs(poles.real),
                np.abs(poles.imag),
            ]
        )
    )
    # Ceros
    plt.plot(
        zeros.real, zeros.imag, "rx", alpha=0.5, label="ceros", markersize=4
    )
    # Polos
    plt.plot(
        poles.real, poles.imag, "bo", alpha=0.5, label="polos", markersize=4
    )
    # círculo unidad
    circle = np.exp(1j * 2 * np.pi * np.arange(500) / 500)
    plt.plot(circle.real, circle.imag, "k--", label="círculo unidad")
    # Etiquetas
    plt.xlabel(" real ")
    plt.ylabel(" imag ")
    # Axis ratio = 1
    plt.axis("square")
    plt.axis(1.15 * XYmax * np.array([-1, 1, -1, 1]))
    # Título
    if title!="":
        plt.title(title)
    # Leyenda
    plt.legend()
    # Parrila
    plt.grid("on")
