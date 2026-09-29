# fuente.py
"""
Generacion de una fuente de calor gaussiana estacionaria.
"""
import numpy as np


def crear_fuente(nx, ny, dx, dy, centro_x, centro_y, sigma, amplitud):
    """
    Crea un campo 2D con perfil gaussiano.

    Parametros
    ----------
    nx, ny : int
        Numero de puntos en x e y.
    dx, dy : float
        Espaciado de la malla.
    centro_x, centro_y : float
        Centro de la gaussiana en coordenadas fisicas.
    sigma : float
        Desviacion estandar en metros.
    amplitud : float
        Valor pico (K/s).
    """
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y)
    dist2 = (X - centro_x)**2 + (Y - centro_y)**2
    fuente = amplitud * np.exp(-dist2 / (2.0 * sigma**2))
    return fuente.astype(np.float32)
