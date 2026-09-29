# utilidades.py
"""
Funciones auxiliares (guardado, calculo de residuo, etc.).
"""
import numpy as np


def guardar_temperatura(nombre_archivo, T, dx, dy):
    """Guarda el campo de temperatura en formato NumPy."""
    np.savez(nombre_archivo, T=T, dx=dx, dy=dy)


def delta_maximo(T_viejo, T_nuevo):
    """Diferencia maxima absoluta entre dos iteraciones."""
    return np.max(np.abs(T_nuevo - T_viejo))
