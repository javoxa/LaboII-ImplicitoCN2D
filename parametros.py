# parametros.py
"""
Parametros fisicos y geometricos de la simulacion.
"""
import numpy as np

# Dimensiones de la malla
NX = 2*1024
NY = 2*1024

# Tamano fisico de la lamina (metros)
LONGITUD_X = 0.5      # 10 cm
LONGITUD_Y = 0.5

# Temperaturas de borde e inicial (C)
T_FRONTERA = 25.0
T_INICIAL = 25.0

# Propiedades del material (cobre por defecto)
RHO = 8960.0        # kg/m^3
CP = 385.0          # J/(kg*K)
K = 401.0           # W/(m*K)

# Factor de seguridad para la condicion CFL explicita
SEGURIDAD_CFL = 0.2

# ============================================================
# Crank-Nicolson ADI
# ============================================================
#
# Como CN-ADI es incondicionalmente estable, podemos agrandar dt
# multiplicando el dt explicito estable por este factor.
#
# Ejemplos:
#   FACTOR_DT_CN = 1    -> dt similar al explicito
#   FACTOR_DT_CN = 100  -> dt 100 veces mayor
#   FACTOR_DT_CN = 500  -> dt 500 veces mayor
#
# Ojo: aunque sea estable, si el factor es muy grande se pierde
# precision temporal y la animacion puede verse incorrecta.
# Arranca con 50 o 100 y ajusta.
FACTOR_DT_CN = 100.0

# Cada cuantos segundos simulados se actualiza la visualizacion
# y/o se guarda una imagen.
GRAFICAR_CADA_SEGUNDOS = 0.05


def calcular_dx_dy():
    """Calcula el espaciado de la malla."""
    dx = LONGITUD_X / (NX - 1)
    dy = LONGITUD_Y / (NY - 1)
    return dx, dy


def calcular_dt(dx, dy, alfa):
    """
    Calcula el paso temporal maximo estable (explicito, 2D).
    alfa: difusividad termica (k/(rho*cp))
    """
    dt_cfl = SEGURIDAD_CFL * min(dx, dy)**2 / (4.0 * alfa)
    return dt_cfl


def obtener_alfa(rho=RHO, cp=CP, k=K):
    """Difusividad termica homogenea."""
    return k / (rho * cp)


def obtener_propiedades_material(nx=1024, ny=1024, homogeneo=True):
    """
    Devuelve los campos de difusividad (alfa), capacidad calorifica
    volumetrica (rho*cp) y conductividad (k).
    Si homogeneo=False, genera un ejemplo no homogeneo.
    """
    alfa0 = obtener_alfa()
    if homogeneo:
        alfa = np.full((ny, nx), alfa0, dtype=np.float32)
        rho_cp = np.full((ny, nx), RHO * CP, dtype=np.float32)
        k_arr = np.full((ny, nx), K, dtype=np.float32)
    else:
        x = np.linspace(0, LONGITUD_X, nx)
        y = np.linspace(0, LONGITUD_Y, ny)
        X, Y = np.meshgrid(x, y)
        # Variacion espacial suave de k
        k_arr = K * (1.0 + 0.2 * np.sin(2 * np.pi * X / LONGITUD_X)
                            * np.cos(2 * np.pi * Y / LONGITUD_Y))
        rho = RHO
        cp = CP
        rho_cp = np.full((ny, nx), rho * cp, dtype=np.float32)
        alfa = k_arr / (rho * cp)
    return alfa.astype(np.float32), rho_cp.astype(np.float32), k_arr.astype(np.float32)
