#!/usr/bin/env python3

# main.py

"""
Programa principal: simulacion de la ecuacion de calor 2D con OpenCL.

La simulacion utiliza siempre una GPU mediante OpenCL.

Modo grafico:

    MOSTRAR_EN_VIVO = True

        Ejecuta la simulacion y muestra la temperatura
        en una ventana en tiempo real.

Modo remoto:

    MOSTRAR_EN_VIVO = False

        Ejecuta la simulacion sin interfaz grafica y
        guarda los resultados periodicamente.

No existe fallback a CPU.
"""

import os
import time

import numpy as np
import pyopencl as cl

from parametros import (
    NX,
    NY,
    LONGITUD_X,
    LONGITUD_Y,
    T_FRONTERA,
    T_INICIAL,
    FACTOR_DT_CN,
    GRAFICAR_CADA_SEGUNDOS,
    calcular_dx_dy,
    calcular_dt,
    obtener_propiedades_material,
)
from solver import SolverCalorCN2D

from fuente import crear_fuente
from visualizador import Visualizador


# ================================================================
# CONFIGURACION
# ================================================================

# ------------------------------------------------
# Modo de ejecucion
# ------------------------------------------------
#
# True:
#     muestra la simulacion en vivo.
#
# False:
#     ejecuta sin ventana y guarda resultados.
#
MOSTRAR_EN_VIVO = True

# ------------------------------------------------
# Guardado de resultados
# ------------------------------------------------
#
# Solo se utiliza cuando MOSTRAR_EN_VIVO = False.
#
GUARDAR_CADA = 10000


# Directorio donde se almacenan los resultados.

DIRECTORIO_RESULTADOS = "resultados"


# Guardar imagen PNG ademas del archivo NPY.

GUARDAR_IMAGENES = True


# ------------------------------------------------
# GPU
# ------------------------------------------------
#
# La simulacion requiere una GPU.
#
# No existe fallback a CPU.
#

REQUIERE_GPU = True
#Modificar en caso de varias GPUs

GPU_ID = 0


# ================================================================
# BUSCAR GPU
# ================================================================

def obtener_gpu():

    plataformas = cl.get_platforms()

    dispositivos_gpu = []

    for plataforma in plataformas:

        dispositivos = plataforma.get_devices(
            device_type=cl.device_type.GPU
        )

        for dispositivo in dispositivos:

            dispositivos_gpu.append(
                (plataforma, dispositivo)
            )

    # ------------------------------------------------------------
    # No se encontro ninguna GPU
    # ------------------------------------------------------------

    if len(dispositivos_gpu) == 0:

        print("")
        print("ERROR: no se encontro ninguna GPU OpenCL.")
        print("")
        print("La simulacion requiere una GPU.")
        print("No se utilizara CPU como alternativa.")
        print("")

        raise RuntimeError(
            "No hay ninguna GPU OpenCL disponible."
        )

    # ------------------------------------------------------------
    # GPU_ID fuera de rango
    # ------------------------------------------------------------

    if GPU_ID < 0 or GPU_ID >= len(dispositivos_gpu):

        print("")
        print("ERROR: GPU_ID fuera de rango.")
        print("")
        print("GPUs disponibles: %d" % len(dispositivos_gpu))
        print("GPU_ID solicitado: %d" % GPU_ID)
        print("")

        for i, datos in enumerate(dispositivos_gpu):

            plataforma, dispositivo = datos

            print(
                "[%d] %s"
                % (
                    i,
                    dispositivo.name
                )
            )

        print("")

        raise RuntimeError(
            "GPU_ID no valido."
        )

    # ------------------------------------------------------------
    # Seleccionar GPU
    # ------------------------------------------------------------

    plataforma, dispositivo = dispositivos_gpu[GPU_ID]

    print("")
    print("=" * 60)
    print("GPU SELECCIONADA")
    print("=" * 60)

    print(
        "Plataforma:       %s"
        % plataforma.name
    )

    print(
        "Dispositivo:      %s"
        % dispositivo.name
    )

    print(
        "Fabricante:       %s"
        % dispositivo.vendor
    )

    print(
        "OpenCL:            %s"
        % dispositivo.version
    )

    print(
        "Memoria:           %.2f GB"
        % (
            dispositivo.global_mem_size
            / (1024.0 ** 3)
        )
    )

    print(
        "Compute Units:     %d"
        % dispositivo.max_compute_units
    )

    print(
        "Max work group:    %d"
        % dispositivo.max_work_group_size
    )

    print("=" * 60)
    print("")

    return plataforma, dispositivo


# ================================================================
# GUARDAR RESULTADO
# ================================================================

def guardar_resultado(T, paso):

    # ------------------------------------------------------------
    # Crear directorio
    # ------------------------------------------------------------

    if not os.path.exists(DIRECTORIO_RESULTADOS):

        os.makedirs(DIRECTORIO_RESULTADOS)

    # ------------------------------------------------------------
    # Guardar matriz NPY
    # ------------------------------------------------------------

    nombre = os.path.join(
        DIRECTORIO_RESULTADOS,
        "temperatura_%010d.npy" % paso
    )

    np.save(
        nombre,
        T
    )

    print(
        "Resultado guardado: %s"
        % nombre
    )

    # ------------------------------------------------------------
    # Guardar imagen
    # ------------------------------------------------------------

    if GUARDAR_IMAGENES:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        nombre_png = os.path.join(
            DIRECTORIO_RESULTADOS,
            "temperatura_%010d.png" % paso
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.imshow(
            T,
            origin="lower",
            aspect="auto",
            cmap="inferno"
        )

        plt.colorbar(
            label="Temperatura"
        )

        plt.xlabel("x")
        plt.ylabel("y")

        plt.title(
            "Ecuacion de calor 2D - paso %d"
            % paso
        )

        plt.tight_layout()

        plt.savefig(
            nombre_png,
            dpi=150
        )

        plt.close()

        print(
            "Imagen guardada: %s"
            % nombre_png
        )


# ================================================================
# PROGRAMA PRINCIPAL
# ================================================================

def principal():

    # ============================================================
    # GPU
    # ============================================================

    if REQUIERE_GPU:

        plataforma, dispositivo = obtener_gpu()

    # ============================================================
    # Geometria
    # ============================================================

    nx = NX
    ny = NY

    dx, dy = calcular_dx_dy()

    # ============================================================
    # Material
    # ============================================================

    # Cambiar homogeneo=False para material no homogeneo.

    alfa_arr, rho_cp_arr, k_arr = (
        obtener_propiedades_material(
            nx,
            ny,
            homogeneo=True
        )
    )

    # Valor representativo para calcular dt.

    alfa_tipica = alfa_arr[0, 0]

    # ============================================================
    # Paso temporal Crank-Nicolson
    # ============================================================

    dt_cfl = calcular_dt(dx, dy, alfa_tipica)
    dt = dt_cfl * FACTOR_DT_CN

    print(
        "dx = %.6f m, dy = %.6f m"
        % (dx, dy)
    )
    print(
        "dt CFL explicito = %.6e s"
        % dt_cfl
    )
    print(
        "FACTOR_DT_CN = %g"
        % FACTOR_DT_CN
    )
    print(
        "dt Crank-Nicolson = %.6e s"
        % dt
    )

    # ============================================================
    # Fuente gaussiana
    # ============================================================

    # Tasa de calentamiento en K/s.

    centro_x = LONGITUD_X / 2.0
    centro_y = LONGITUD_Y / 2.0

    sigma = 0.01
    amplitud_fuente = 1e3

    Q_arr = crear_fuente(
        nx,
        ny,
        dx,
        dy,
        centro_x,
        centro_y,
        sigma,
        amplitud_fuente
    )

    potencia_total = (
        np.sum(Q_arr)
        * dx
        * dy
        * rho_cp_arr[0, 0]
    )

    print(
        "Potencia total inyectada: %.2f W"
        % potencia_total
    )

    # ============================================================
    # Inicializar solver OpenCL
    # ============================================================

    solver = SolverCalorCN2D(
        nx,
        ny,
        dx,
        dy,
        dt,
        alfa_arr,
        Q_arr,
        T_frontera=T_FRONTERA,
        T_inicial=T_INICIAL
    )

    # ============================================================
    # Visualizacion
    # ============================================================

    visualizador = None

    if MOSTRAR_EN_VIVO:

        visualizador = Visualizador(
            titulo="Ecuacion de calor 2D - Cobre",
            forma_datos=(ny, nx),
            niveles=(
                T_FRONTERA,
                T_FRONTERA + 50
            )
        )

        print(
            "Simulacion iniciada."
        )

        print(
            "Cierra la ventana para detener."
        )

    else:

        print(
            "Simulacion iniciada en modo remoto."
        )

        print(
            "No se mostrara ninguna ventana."
        )

        print(
            "Los resultados se guardaran cada %d pasos."
            % GUARDAR_CADA
        )

    # ============================================================
    # Variables de medicion
    # ============================================================

    paso = 0

    t_inicio = time.perf_counter()
    # ============================================================
    # Frecuencia de visualizacion por tiempo simulado
    # ============================================================

    pasos_por_grafico = max(
        1,
        int(round(GRAFICAR_CADA_SEGUNDOS / dt))
    )

    print(
        "Graficando cada %.4f s simulados = %d pasos CN."
        % (GRAFICAR_CADA_SEGUNDOS, pasos_por_grafico)
    )

    try:

        while True:

            # ====================================================
            # Ejecutar pasos en GPU
            # ====================================================

            solver.ejecutar_pasos(pasos_por_grafico)
            paso += pasos_por_grafico

            # ====================================================
            # MODO GRAFICO
            # ====================================================

            if MOSTRAR_EN_VIVO:

                T = solver.obtener_temperatura()

                visualizador.actualizar(
                    T
                )

            # ====================================================
            # MODO REMOTO
            # ====================================================

            else:

                if paso % GUARDAR_CADA == 0:

                    T = solver.obtener_temperatura()

                    guardar_resultado(
                        T,
                        paso
                    )

            # ====================================================
            # Mostrar rendimiento
            # ====================================================

            if paso % (10 * pasos_por_grafico) == 0:

                tiempo = (
                    time.perf_counter()
                    - t_inicio
                )

                pasos_por_segundo = (
                    paso / tiempo
                )

                tiempo_por_paso = (
                    tiempo / paso
                )

                tiempo_fisico = (
                    paso * dt
                )

                print("")

                print(
                    "Paso:              %d"
                    % paso
                )

                # ------------------------------------------------
                # Temperatura maxima
                # ------------------------------------------------

                if MOSTRAR_EN_VIVO:

                    temperatura_maxima = T.max()

                    print(
                        "T_max:             %.2f C"
                        % temperatura_maxima
                    )

                # ------------------------------------------------
                # Rendimiento
                # ------------------------------------------------

                print(
                    "Tiempo real:       %.3f s"
                    % tiempo
                )

                print(
                    "Rendimiento:       %.1f pasos/s"
                    % pasos_por_segundo
                )

                print(
                    "Tiempo por paso:   %.4f ms"
                    % (
                        tiempo_por_paso * 1000.0
                    )
                )

                print(
                    "Tiempo fisico:     %.6f s"
                    % tiempo_fisico
                )

            # ====================================================
            # Comprobar ventana
            # ====================================================

            if MOSTRAR_EN_VIVO:

                if visualizador.esta_cerrado():

                    break

    except KeyboardInterrupt:

        print("")
        print(
            "Interrumpido por el usuario."
        )

    # ============================================================
    # Esperar a que la GPU termine
    # ============================================================

    solver.cola.finish()

    # ============================================================
    # Resultado final
    # ============================================================

    T_final = solver.obtener_temperatura()

    guardar_resultado(
        T_final,
        paso
    )

    # ============================================================
    # Tiempo total
    # ============================================================

    t_total = (
        time.perf_counter()
        - t_inicio
    )

    if paso > 0:

        rendimiento_medio = (
            paso / t_total
        )

        tiempo_medio_paso = (
            t_total / paso
        )

        tiempo_fisico_total = (
            paso * dt
        )

    else:

        rendimiento_medio = 0.0
        tiempo_medio_paso = 0.0
        tiempo_fisico_total = 0.0

    # ============================================================
    # Resultado final
    # ============================================================

    print("")
    print("=" * 60)
    print("SIMULACION FINALIZADA")
    print("=" * 60)

    print(
        "Tiempo total de ejecucion: %.3f s"
        % t_total
    )

    print(
        "Pasos realizados:          %d"
        % paso
    )

    print(
        "Rendimiento medio:         %.2f pasos/s"
        % rendimiento_medio
    )

    print(
        "Tiempo medio por paso:     %.6f ms"
        % (
            tiempo_medio_paso * 1000.0
        )
    )

    print(
        "Tiempo fisico simulado:    %.6f s"
        % tiempo_fisico_total
    )

    if t_total > 0.0:

        aceleracion_tiempo_real = (
            tiempo_fisico_total
            / t_total
        )

        print(
            "Relacion fisico/real:      %.4fx"
            % aceleracion_tiempo_real
        )

    print(
        "Temperatura maxima final:  %.2f C"
        % T_final.max()
    )

    print("=" * 60)


# ================================================================
# EJECUCION
# ================================================================

if __name__ == "__main__":

    principal()
