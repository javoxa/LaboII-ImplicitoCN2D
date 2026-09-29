# solver.py
"""
Clase que gestiona la resolucion con Crank-Nicolson ADI y OpenCL.
"""
import os
import numpy as np
import pyopencl as cl


class SolverCalorCN2D:
    def __init__(self, nx, ny, dx, dy, dt, alfa_arr, Q_arr,
                 T_frontera=25.0, T_inicial=25.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.T_frontera = T_frontera

        # El esquema ADI implementado requiere difusividad homogenea.
        alfa_0 = float(alfa_arr[0, 0])
        if not np.allclose(alfa_arr, alfa_0, rtol=1e-6, atol=1e-8):
            raise ValueError(
                "SolverCalorCN2D requiere alfa homogenea. "
                "Usa homogeneo=True en obtener_propiedades_material()."
            )

        self.alfa = alfa_0
        self.rx = 0.5 * self.dt * self.alfa / (self.dx * self.dx)
        self.ry = 0.5 * self.dt * self.alfa / (self.dy * self.dy)

        # ---- Configuracion OpenCL ----
        plataformas = cl.get_platforms()
        self.contexto = None
        self.cola = None

        for plataforma in plataformas:
            for dispositivo in plataforma.get_devices():
                if dispositivo.type == cl.device_type.GPU:
                    self.contexto = cl.Context([dispositivo])
                    self.cola = cl.CommandQueue(
                        self.contexto,
                        properties=cl.command_queue_properties.PROFILING_ENABLE
                    )
                    print(f"Usando dispositivo: {dispositivo.name}")
                    break
            if self.contexto:
                break

        if not self.contexto:
            raise RuntimeError("No se encontro ningun dispositivo GPU OpenCL.")

        # Compilar kernel
        ruta_kernel = os.path.join(os.path.dirname(__file__), 'kernel_calor.cl')
        with open(ruta_kernel, 'r') as f:
            fuente_kernel = f.read()

        self.programa = cl.Program(self.contexto, fuente_kernel).build()

        # ---- Arrays en host ----
        self.T = np.full((ny, nx), T_inicial, dtype=np.float32)
        self.T[0, :] = T_frontera
        self.T[-1, :] = T_frontera
        self.T[:, 0] = T_frontera
        self.T[:, -1] = T_frontera

        self.T_half = np.copy(self.T)
        self.T_nuevo = np.copy(self.T)
        self.b = np.zeros((ny, nx), dtype=np.float32)
        self.cp = np.zeros((ny, nx), dtype=np.float32)
        self.dp = np.zeros((ny, nx), dtype=np.float32)

        # ---- Buffers en dispositivo ----
        mf = cl.mem_flags
        self.buf_T = cl.Buffer(self.contexto, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.T)
        self.buf_T_half = cl.Buffer(self.contexto, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.T_half)
        self.buf_T_nuevo = cl.Buffer(self.contexto, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.T_nuevo)
        self.buf_b = cl.Buffer(self.contexto, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.b)
        self.buf_cp = cl.Buffer(self.contexto, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.cp)
        self.buf_dp = cl.Buffer(self.contexto, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dp)
        self.buf_Q = cl.Buffer(self.contexto, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q_arr)

        self.contador_pasos = 0

    def paso(self):
        """Ejecuta un paso temporal Crank-Nicolson ADI."""

        # 1) b = B_y T^n + 0.5*dt*Q
        self.programa.calcular_b_y(
            self.cola, (self.nx, self.ny), None,
            self.buf_T, self.buf_Q, self.buf_b,
            np.int32(self.nx), np.int32(self.ny),
            np.float32(self.ry), np.float32(self.dt)
        )

        # 2) Resolver A_x T_half = b
        self.programa.solve_x(
            self.cola, (self.ny - 2,), None,
            self.buf_b, self.buf_T_half, self.buf_cp, self.buf_dp,
            np.int32(self.nx), np.int32(self.ny),
            np.float32(self.rx)
        )

        # 3) Bordes de T_half
        self.programa.set_boundaries(
            self.cola, (self.nx, self.ny), None,
            self.buf_T_half,
            np.int32(self.nx), np.int32(self.ny),
            np.float32(self.T_frontera)
        )

        # 4) b = B_x T_half + 0.5*dt*Q
        self.programa.calcular_b_x(
            self.cola, (self.nx, self.ny), None,
            self.buf_T_half, self.buf_Q, self.buf_b,
            np.int32(self.nx), np.int32(self.ny),
            np.float32(self.rx), np.float32(self.dt)
        )

        # 5) Resolver A_y T_new = b
        self.programa.solve_y(
            self.cola, (self.nx - 2,), None,
            self.buf_b, self.buf_T_nuevo, self.buf_cp, self.buf_dp,
            np.int32(self.nx), np.int32(self.ny),
            np.float32(self.ry)
        )

        # 6) Bordes de T_new
        self.programa.set_boundaries(
            self.cola, (self.nx, self.ny), None,
            self.buf_T_nuevo,
            np.int32(self.nx), np.int32(self.ny),
            np.float32(self.T_frontera)
        )

        # Intercambiar punteros T <-> T_nuevo
        self.buf_T, self.buf_T_nuevo = self.buf_T_nuevo, self.buf_T
        self.contador_pasos += 1

    def ejecutar_pasos(self, n):
        """Ejecuta n pasos temporales."""
        for _ in range(n):
            self.paso()

    def obtener_temperatura(self):
        """Devuelve la temperatura actual como array numpy (ny, nx)."""
        temperatura = np.empty((self.ny, self.nx), dtype=np.float32)
        cl.enqueue_copy(self.cola, temperatura, self.buf_T)
        return temperatura
