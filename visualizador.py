# visualizador.py
"""
Visualizacion en tiempo real con PyQtGraph.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


class Visualizador:
    def __init__(self, titulo="Calor 2D", forma_datos=(1024, 1024),
                 niveles=(25.0, 100.0)):
        # Crear aplicacion Qt si no existe
        self.aplicacion = QtWidgets.QApplication.instance()
        if self.aplicacion is None:
            self.aplicacion = QtWidgets.QApplication([])

        self.ventana = pg.GraphicsLayoutWidget(show=True, title=titulo)
        self.ventana.resize(800, 800)

        self.grafico = self.ventana.addPlot()
        self.grafico.setTitle("Temperatura (C)")
        self.imagen = pg.ImageItem()
        self.grafico.addItem(self.imagen)
        self.grafico.setAspectLocked(True)

        # Barra de color
        self.barra_color = pg.ColorBarItem(values=niveles, colorMap='inferno')
        self.barra_color.setImageItem(self.imagen)

        # Imagen inicial vacia
        self.imagen.setImage(np.zeros(forma_datos))
        self.ventana.show()
        self.aplicacion.processEvents()

    def actualizar(self, datos):
        """
        Actualiza la imagen con nuevos datos.
        datos: array 2D (ny, nx). Se transpone para coincidir con la
        orientacion (x, y) de pyqtgraph.
        """
        self.imagen.setImage(datos.T, autoLevels=False)
        self.aplicacion.processEvents()

    def esta_cerrado(self):
        """True si la ventana ha sido cerrada."""
        return not self.ventana.isVisible()
