import numpy as np
from OpenGL.GL import *
import ctypes

class VisualizadorLidar:
    def __init__(self):
        self.puntos = []
        self.vao_puntos = glGenVertexArrays(1)
        self.vbo_puntos = glGenBuffers(1)

        self.vao_rayos = glGenVertexArrays(1)
        self.vbo_rayos = glGenBuffers(1)

    def actualizar_y_dibujar(self, nuevos_puntos, posicion_dron, loc_color, loc_tipo):
        if nuevos_puntos:
            self.puntos.extend(nuevos_puntos)
            if len(self.puntos) > 70000:
                self.puntos = self.puntos[-70000:]

        if len(self.puntos) > 0:
            glUniform1i(loc_tipo, 0)
            glUniform3f(loc_color, 1.0, 0.0, 0.0)

            datos_puntos = np.array(self.puntos, dtype=np.float32).flatten()
            glBindVertexArray(self.vao_puntos)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_puntos)
            glBufferData(GL_ARRAY_BUFFER, datos_puntos.nbytes, datos_puntos, GL_DYNAMIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

            glPointSize(2.5)
            glDrawArrays(GL_POINTS, 0, len(self.puntos))

        if nuevos_puntos:
            glUniform1i(loc_tipo, 1)
            glUniform3f(loc_color, 1.0, 0.1, 0.1)

            lineas = []
            for p in nuevos_puntos:
                lineas.extend([posicion_dron[0], posicion_dron[1], posicion_dron[2]])
                lineas.extend([p[0], p[1], p[2]])

            datos_lineas = np.array(lineas, dtype=np.float32)
            glBindVertexArray(self.vao_rayos)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_rayos)
            glBufferData(GL_ARRAY_BUFFER, datos_lineas.nbytes, datos_lineas, GL_DYNAMIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

            glLineWidth(1.2)
            glDrawArrays(GL_LINES, 0, len(nuevos_puntos) * 2)
            glLineWidth(1.0)

