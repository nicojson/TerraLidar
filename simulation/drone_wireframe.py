import numpy as np
import math
from OpenGL.GL import *
import ctypes

class DronWireframe:
    def __init__(self):
        self.posicion = np.array([-20.0, 12.0, -15.0], dtype=np.float32)
        self.tiempo = 0.0

        s = 0.5
        self.vertices_lines = np.array([
            -s, s, -s, s, s, -s, s, s, -s, s, s, s, s, s, s, -s, s, s, -s, s, s, -s, s, -s,
            -s, -s, -s, s, -s, -s, s, -s, -s, s, -s, s, s, -s, s, -s, -s, s, -s, -s, s, -s, -s, -s,
            -s, s, -s, -s, -s, -s, s, s, -s, s, -s, -s, s, s, s, s, -s, s, -s, s, s, -s, -s, s,
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices_lines.nbytes, self.vertices_lines, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

    def actualizar_vuelo(self):
        self.tiempo += 0.02
        self.posicion[0] = math.sin(self.tiempo) * 18.0
        self.posicion[2] += 0.04
        if self.posicion[2] > 20.0:
            self.posicion[2] = -20.0

    def dibujar(self, shader_programa, loc_modelo, loc_color, loc_tipo):
        matriz_modelo = np.eye(4, dtype=np.float32)
        matriz_modelo[3, 0:3] = self.posicion
        glUniformMatrix4fv(loc_modelo, 1, GL_TRUE, matriz_modelo)

        glUniform3f(loc_color, 1.0, 1.0, 0.0)
        glUniform1i(loc_tipo, 1)

        glBindVertexArray(self.vao)
        glLineWidth(2.0)
        glDrawArrays(GL_LINES, 0, 24)
        glLineWidth(1.0)
        glBindVertexArray(0)

        glUniformMatrix4fv(loc_modelo, 1, GL_TRUE, np.eye(4, dtype=np.float32))

