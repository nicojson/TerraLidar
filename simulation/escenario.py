import numpy as np
from OpenGL.GL import *
import ctypes

class Escenario3DAgricola:
    def __init__(self):
        tamaño = 50.0
        self.vertices = np.array([
            -tamaño, 0.0, -tamaño, tamaño, 0.0, -tamaño,
            -tamaño, 0.0, tamaño, tamaño, 0.0, tamaño,
        ], dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

    def dibujar(self, loc_tipo):
        glUniform1i(loc_tipo, 2)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

