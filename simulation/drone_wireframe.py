import numpy as np
from OpenGL.GL import *
import ctypes

class DronWireframe:
    def __init__(self):
        s = 2.5
        h = 0.5
        self.vertices = np.array([
            -s, h, -s, s, h, -s, s, h, -s, s, h, s,
            s, h, s, -s, h, s, -s, h, s, -s, h, -s,
            -s, -h, -s, s, -h, -s, s, -h, -s, s, -h, s,
            s, -h, s, -s, -h, s, -s, -h, s, -s, -h, -s,
            -s, h, -s, -s, -h, -s, s, h, -s, s, -h, -s,
            s, h, s, s, -h, s, -s, h, s, -s, -h, s,
            -s, h, -s, s, -h, s, -s, -h, -s, s, h, s
        ], dtype=np.float32)
        
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

    def dibujar(self, pos, loc_modelo, loc_color, loc_tipo):
        matriz = np.eye(4, dtype=np.float32)
        matriz[3, 0:3] = pos
        glUniformMatrix4fv(loc_modelo, 1, GL_TRUE, matriz)
        glUniform3f(loc_color, 1.0, 0.8, 0.0)
        glUniform1i(loc_tipo, 1)
        
        glBindVertexArray(self.vao)
        glLineWidth(4.0)
        glDrawArrays(GL_LINES, 0, len(self.vertices) // 3)
        glLineWidth(1.0)
