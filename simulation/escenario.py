import math
import numpy as np
from OpenGL.GL import *
import ctypes

class Escenario3DAgricola:
    def __init__(self, tipo_terreno=1):
        self.tipo_terreno = tipo_terreno
        self.triangulos = []
        self.generar_malla()

    def calcular_altura(self, x, z):
        if self.tipo_terreno == 2:
            return math.sin(x * 0.1) * 2.5
        elif self.tipo_terreno == 3:
            return math.sin(x * 0.15) * 4.5 + math.cos(z * 0.15) * 4.5
        return 0.0

    def generar_malla(self):
        tamano = 100.0
        resolucion = 60
        paso = (tamano * 2) / resolucion
        vertices = []
        self.triangulos = []

        for i in range(resolucion):
            for j in range(resolucion):
                x0 = -tamano + (i * paso)
                z0 = -tamano + (j * paso)
                x1 = x0 + paso
                z1 = z0 + paso

                y00 = self.calcular_altura(x0, z0)
                y10 = self.calcular_altura(x1, z0)
                y01 = self.calcular_altura(x0, z1)
                y11 = self.calcular_altura(x1, z1)

                v00 = np.array([x0, y00, z0])
                v10 = np.array([x1, y10, z0])
                v01 = np.array([x0, y01, z1])
                v11 = np.array([x1, y11, z1])

                vertices.extend([x0, y00, z0, x1, y10, z0, x0, y01, z1])
                self.triangulos.append((v00, v10, v01))
                
                vertices.extend([x1, y10, z0, x1, y11, z1, x0, y01, z1])
                self.triangulos.append((v10, v11, v01))

        self.vertices_array = np.array(vertices, dtype=np.float32)
        self.cantidad_vertices = len(self.vertices_array) // 3

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices_array.nbytes, self.vertices_array, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

    def dibujar(self, loc_tipo):
        glUniform1i(loc_tipo, 2)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.cantidad_vertices)
