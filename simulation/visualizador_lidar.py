from OpenGL.GL import *
import numpy as np
import random
import ctypes

class VisualizadorLidar:
    MAX_PUNTOS_VISIBLES = 30000

    def __init__(self):
        self.puntos = []
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.capacidad_buffer = 0
        
        self.barrido_hits = []
        self.barrido_origins = []
        self.barrido_directions = []

    def actualizar(self, nuevos):
        if nuevos:
            self.puntos.extend(nuevos)
            if len(self.puntos) > self.MAX_PUNTOS_VISIBLES:
                self.puntos = self.puntos[-self.MAX_PUNTOS_VISIBLES:]

    def actualizar_barrido(self, hits, origins, directions):
        self.barrido_hits = hits
        self.barrido_origins = origins
        self.barrido_directions = directions

    def dibujar_puntos(self, loc_tipo, loc_color):
        if not self.puntos:
            return
            
        glUniform1i(loc_tipo, 0)
        glUniform3f(loc_color, 1.0, 0.0, 0.0)

        datos = np.array(self.puntos, dtype=np.float32).flatten()
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        
        if len(datos) <= self.capacidad_buffer:
            glBufferSubData(GL_ARRAY_BUFFER, 0, datos.nbytes, datos)
        else:
            glBufferData(GL_ARRAY_BUFFER, datos.nbytes, datos, GL_DYNAMIC_DRAW)
            self.capacidad_buffer = len(datos)
            
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glPointSize(2.5)
        glDrawArrays(GL_POINTS, 0, len(self.puntos))

    def dibujar_rayos_activos(self, pos_dron, nuevos_puntos):
        if not nuevos_puntos:
            return
            
        muestra = random.sample(nuevos_puntos, min(100, len(nuevos_puntos)))
        glUseProgram(0)
        glLineWidth(1.5)
        glColor3f(1.0, 0.2, 0.0)
        glBegin(GL_LINES)
        for p in muestra:
            glVertex3f(pos_dron[0], pos_dron[1] - 0.5, pos_dron[2])
            glVertex3f(p[0], p[1], p[2])
        glEnd()
        glLineWidth(1.0)

    def dibujar_barrido_rotatorio(self, pos_dron):
        if not self.barrido_hits:
            return
            
        glUseProgram(0)
        glLineWidth(2.0)
        glColor3f(0.0, 1.0, 1.0) 
        glBegin(GL_LINES)
        ox, oy, oz = pos_dron[0], pos_dron[1] - 0.5, pos_dron[2]
        for hit in self.barrido_hits:
            glVertex3f(ox, oy, oz)
            glVertex3f(hit[0], hit[1], hit[2])
        glEnd()
        glLineWidth(1.0)
