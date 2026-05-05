import numpy as np
from OpenGL.GL import *
import threading

class NubePuntos:
    MAX_PUNTOS = 10_000_000  

    def __init__(self):
        self.puntos = np.zeros((self.MAX_PUNTOS, 3), dtype=np.float32)
        self.colores = np.zeros((self.MAX_PUNTOS, 3), dtype=np.float32)
        self.cabeza = 0
        self.total = 0

        self.vao = glGenVertexArrays(1)
        self.vbo_pos = glGenBuffers(1)
        self.vbo_col = glGenBuffers(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, self.puntos.nbytes, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, self.colores.nbytes, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        

        # Para asegurar que la subida a GPU y la adición de puntos no colisionen
        self.lock = threading.Lock()

    @staticmethod
    def altura_a_color_batch(y_array, y_min=-0.5, y_max=1.2):
        t = np.clip((y_array - y_min) / (y_max - y_min), 0.0, 1.0)
        colores = np.zeros((len(y_array), 3), dtype=np.float32)
        
        mask1 = t < 0.25
        colores[mask1, 1] = 4 * t[mask1]
        colores[mask1, 2] = 1.0
        
        mask2 = (t >= 0.25) & (t < 0.5)
        colores[mask2, 1] = 1.0
        colores[mask2, 2] = 1.0 - 4 * (t[mask2] - 0.25)
        
        mask3 = (t >= 0.5) & (t < 0.75)
        colores[mask3, 0] = 4 * (t[mask3] - 0.5)
        colores[mask3, 1] = 1.0
        
        mask4 = t >= 0.75
        colores[mask4, 0] = 1.0
        colores[mask4, 1] = 1.0 - 4 * (t[mask4] - 0.75)
        
        return colores

    def agregar(self, nuevos):
        num_nuevos = len(nuevos)
        if num_nuevos == 0: 
            return

        # Cuando viene de la GPU, "nuevos" ya es una lista de listas o un array 2D
        nuevos_np = np.array(nuevos, dtype=np.float32)
        if len(nuevos_np.shape) != 2 or nuevos_np.shape[1] != 3:
            # Check rápido de seguridad
            return

        y_array = nuevos_np[:, 1]
        colores_np = self.altura_a_color_batch(y_array)

        # Controlar si estamos en el límite
        if self.total >= self.MAX_PUNTOS:
            # Ya hemos alcanzado 10M, no agregar más
            return
            
        # Calcular cuántos puntos podemos agregar sin exceder el máximo
        espacio_disponible = self.MAX_PUNTOS - self.total
        puntos_a_agregar = min(num_nuevos, espacio_disponible)
        
        # Agregar linealmente sin envolver
        idx_fin = self.total + puntos_a_agregar
        self.puntos[self.total:idx_fin] = nuevos_np[:puntos_a_agregar]
        self.colores[self.total:idx_fin] = colores_np[:puntos_a_agregar]
        self.total += puntos_a_agregar
        self.cabeza = self.total

    def subir_gpu(self):
        """Sube todos los puntos disponibles a GPU de forma completa y sincrónica"""
        if self.total == 0: 
            return
        
        # Subir TODOS los puntos que tenemos - garantiza visualización completa
        # hasta 10M sin pérdidas
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.total * 3 * 4, self.puntos[:self.total])
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.total * 3 * 4, self.colores[:self.total])

    def dibujar(self, loc_tipo):
        if self.total == 0: 
            return
        glUniform1i(loc_tipo, 4)
        glBindVertexArray(self.vao)
        # Hacemos los puntos un poco más grandes para que se vean más sólidos
        glPointSize(2.0)
        # Dibujar todos los puntos que se han subido a GPU correctamente
        glDrawArrays(GL_POINTS, 0, self.total)
        glPointSize(1.0)
        glBindVertexArray(0)
