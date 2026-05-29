import math
import numpy as np
from .math3d import M3D

class CamaraLibre:
    def __init__(self):
        self.posicion = np.array([0.0, 65.0, 105.0], dtype=np.float32)
        self.frente = np.array([0.0, -0.55, -0.8], dtype=np.float32)
        self.frente = self.frente / np.linalg.norm(self.frente)
        self.arriba = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.velocidad = 1.0
        self.velocidad_rotacion = 0.005
        self.yaw = math.atan2(self.frente[2], self.frente[0])
        self.pitch = math.asin(self.frente[1])
        self.ultima_x = 0.0
        self.ultima_y = 0.0
        self.primer_movimiento = True

    def rotar_con_raton(self, delta_x, delta_y):
        self.yaw += delta_x * self.velocidad_rotacion
        self.pitch -= delta_y * self.velocidad_rotacion
        self.pitch = np.clip(self.pitch, -math.pi / 2 + 0.1, math.pi / 2 - 0.1)

        cos_pitch = math.cos(self.pitch)
        self.frente[0] = math.cos(self.yaw) * cos_pitch
        self.frente[1] = math.sin(self.pitch)
        self.frente[2] = math.sin(self.yaw) * cos_pitch
        self.frente = self.frente / np.linalg.norm(self.frente)

    def obtener_matriz_vista(self):
        return M3D.matriz_vista(self.posicion, self.posicion + self.frente, self.arriba)

    def mover(self, direccion):
        derecha = np.cross(self.frente, self.arriba)
        derecha = derecha / np.linalg.norm(derecha)
        
        if direccion == "ADELANTE":
            self.posicion += self.frente * self.velocidad
        elif direccion == "ATRAS":
            self.posicion -= self.frente * self.velocidad
        elif direccion == "IZQUIERDA":
            self.posicion -= derecha * self.velocidad
        elif direccion == "DERECHA":
            self.posicion += derecha * self.velocidad
        elif direccion == "ARRIBA":
            self.posicion += self.arriba * self.velocidad
        elif direccion == "ABAJO":
            self.posicion -= self.arriba * self.velocidad
