import glfw
import math
import numpy as np
from core.math3d import M3D

class CamaraLibre:
    def __init__(self, ventana):
        self.ventana = ventana
        self.posicion = np.array([0.0, 18.0, 35.0], dtype=np.float32)
        self.frente = np.array([0.0, -0.4, -1.0], dtype=np.float32)
        self.arriba = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.velocidad = 0.5
        self.velocidad_rotacion = 0.005

        self.yaw = math.atan2(self.frente[2], self.frente[0])
        self.pitch = math.asin(self.frente[1])

        self.ultima_x = 0.0
        self.ultima_y = 0.0
        self.primer_movimiento = True

        # Hacemos que el cursor sea visible y no registramos el callback para bloquear el control del mouse
        glfw.set_input_mode(ventana, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def actualizar_angulos_desde_frente(self):
        frente_norm = self.frente / np.linalg.norm(self.frente)
        self.yaw = math.atan2(frente_norm[2], frente_norm[0])
        self.pitch = math.asin(np.clip(frente_norm[1], -1.0, 1.0))

    def rotar_con_raton(self, delta_x, delta_y):
        self.yaw += delta_x * self.velocidad_rotacion
        self.pitch -= delta_y * self.velocidad_rotacion

        self.pitch = np.clip(self.pitch, -math.pi / 2 + 0.1, math.pi / 2 - 0.1)

        cos_pitch = math.cos(self.pitch)
        self.frente[0] = math.cos(self.yaw) * cos_pitch
        self.frente[1] = math.sin(self.pitch)
        self.frente[2] = math.sin(self.yaw) * cos_pitch

    def zoom(self, factor):
        self.velocidad = np.clip(self.velocidad + factor, 0.1, 3.0)

    def obtener_matriz_vista(self):
        norm_f = self.frente / np.linalg.norm(self.frente)
        return M3D.look_at(self.posicion, self.posicion + norm_f, self.arriba)

    def mover(self, direccion):
        norm_frente = self.frente / np.linalg.norm(self.frente)
        derecha = np.cross(norm_frente, self.arriba)
        derecha = derecha / np.linalg.norm(derecha)
        arriba_local = np.cross(derecha, norm_frente)
        arriba_local = arriba_local / np.linalg.norm(arriba_local)

        if direccion == "ADELANTE":
            self.posicion += norm_frente * self.velocidad
        elif direccion == "ATRAS":
            self.posicion -= norm_frente * self.velocidad
        elif direccion == "IZQUIERDA":
            self.posicion -= derecha * self.velocidad
        elif direccion == "DERECHA":
            self.posicion += derecha * self.velocidad
        elif direccion == "ARRIBA":
            self.posicion += arriba_local * self.velocidad
        elif direccion == "ABAJO":
            self.posicion -= arriba_local * self.velocidad
