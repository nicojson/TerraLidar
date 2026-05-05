import numpy as np
import random

class SensorLidar:
    def __init__(self):
        self.alcance = 150.0
        self.normal_suelo = np.array([0.0, 1.0, 0.0])
        self.punto_suelo = np.array([0.0, 0.0, 0.0])

    def emitir_rayos(self, posicion_dron):
        puntos_generados = []
        for _ in range(50):
            dx = random.uniform(-0.6, 0.6)
            dz = random.uniform(-0.6, 0.6)
            direccion = np.array([dx, -1.0, dz])
            direccion = direccion / np.linalg.norm(direccion)

            prod_punto = np.dot(self.normal_suelo, direccion)
            if abs(prod_punto) > 1e-6:
                t = np.dot(self.normal_suelo, self.punto_suelo - posicion_dron) / prod_punto
                if 0 < t < self.alcance:
                    impacto = posicion_dron + (t * direccion)
                    impacto[1] += random.uniform(0.0, 0.3)
                    puntos_generados.append(impacto)
        return puntos_generados

