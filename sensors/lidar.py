import numpy as np
import math
from .raycaster import RaycasterCompute

class SensorLidar:
    def __init__(self, densidad, escenario):
        self.densidad = densidad
        self.raycaster = RaycasterCompute(escenario.triangulos)
        self.angulo_rotacion = 0.0
        self.velocidad_rotacion = 120.0  # grados por segundo

    def emitir_rayos_aleatorios(self, pos_dron):
        origin = np.array([pos_dron[0], pos_dron[1] - 0.5, pos_dron[2]], dtype=np.float32)
        directions = np.random.uniform(-1.0, 1.0, (self.densidad, 3)).astype(np.float32)
        directions[:, 1] = -np.abs(directions[:, 1])
        directions[:, 0] *= 3.0
        directions[:, 2] *= 22.0
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        origins = np.tile(origin, (self.densidad, 1))
        return self.raycaster.trace_batch(origins, directions)

    def emitir_barrido_rotatorio(self, pos_dron, delta_time):
        self.angulo_rotacion += self.velocidad_rotacion * delta_time
        self.angulo_rotacion %= 360.0

        num_rays = 72
        angulo_inicial = math.radians(self.angulo_rotacion)
        incremento = 2.0 * math.pi / num_rays

        origin = np.array([pos_dron[0], pos_dron[1] - 0.5, pos_dron[2]], dtype=np.float32)
        origins = np.tile(origin, (num_rays, 1))
        directions = np.zeros((num_rays, 3), dtype=np.float32)

        for i in range(num_rays):
            angulo = angulo_inicial + i * incremento
            inclinacion = math.radians(15.0)
            dx = math.cos(angulo) * math.cos(inclinacion)
            dy = -math.sin(inclinacion)
            dz = math.sin(angulo) * math.cos(inclinacion)
            directions[i] = (dx, dy, dz)

        return self.raycaster.trace_batch(origins, directions), origins, directions
