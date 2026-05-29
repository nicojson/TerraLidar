import math
import numpy as np

class M3D:
    @staticmethod
    def matriz_perspectiva(fov_grados, aspecto, z_near, z_far):
        f = 1.0 / math.tan(math.radians(fov_grados) / 2.0)
        matriz = np.zeros((4, 4), dtype=np.float32)
        matriz[0, 0] = f / aspecto
        matriz[1, 1] = f
        matriz[2, 2] = (z_far + z_near) / (z_near - z_far)
        matriz[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
        matriz[3, 2] = -1.0
        return matriz

    @staticmethod
    def matriz_vista(posicion, objetivo, up_vector):
        f = objetivo - posicion
        norm_f = np.linalg.norm(f)
        f = f / norm_f if norm_f > 0 else f
        
        s = np.cross(f, up_vector)
        norm_s = np.linalg.norm(s)
        s = s / norm_s if norm_s > 0 else s
        
        u = np.cross(s, f)

        matriz = np.eye(4, dtype=np.float32)
        matriz[0, 0:3] = s
        matriz[1, 0:3] = u
        matriz[2, 0:3] = -f
        matriz[0, 3] = -np.dot(s, posicion)
        matriz[1, 3] = -np.dot(u, posicion)
        matriz[2, 3] = np.dot(f, posicion)
        return matriz

    @staticmethod
    def traslacion(x, y, z):
        m = np.eye(4, dtype=np.float32)
        m[3, 0] = x
        m[3, 1] = y
        m[3, 2] = z
        return m

    @staticmethod
    def mul(*matrices):
        result = matrices[0]
        for m in matrices[1:]:
            result = result @ m
        return result
