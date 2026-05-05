import math
import numpy as np

class M3D:
    @staticmethod
    def perspectiva(fov_deg, aspect, zn, zf):
        f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (zf + zn) / (zn - zf)
        m[2, 3] = (2.0 * zf * zn) / (zn - zf)
        m[3, 2] = -1.0
        return m

    @staticmethod
    def look_at(eye, center, up):
        def norm(v): n = np.linalg.norm(v); return v / n if n > 0 else v

        f = norm(center - eye)
        s = norm(np.cross(f, norm(up)))
        u = np.cross(s, f)
        m = np.eye(4, dtype=np.float32)
        m[0, 0:3] = s
        m[0, 3] = -np.dot(s, eye)
        m[1, 0:3] = u
        m[1, 3] = -np.dot(u, eye)
        m[2, 0:3] = -f
        m[2, 3] = np.dot(f, eye)
        return m

    @staticmethod
    def traslacion(x, y, z):
        m = np.eye(4, dtype=np.float32)
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
        return m

    @staticmethod
    def escala(sx, sy, sz):
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        m[2, 2] = sz
        return m

    @staticmethod
    def rot_y(ang_deg):
        a = math.radians(ang_deg)
        c, s = math.cos(a), math.sin(a)
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return m

    @staticmethod
    def rot_x(ang_deg):
        a = math.radians(ang_deg)
        c, s = math.cos(a), math.sin(a)
        m = np.eye(4, dtype=np.float32)
        m[1, 1] = c
        m[1, 2] = -s
        m[2, 1] = s
        m[2, 2] = c
        return m

    @staticmethod
    def rot_z(ang_deg):
        a = math.radians(ang_deg)
        c, s = math.cos(a), math.sin(a)
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 1] = -s
        m[1, 0] = s
        m[1, 1] = c
        return m

    @staticmethod
    def mul(*matrices):
        result = matrices[0]
        for m in matrices[1:]:
            result = result @ m
        return result