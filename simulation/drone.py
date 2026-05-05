import math
import numpy as np
from OpenGL.GL import *
import ctypes
from core.math3d import M3D

def generar_cuerpo_dron():
    lineas = []

    r = 0.55;
    h = 0.18
    n_sides = 8
    top_pts, bot_pts = [], []
    for i in range(n_sides):
        ang = 2 * math.pi * i / n_sides
        x, z = r * math.cos(ang), r * math.sin(ang)
        top_pts.append((x, h, z))
        bot_pts.append((x, -h, z))

    for i in range(n_sides):
        a, b = top_pts[i], top_pts[(i + 1) % n_sides]
        lineas += [a[0], a[1], a[2], b[0], b[1], b[2]]
        a, b = bot_pts[i], bot_pts[(i + 1) % n_sides]
        lineas += [a[0], a[1], a[2], b[0], b[1], b[2]]
        if i % 2 == 0:
            t, bt = top_pts[i], bot_pts[i]
            lineas += [t[0], t[1], t[2], bt[0], bt[1], bt[2]]

    arm_angles = [45, 135, 225, 315]
    arm_len = 1.8
    arm_verts = []
    for ang_deg in arm_angles:
        ang = math.radians(ang_deg)
        tip_x = arm_len * math.cos(ang)
        tip_z = arm_len * math.sin(ang)
        lineas += [0, 0.05, 0, tip_x, 0.05, tip_z]
        lineas += [0, -0.05, 0, tip_x, -0.05, tip_z]
        lineas += [0, 0.05, 0, 0, -0.05, 0]
        arm_verts.append((tip_x, tip_z))

        nr = 12;
        rr = 0.45
        for i in range(nr):
            a1 = 2 * math.pi * i / nr
            a2 = 2 * math.pi * (i + 1) / nr
            x1 = tip_x + rr * math.cos(a1);
            z1 = tip_z + rr * math.sin(a1)
            x2 = tip_x + rr * math.cos(a2);
            z2 = tip_z + rr * math.sin(a2)
            lineas += [x1, 0.05, z1, x2, 0.05, z2]

        mr = 0.12
        for i in range(6):
            a1 = 2 * math.pi * i / 6
            a2 = 2 * math.pi * (i + 1) / 6
            for dy in [0.12, -0.12]:
                lineas += [
                    tip_x + mr * math.cos(a1), dy, tip_z + mr * math.sin(a1),
                    tip_x + mr * math.cos(a2), dy, tip_z + mr * math.sin(a2)
                ]
            lineas += [
                tip_x + mr * math.cos(a1 * 2), 0.12, tip_z + mr * math.sin(a1 * 2),
                tip_x + mr * math.cos(a1 * 2), -0.12, tip_z + mr * math.sin(a1 * 2),
            ]

    for ang_deg in [45, 135, 225, 315]:
        ang = math.radians(ang_deg)
        px = 0.9 * math.cos(ang);
        pz = 0.9 * math.sin(ang)
        lineas += [px, -h, pz, px, -0.55, pz]
        lineas += [px - 0.15, -0.55, pz, px + 0.15, -0.55, pz]

    ls = 0.18
    for i in range(8):
        a1 = 2 * math.pi * i / 8
        a2 = 2 * math.pi * (i + 1) / 8
        for dy in [-0.25, -0.55]:
            lineas += [
                ls * math.cos(a1), dy, ls * math.sin(a1),
                ls * math.cos(a2), dy, ls * math.sin(a2)
            ]
        lineas += [ls * math.cos(a1), -0.25, ls * math.sin(a1), ls * math.cos(a1), -0.55, ls * math.sin(a1)]

    lineas += [0, h, 0, 0, h + 0.4, 0]
    lineas += [-0.05, h + 0.38, 0, 0.05, h + 0.38, 0]
    lineas += [0, h + 0.38, -0.05, 0, h + 0.38, 0.05]

    return np.array(lineas, dtype=np.float32), arm_verts

class DronAgricola:
    def __init__(self):
        # Empezamos desde más atrás para un área mayor
        self.pos = np.array([-40.0, 16.0, -40.0], dtype=np.float32)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.roll = 0.0
        self.pitch_d = 0.0
        self.yaw_d = 0.0
        self.t = 0.0
        self.rotor_ang = 0.0
        self.led_on = True
        self.led_timer = 0.0
        self.completado = False

        self.waypoints = self._generar_ruta()
        self.wp_idx = 0
        self.wp_blend = 0.0

        self.lineas_dron, self.arm_tips = generar_cuerpo_dron()
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.lineas_dron.nbytes, self.lineas_dron, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        self.rotor_vao = glGenVertexArrays(1)
        self.rotor_vbo = glGenBuffers(1)

    def _generar_ruta(self):
        pts = []
        alt = 16.0
        # Hacemos una ruta mucho más grande para generar un mapa extenso
        for col in range(-8, 9):
            x = col * 5.0
            if col % 2 == 0:
                pts.append(np.array([x, alt, -40.0], dtype=np.float32))
                pts.append(np.array([x, alt, 40.0], dtype=np.float32))
            else:
                pts.append(np.array([x, alt, 40.0], dtype=np.float32))
                pts.append(np.array([x, alt, -40.0], dtype=np.float32))
        return pts

    def actualizar(self, dt):
        self.t += dt
        self.rotor_ang += dt * 1200.0

        self.led_timer += dt
        if self.led_timer > 0.5:
            self.led_on = not self.led_on
            self.led_timer = 0.0

        if not self.completado:
            objetivo = self.waypoints[self.wp_idx]
            diff = objetivo - self.pos
            dist = np.linalg.norm(diff)

            if dist < 1.2:
                if self.wp_idx < len(self.waypoints) - 1:
                    self.wp_idx += 1
                else:
                    self.completado = True  # ¡Misión terminada!
            else:
                dir_norm = diff / dist
                # Velocidad base aumentada aún más
                fuerza = 20.0
                self.vel += dir_norm * fuerza * dt
                
        self.vel *= 0.92
        self.pos += self.vel * dt

        hvel = np.array([self.vel[0], 0, self.vel[2]])
        spd = np.linalg.norm(hvel)
        self.roll = math.atan2(self.vel[0], 5.0) * 12.0
        self.pitch_d = -spd * 3.0
        self.pos[1] += math.sin(self.t * 3.7) * 0.015

    def _dibujar_rotores_animados(self, shader, loc_modelo, loc_color, loc_tipo):
        for tip_x, tip_z in self.arm_tips:
            mat = M3D.mul(
                M3D.traslacion(self.pos[0], self.pos[1], self.pos[2]),
                M3D.rot_y(-self.yaw_d),
                M3D.rot_x(self.pitch_d),
                M3D.rot_z(self.roll),
                M3D.traslacion(tip_x, 0.05, tip_z),
            )
            aspa_ang = self.rotor_ang + tip_x * 30
            r_aspa = 0.4
            for i in range(2):
                a = math.radians(aspa_ang + i * 90)
                x1 = r_aspa * math.cos(a);
                z1 = r_aspa * math.sin(a)
                x2 = -r_aspa * math.cos(a);
                z2 = -r_aspa * math.sin(a)
                datos = np.array([x1, 0, z1, x2, 0, z2], dtype=np.float32)
                glBindVertexArray(self.rotor_vao)
                glBindBuffer(GL_ARRAY_BUFFER, self.rotor_vbo)
                glBufferData(GL_ARRAY_BUFFER, datos.nbytes, datos, GL_DYNAMIC_DRAW)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glEnableVertexAttribArray(0)
                glUniformMatrix4fv(loc_modelo, 1, GL_TRUE, mat)
                glUniform3f(loc_color, 0.85, 0.85, 0.85)
                glUniform1i(loc_tipo, 1)
                glLineWidth(2.5)
                glDrawArrays(GL_LINES, 0, 2)

    def dibujar(self, shader, loc_modelo, loc_color, loc_tipo):
        mat_dron = M3D.mul(
            M3D.traslacion(self.pos[0], self.pos[1], self.pos[2]),
            M3D.rot_y(-self.yaw_d),
            M3D.rot_x(self.pitch_d),
            M3D.rot_z(self.roll),
        )
        glUniformMatrix4fv(loc_modelo, 1, GL_TRUE, mat_dron)
        glUniform1i(loc_tipo, 3)
        glUniform3f(loc_color, 0.85, 0.85, 0.85)
        glBindVertexArray(self.vao)
        glLineWidth(1.8)
        glDrawArrays(GL_LINES, 0, len(self.lineas_dron) // 3)
        glLineWidth(1.0)

        if self.led_on:
            glUniform1i(loc_tipo, 0)
            glUniform3f(loc_color, 1.0, 0.0, 0.0)
            led_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            glBindVertexArray(self.rotor_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.rotor_vbo)
            glBufferData(GL_ARRAY_BUFFER, led_data.nbytes, led_data, GL_DYNAMIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glPointSize(8.0)
            glDrawArrays(GL_POINTS, 0, 1)
            glPointSize(1.0)

        self._dibujar_rotores_animados(shader, loc_modelo, loc_color, loc_tipo)
        glUniformMatrix4fv(loc_modelo, 1, GL_TRUE, np.eye(4, dtype=np.float32))
