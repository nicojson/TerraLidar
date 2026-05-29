import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import math
import random
import ctypes
import os
import time
import csv
from scipy.ndimage import grey_opening
from scipy.interpolate import griddata

# ==========================================
# 1. MATEMÁTICAS Y CÁMARA
# ==========================================
class Matematicas3D:
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

    def obtener_matriz_vista(self):
        return Matematicas3D.matriz_vista(self.posicion, self.posicion + self.frente, self.arriba)

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


# ==========================================
# 2. MODELOS VISUALES
# ==========================================
class Escenario3DAgricola:
    def __init__(self, tipo_terreno):
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


class DronWireframe:
    def __init__(self):
        s = 2.5
        h = 0.5
        self.vertices = np.array([
            -s, h, -s, s, h, -s, s, h, -s, s, h, s,
            s, h, s, -s, h, s, -s, h, s, -s, h, -s,
            -s, -h, -s, s, -h, -s, s, -h, -s, s, -h, s,
            s, -h, s, -s, -h, s, -s, -h, s, -s, -h, -s,
            -s, h, -s, -s, -h, -s, s, h, -s, s, -h, -s,
            s, h, s, s, -h, s, -s, h, s, -s, -h, s,
            -s, h, -s, s, -h, s, -s, -h, -s, s, h, s
        ], dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

    def dibujar(self, pos, loc_modelo, loc_color, loc_tipo):
        matriz = np.eye(4, dtype=np.float32)
        matriz[3, 0:3] = pos
        glUniformMatrix4fv(loc_modelo, 1, GL_TRUE, matriz)
        glUniform3f(loc_color, 1.0, 0.8, 0.0)
        glUniform1i(loc_tipo, 1)
        glBindVertexArray(self.vao)
        glLineWidth(4.0)
        glDrawArrays(GL_LINES, 0, len(self.vertices) // 3)
        glLineWidth(1.0)


# ==========================================
# 3. RAYCASTER POR COMPUTE SHADER (GPU)
# ==========================================
class RaycasterCompute:
    def __init__(self, triangulos):
        self.num_triangulos = len(triangulos)
        tri_data = np.zeros((self.num_triangulos, 9), dtype=np.float32)
        for i, (v0, v1, v2) in enumerate(triangulos):
            tri_data[i, 0:3] = v0
            tri_data[i, 3:6] = v1
            tri_data[i, 6:9] = v2

        self.tri_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.tri_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, tri_data.nbytes, tri_data, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.tri_ssbo)

        with open("raytracing.comp", "w") as f:
            f.write("""
#version 430
layout(local_size_x = 256) in;

struct Ray {
    vec4 origin;
    vec4 direction;
};

layout(std430, binding = 0) buffer Tris {
    float triangle_data[];
};

layout(std430, binding = 1) buffer RaysIn {
    Ray rays[];
};

struct HitResult {
    vec3 position;
    float hit;
};

layout(std430, binding = 2) buffer HitsOut {
    HitResult hits[];
};

uniform int num_triangles;

bool rayTriangleIntersect(vec3 orig, vec3 dir,
                          vec3 v0, vec3 v1, vec3 v2,
                          out float t, out vec3 hitPoint) {
    const float EPS = 1e-6;
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(dir, edge2);
    float a = dot(edge1, h);
    if (abs(a) < EPS) return false;
    float f = 1.0 / a;
    vec3 s = orig - v0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return false;
    vec3 q = cross(s, edge1);
    float v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) return false;
    t = f * dot(edge2, q);
    if (t > EPS) {
        hitPoint = orig + dir * t;
        return true;
    }
    return false;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rays.length()) return;

    vec3 orig = rays[idx].origin.xyz;
    vec3 dir = rays[idx].direction.xyz;

    float bestT = 1e30;
    vec3 bestHit = vec3(0.0);
    bool found = false;

    for (int i = 0; i < num_triangles; i++) {
        int off = i * 9;
        vec3 v0 = vec3(triangle_data[off], triangle_data[off+1], triangle_data[off+2]);
        vec3 v1 = vec3(triangle_data[off+3], triangle_data[off+4], triangle_data[off+5]);
        vec3 v2 = vec3(triangle_data[off+6], triangle_data[off+7], triangle_data[off+8]);

        float t;
        vec3 pt;
        if (rayTriangleIntersect(orig, dir, v0, v1, v2, t, pt)) {
            if (t < bestT) {
                bestT = t;
                bestHit = pt;
                found = true;
            }
        }
    }

    if (found) {
        hits[idx].position = bestHit;
        hits[idx].hit = 1.0;
    } else {
        hits[idx].hit = 0.0;
    }
}
""")
        with open("raytracing.comp", "r") as f:
            cs = f.read()
        self.compute_shader = compileShader(cs, GL_COMPUTE_SHADER)
        self.program = compileProgram(self.compute_shader)
        glDeleteShader(self.compute_shader)
        os.remove("raytracing.comp")

        self.ray_ssbo = None
        self.hit_ssbo = None
        self.max_rays = 0

    def trace_batch(self, origins, directions):
        N = len(origins)
        if N == 0:
            return []
        if N > self.max_rays:
            if self.ray_ssbo:
                glDeleteBuffers(1, [self.ray_ssbo])
                glDeleteBuffers(1, [self.hit_ssbo])
            self.max_rays = N
            self.ray_ssbo = glGenBuffers(1)
            self.hit_ssbo = glGenBuffers(1)

        ray_data = np.zeros((N, 8), dtype=np.float32)
        ray_data[:, 0:3] = origins
        ray_data[:, 4:7] = directions

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ray_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, ray_data.nbytes, ray_data, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ray_ssbo)

        hit_data = np.zeros((N, 4), dtype=np.float32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.hit_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, hit_data.nbytes, hit_data, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.hit_ssbo)

        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, "num_triangles"), self.num_triangulos)
        glDispatchCompute((N + 255) // 256, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.hit_ssbo)
        result = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, hit_data.nbytes), dtype=np.float32)
        result = result.reshape(-1, 4)
        hits = []
        for r in result:
            if r[3] > 0.5:
                hits.append(r[:3])
        return hits


# ==========================================
# 4. SENSOR LIDAR (datos aleatorios + barrido rotatorio)
# ==========================================
class SensorLidar:
    def __init__(self, densidad, escenario):
        self.densidad = densidad
        self.raycaster = RaycasterCompute(escenario.triangulos)
        self.angulo_rotacion = 0.0           # ángulo base del abanico
        self.velocidad_rotacion = 120.0      # grados por segundo (ajustable)

    def emitir_rayos_aleatorios(self, pos_dron):
        """Genera los rayos aleatorios para la nube densa."""
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
        """Genera un abanico de 72 rayos que gira progresivamente."""
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
            # Rayo inclinado hacia abajo con un ángulo fijo (ej. 15° desde la vertical)
            inclinacion = math.radians(15.0)  # respecto al plano horizontal
            dx = math.cos(angulo) * math.cos(inclinacion)
            dy = -math.sin(inclinacion)
            dz = math.sin(angulo) * math.cos(inclinacion)
            directions[i] = (dx, dy, dz)

        return self.raycaster.trace_batch(origins, directions), origins, directions


# ==========================================
# 5. VISUALIZADOR LIDAR (puntos + rayos activos + barrido rotatorio)
# ==========================================
class VisualizadorLidar:
    MAX_PUNTOS_VISIBLES = 30000

    def __init__(self):
        self.puntos = []
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.capacidad_buffer = 0
        # Para el barrido rotatorio guardamos los puntos de la última pasada
        self.barrido_hits = []
        self.barrido_origins = []
        self.barrido_directions = []

    def actualizar(self, nuevos):
        if nuevos:
            self.puntos.extend(nuevos)
            if len(self.puntos) > self.MAX_PUNTOS_VISIBLES:
                self.puntos = self.puntos[-self.MAX_PUNTOS_VISIBLES:]

    def actualizar_barrido(self, hits, origins, directions):
        """Recibe los resultados del barrido rotatorio."""
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
        """Muestra hasta 100 rayos aleatorios (rojos)."""
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
        """Dibuja el abanico completo del LIDAR rotatorio (color cian)."""
        if not self.barrido_hits:
            return
        glUseProgram(0)
        glLineWidth(2.0)
        glColor3f(0.0, 1.0, 1.0)  # cian brillante
        glBegin(GL_LINES)
        ox, oy, oz = pos_dron[0], pos_dron[1] - 0.5, pos_dron[2]
        for i, hit in enumerate(self.barrido_hits):
            glVertex3f(ox, oy, oz)
            glVertex3f(hit[0], hit[1], hit[2])
        glEnd()
        glLineWidth(1.0)


# ==========================================
# 6. PROCESADOR PMF Y CSV (sin cambios)
# ==========================================
class ProcesadorNubePuntos:
    def __init__(self, resolucion_malla=0.5):
        self.resolucion = resolucion_malla
        self.dtm = None
        self.dsm = None
        self.chm = None

    def filtrar_pmf(self, puntos, tam_ventana_inicial=1.0, tam_ventana_final=10.0, umbral_altura=0.5):
        if len(puntos) == 0:
            return [], []

        pts = np.array(puntos)
        x = pts[:, 0]
        z = pts[:, 2]
        y = pts[:, 1]

        x_min, x_max = np.min(x), np.max(x)
        z_min, z_max = np.min(z), np.max(z)
        xi = np.arange(x_min, x_max, self.resolucion)
        zi = np.arange(z_min, z_max, self.resolucion)
        xi, zi = np.meshgrid(xi, zi)

        dsm_interp = griddata((x, z), y, (xi, zi), method='linear', fill_value=np.nan)

        suelo_mask = np.isfinite(dsm_interp)
        for ventana in np.arange(tam_ventana_inicial, tam_ventana_final, 0.5):
            size = max(1, int(ventana / self.resolucion))
            opened = grey_opening(dsm_interp, size=size)
            diff = dsm_interp - opened
            suelo_mask = suelo_mask & (diff <= umbral_altura)

        puntos_suelo = []
        puntos_vegetacion = []
        for i, (xx, zz) in enumerate(zip(x, z)):
            ix = int((xx - x_min) / self.resolucion)
            iz = int((zz - z_min) / self.resolucion)
            if 0 <= ix < dsm_interp.shape[1] and 0 <= iz < dsm_interp.shape[0]:
                if suelo_mask[iz, ix]:
                    puntos_suelo.append(pts[i])
                else:
                    puntos_vegetacion.append(pts[i])
            else:
                puntos_vegetacion.append(pts[i])

        if len(puntos_suelo) > 0:
            pts_suelo = np.array(puntos_suelo)
            self.dtm = griddata((pts_suelo[:, 0], pts_suelo[:, 2]), pts_suelo[:, 1],
                                (xi, zi), method='linear', fill_value=np.nan)
        else:
            self.dtm = np.full_like(dsm_interp, np.nan)

        if len(puntos_vegetacion) > 0:
            pts_veg = np.array(puntos_vegetacion)
            self.dsm = griddata((pts_veg[:, 0], pts_veg[:, 2]), pts_veg[:, 1],
                                (xi, zi), method='linear', fill_value=np.nan)
        else:
            self.dsm = dsm_interp.copy()

        self.chm = self.dsm - self.dtm
        self.chm[self.chm < 0] = 0

        return puntos_suelo, puntos_vegetacion

    def calcular_metricas_cultivo(self, altura_min_planta=0.2):
        if self.chm is None:
            return {}
        mask_vegetacion = self.chm > altura_min_planta
        altura_media = np.nanmean(self.chm[mask_vegetacion]) if np.any(mask_vegetacion) else 0.0
        altura_maxima = np.nanmax(self.chm) if np.any(np.isfinite(self.chm)) else 0.0
        resolucion_area = self.resolucion ** 2
        volumen_biomasa = np.nansum(self.chm[mask_vegetacion]) * resolucion_area
        rendimiento_kg_ha = volumen_biomasa * 200 * 10000
        return {
            'altura_media_dosel_m': round(altura_media, 2),
            'altura_maxima_m': round(altura_maxima, 2),
            'volumen_biomasa_m3': round(volumen_biomasa, 2),
            'rendimiento_estimado_kg_ha': round(rendimiento_kg_ha, 0)
        }

    def exportar_a_csv(self, puntos_suelo, puntos_vegetacion, nombre_archivo="resultados_terralidar.csv"):
        with open(nombre_archivo, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Tipo', 'X', 'Y', 'Z'])
            for p in puntos_suelo:
                writer.writerow(['Suelo', p[0], p[1], p[2]])
            for p in puntos_vegetacion:
                writer.writerow(['Vegetacion', p[0], p[1], p[2]])
            metricas = self.calcular_metricas_cultivo()
            writer.writerow([])
            writer.writerow(['Metrica', 'Valor'])
            for k, v in metricas.items():
                writer.writerow([k, v])
        print(f"✅ Datos exportados a {nombre_archivo}")


# ==========================================
# 7. ANÁLISIS LEGACY (sin cambios)
# ==========================================
class AnalizadorAgricola:
    def __init__(self, config):
        self.tipo_analisis = config['analisis']
        self.densidad_laser = config['densidad']

        self.cultivos = {
            'CEBOLLA (Bulbos)': {'h_min': 0.0, 'h_max': 0.6, 'rend_kg': 45000},
            'JITOMATE (Hortaliza)': {'h_min': 0.2, 'h_max': 1.2, 'rend_kg': 80000},
            'MAIZ (Grano)': {'h_min': 0.8, 'h_max': 2.5, 'rend_kg': 12000},
            'TRIGO (Cereal)': {'h_min': 0.4, 'h_max': 1.0, 'rend_kg': 6000}
        }

        self.pastos = {
            'ALFALFA (Ganado Bovino)': {'h_min': 0.3, 'h_max': 0.9, 'rend_kg': 25000},
            'BERMUDA (Ganado Equino)': {'h_min': 0.1, 'h_max': 0.4, 'rend_kg': 15000},
            'AVENA FORRAJERA (Ovinos)': {'h_min': 0.4, 'h_max': 1.1, 'rend_kg': 18000},
            'SORGO FORRAJERO (Porcinos)': {'h_min': 0.6, 'h_max': 1.8, 'rend_kg': 22000}
        }

    def calcular_aptitud(self, puntos_totales):
        if len(puntos_totales) == 0:
            return {}
        p_arr = np.array(puntos_totales)
        altura_promedio = np.std(p_arr[:, 1])
        resultados = {}
        catalogo = self.cultivos if self.tipo_analisis == 1 else self.pastos
        area_estimada = 14400.0
        margen_error = (500 - self.densidad_laser) / 500.0
        for nombre, p in catalogo.items():
            desviacion = abs(altura_promedio - ((p['h_max'] + p['h_min']) / 2.0))
            aptitud_real = max(0.0, 100.0 - (desviacion * 40.0))
            if margen_error > 0:
                ruido = random.uniform(-margen_error * 20.0, margen_error * 20.0)
                aptitud_final = np.clip(aptitud_real + ruido, 0.0, 100.0)
            else:
                aptitud_final = aptitud_real
            rendimiento = int((aptitud_final / 100.0) * p['rend_kg'] * (area_estimada / 10000.0))
            confianza = 100 - int(margen_error * 80)
            resultados[nombre] = {
                'aptitud': round(aptitud_final, 1),
                'rendimiento_kg': rendimiento,
                'confianza': confianza
            }
        return dict(sorted(resultados.items(), key=lambda item: item[1]['aptitud'], reverse=True))


# ==========================================
# 8. MOTOR GRÁFICO PRINCIPAL
# ==========================================
class SimuladorTerraLidar:
    def __init__(self, config):
        self.config = config
        if not glfw.init():
            return
        glutInit()

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)

        self.ventana = glfw.create_window(1400, 900, "TerraLidar PRO V7.2 - Barrido láser rotatorio", None, None)
        glfw.make_context_current(self.ventana)
        glfw.set_input_mode(self.ventana, glfw.CURSOR, glfw.CURSOR_DISABLED)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.camara = CamaraLibre()
        self.escenario = Escenario3DAgricola(config['terreno'])
        self.sensor = SensorLidar(config['densidad'], self.escenario)
        self.visualizador = VisualizadorLidar()
        self.analizador = AnalizadorAgricola(config)
        self.procesador = ProcesadorNubePuntos(resolucion_malla=0.5)
        self.dron = DronWireframe()

        self.pos_dron = np.array([-60.0, 30.0, -40.0], dtype=np.float32)
        self.escaneando = False
        self.tiempo_inicial_escaneo = 0.0
        self.tiempo_restante = config['tiempo_limite']
        self.resultados_calculados = None
        self.metricas_pmf = None
        self.tecla_l_presionada = False
        self.ultimo_tiempo = time.time()

        glfw.set_cursor_pos_callback(self.ventana, self._mouse_cb)

        vs = compileShader("""
        #version 430
        layout (location = 0) in vec3 aPos;
        out vec3 WorldPos;
        uniform mat4 model; uniform mat4 view; uniform mat4 projection;
        void main() { 
            WorldPos = vec3(model * vec4(aPos, 1.0));
            gl_Position = projection * view * vec4(WorldPos, 1.0); 
        }
        """, GL_VERTEX_SHADER)
        fs = compileShader("""
        #version 430
        in vec3 WorldPos;
        out vec4 FragColor; 
        uniform vec3 colorBase;
        uniform int objectType;
        void main() { 
            if(objectType == 2) {
                float grid = mod(floor(WorldPos.x * 0.5) + floor(WorldPos.z * 0.5), 2.0);
                vec3 colorTierra = mix(vec3(0.42, 0.28, 0.15), vec3(0.36, 0.23, 0.11), grid);
                FragColor = vec4(colorTierra, 1.0);
            } else {
                FragColor = vec4(colorBase, 1.0); 
            }
        }
        """, GL_FRAGMENT_SHADER)
        self.shader = compileProgram(vs, fs)
        glDeleteShader(vs)
        glDeleteShader(fs)

        self.proyeccion = Matematicas3D.matriz_perspectiva(45.0, 1400 / 900, 0.1, 800.0)

    def _mouse_cb(self, w, x, y):
        if self.camara.primer_movimiento:
            self.camara.ultima_x = x
            self.camara.ultima_y = y
            self.camara.primer_movimiento = False
            return
        self.camara.rotar_con_raton(x - self.camara.ultima_x, y - self.camara.ultima_y)
        self.camara.ultima_x = x
        self.camara.ultima_y = y

    def render_texto(self):
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1400, 900, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        # Panel izquierdo (telemetría)
        glColor4f(0.05, 0.05, 0.1, 0.85)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(550, 10)
        glVertex2f(550, 150)
        glVertex2f(10, 150)
        glEnd()

        glColor3f(1.0, 1.0, 1.0)
        terrenos_nombre = {1: "Valle Plano", 2: "Ondulado/Colinas", 3: "Agreste/Montañoso"}
        status_string = f"ESCANEANDO... {self.tiempo_restante:.1f}s" if self.escaneando else "EN ESPERA (Presione L)"
        if self.resultados_calculados:
            status_string = "PROCESAMIENTO FINALIZADO"

        textos_config = [
            f"MODO: {'Agricola' if self.config['analisis'] == 1 else 'Ganadero'} | TOPO: {terrenos_nombre[self.config['terreno']]}",
            f"LASER: {self.config['densidad']}/seg | PUNTOS: {len(self.visualizador.puntos)}",
            f"ESTADO: {status_string}",
            f"COORDENADAS: X:{self.pos_dron[0]:.1f} Z:{self.pos_dron[2]:.1f}"
        ]
        for i, txt in enumerate(textos_config):
            glRasterPos2f(25, 35 + (i * 28))
            for ch in txt:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

        # Panel derecho (resultados PMF o análisis legacy)
        if self.resultados_calculados:
            glColor4f(0.02, 0.1, 0.05, 0.9)
            glBegin(GL_QUADS)
            glVertex2f(800, 10)
            glVertex2f(1390, 10)
            glVertex2f(1390, 300)
            glVertex2f(800, 300)
            glEnd()
            glColor3f(0.2, 1.0, 0.4)
            glRasterPos2f(820, 35)
            titulo = "=== REPORTE PMF (CLASIFICACION SUELO/VEGETACION) ==="
            for ch in titulo:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

            y_offset = 70
            if self.metricas_pmf:
                for key, val in self.metricas_pmf.items():
                    glColor3f(1.0, 1.0, 1.0)
                    glRasterPos2f(820, y_offset)
                    for ch in f"{key}: {val}":
                        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
                    y_offset += 30
            else:
                for cultivo, datos in self.resultados_calculados.items():
                    if y_offset > 280:
                        break
                    glColor3f(1.0, 1.0, 1.0)
                    glRasterPos2f(820, y_offset)
                    for ch in f"{cultivo}: {datos['aptitud']}%  {datos['rendimiento_kg']} kg":
                        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
                    y_offset += 25

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def ejecutar(self):
        loc_col = glGetUniformLocation(self.shader, "colorBase")
        loc_vis = glGetUniformLocation(self.shader, "view")
        loc_pro = glGetUniformLocation(self.shader, "projection")
        loc_mod = glGetUniformLocation(self.shader, "model")
        loc_tip = glGetUniformLocation(self.shader, "objectType")

        while not glfw.window_should_close(self.ventana):
            ahora = time.time()
            delta_time = ahora - self.ultimo_tiempo
            self.ultimo_tiempo = ahora

            if glfw.get_key(self.ventana, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(self.ventana, True)
            if glfw.get_key(self.ventana, glfw.KEY_W) == glfw.PRESS:
                self.camara.mover("ADELANTE")
            if glfw.get_key(self.ventana, glfw.KEY_S) == glfw.PRESS:
                self.camara.mover("ATRAS")
            if glfw.get_key(self.ventana, glfw.KEY_A) == glfw.PRESS:
                self.camara.mover("IZQUIERDA")
            if glfw.get_key(self.ventana, glfw.KEY_D) == glfw.PRESS:
                self.camara.mover("DERECHA")

            estado_l = glfw.get_key(self.ventana, glfw.KEY_L)
            if estado_l == glfw.PRESS and not self.tecla_l_presionada:
                self.tecla_l_presionada = True
                if not self.escaneando:
                    self.escaneando = True
                    self.tiempo_inicial_escaneo = time.time()
                    self.resultados_calculados = None
                    self.metricas_pmf = None
                    self.visualizador.puntos = []
                    self.pos_dron = np.array([-60.0, 30.0, -40.0], dtype=np.float32)
            elif estado_l == glfw.RELEASE:
                self.tecla_l_presionada = False

            nuevos = []
            if self.escaneando:
                tiempo_transcurrido = time.time() - self.tiempo_inicial_escaneo
                self.tiempo_restante = max(0.0, self.config['tiempo_limite'] - tiempo_transcurrido)

                if self.tiempo_restante <= 0.0:
                    self.escaneando = False
                    print("Escaneo completado. Procesando nube de puntos con PMF...")
                    suelo, vegetacion = self.procesador.filtrar_pmf(self.visualizador.puntos)
                    self.metricas_pmf = self.procesador.calcular_metricas_cultivo()
                    self.procesador.exportar_a_csv(suelo, vegetacion, "lidar_scan.csv")
                    self.resultados_calculados = self.analizador.calcular_aptitud(self.visualizador.puntos)
                else:
                    progreso = tiempo_transcurrido / self.config['tiempo_limite']
                    total_filas = 3
                    pos_lineal = progreso * total_filas
                    fila_actual = min(math.floor(pos_lineal), total_filas - 1)
                    fraccion_fila = pos_lineal - fila_actual
                    z_base = -40.0 + (fila_actual * 40.0)
                    if fila_actual % 2 == 0:
                        x_base = -60.0 + (fraccion_fila * 120.0)
                    else:
                        x_base = 60.0 - (fraccion_fila * 120.0)
                    y_terreno_base = self.escenario.calcular_altura(x_base, z_base)
                    self.pos_dron = np.array([x_base, y_terreno_base + 30.0, z_base], dtype=np.float32)

                    # 1. Rayos aleatorios para la nube densa
                    nuevos = self.sensor.emitir_rayos_aleatorios(self.pos_dron)
                    self.visualizador.actualizar(nuevos)

                    # 2. Barrido láser rotatorio (efecto visual)
                    barrido_hits, origins, dirs = self.sensor.emitir_barrido_rotatorio(self.pos_dron, delta_time)
                    self.visualizador.actualizar_barrido(barrido_hits, origins, dirs)

                    # 3. Dibujar algunos rayos aleatorios (opcional)
                    self.visualizador.dibujar_rayos_activos(self.pos_dron, nuevos)
            else:
                y_terreno_base = self.escenario.calcular_altura(self.pos_dron[0], self.pos_dron[2])
                self.pos_dron[1] = y_terreno_base + 30.0 + math.sin(time.time() * 2) * 0.5

            glClearColor(0.45, 0.68, 0.88, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUseProgram(self.shader)
            glUniformMatrix4fv(loc_vis, 1, GL_TRUE, self.camara.obtener_matriz_vista())
            glUniformMatrix4fv(loc_pro, 1, GL_TRUE, self.proyeccion)
            glUniformMatrix4fv(loc_mod, 1, GL_TRUE, np.eye(4, dtype=np.float32))

            self.escenario.dibujar(loc_tip)
            self.visualizador.dibujar_puntos(loc_tip, loc_col)
            self.dron.dibujar(self.pos_dron, loc_mod, loc_col, loc_tip)

            # Dibujar el barrido rotatorio (incluso cuando no está escaneando se puede ver)
            if self.escaneando:
                self.visualizador.dibujar_barrido_rotatorio(self.pos_dron)
            else:
                # Durante la espera mostramos un barrido estático (si se desea)
                pass

            self.render_texto()

            glfw.swap_buffers(self.ventana)
            glfw.poll_events()

        glfw.terminate()


# ==========================================
# MENÚ DE TERMINAL
# ==========================================
def menu():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 50)
    print("   TERRALIDAR PRO V7.2 - BARRIDO LÁSER ROTATORIO")
    print("   (Animación de rayos giratorios + nube densa)")
    print("=" * 50)

    print("\n[1] DENSIDAD DEL SENSOR (Rayos/seg)")
    print("  1. Ultra (10 000 rayos/seg) -> 8 seg escaneo")
    print("  2. Alta (5 000 rayos/seg) -> 12 seg")
    print("  3. Media (1 000 rayos/seg) -> 20 seg")
    print("  4. Baja (300 rayos/seg) -> 30 seg")
    opc_d = input("Opcion (1/2/3/4): ")

    if opc_d == '1':
        d, t_limite = 10000, 8.0
    elif opc_d == '2':
        d, t_limite = 5000, 12.0
    elif opc_d == '4':
        d, t_limite = 300, 30.0
    else:
        d, t_limite = 1000, 20.0

    print("\n[2] TOPOGRAFÍA FÍSICA DE LA SUPERFICIE")
    print("  1. Valle Plano")
    print("  2. Ondulado (Colinas Suaves)")
    print("  3. Agreste / Montañoso")
    opc_t = input("Opcion (1/2/3): ")
    t = 3 if opc_t == '3' else (2 if opc_t == '2' else 1)

    print("\n[3] SECTOR DE ANÁLISIS EN PANTALLA")
    print("  1. Agrícola (Cebolla, Jitomate, Maíz, Trigo)")
    print("  2. Ganadero (Alfalfa, Bermuda, Avena, Sorgo)")
    opc_a = input("Opcion (1/2): ")
    a = 2 if opc_a == '2' else 1

    return {'densidad': d, 'tiempo_limite': t_limite, 'terreno': t, 'analisis': a}


if __name__ == "__main__":
    config = menu()
    SimuladorTerraLidar(config).ejecutar()