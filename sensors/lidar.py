import math
import random
import numpy as np
from OpenGL.GL import *
from simulation.terrain import altura_terreno

# ==============================================================================
# COMPUTE SHADER PARA RAYMARCHING EN GPU
# ==============================================================================
COMPUTE_SHADER_SRC = """
#version 430 core

layout(local_size_x = 256) in;

// Estructura de un rayo (input)
struct Ray {
    vec3 dir;
};

// Estructura de un punto de impacto (output)
struct Hit {
    vec3 pos;
    float hit; // 1.0 si impactó, 0.0 si no
};

layout(std430, binding = 0) buffer RayBuffer {
    Ray rays[];
};

layout(std430, binding = 1) buffer HitBuffer {
    Hit hits[];
};

uniform vec3 pos_dron;
uniform float max_rango;

// --- Simulación de ruido Perlin en GPU (simplificado) ---
// Función hash pseudo-aleatoria
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Interpolación suave
float fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Gradiente para ruido
float grad(float hash_val, float x, float y) {
    int h = int(hash_val * 4.0) & 3;
    float u = h < 2 ? x : y;
    float v = h < 2 ? y : x;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// Ruido básico
float noise(vec2 p) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);
    
    float u = fade(pf.x);
    float v = fade(pf.y);
    
    float a = hash(pi);
    float b = hash(pi + vec2(1.0, 0.0));
    float c = hash(pi + vec2(0.0, 1.0));
    float d = hash(pi + vec2(1.0, 1.0));
    
    float res = mix(
        mix(grad(a, pf.x, pf.y), grad(b, pf.x - 1.0, pf.y), u),
        mix(grad(c, pf.x, pf.y - 1.0), grad(d, pf.x - 1.0, pf.y - 1.0), u),
        v
    );
    return res;
}

// Fractional Brownian Motion (1 octava para velocidad extrema)
float fbm(vec2 p) {
    return noise(p) * 0.5;
}

// Equivalente a altura_terreno en Python
float alturaTerreno(float x, float z) {
    float surco = 0.12 * sin(x * 3.1415);
    float n = fbm(vec2(x * 0.15, z * 0.15)) * 0.6;
    float crop = 0.18 * max(0.0, sin(x * 3.1415));
    return surco + n + crop;
}

// Generador de números pseudoaleatorios para el ruido gaussiano
float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}
float gauss(vec2 co) {
    float u1 = max(0.0001, rand(co));
    float u2 = max(0.0001, rand(co * 1.5));
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    vec3 dir = rays[idx].dir;
    
    // Si la dirección es (0,0,0), ignoramos
    if(length(dir) < 0.1) {
        hits[idx].hit = 0.0;
        return;
    }

    float t = 0.5;
    float dt_step = 2.0; // Paso grande
    int max_iters = int(max_rango / dt_step);
    
    bool impacto = false;
    vec3 p_final;
    
    for(int i = 0; i < max_iters; i++) {
        vec3 p = pos_dron + dir * t;
        float y_terr = alturaTerreno(p.x, p.z);
        
        if (p.y <= y_terr) {
            // Bisección rápida
            float t_lo = t - dt_step;
            float t_hi = t;
            
            float t_mid = (t_lo + t_hi) * 0.5;
            vec3 pm = pos_dron + dir * t_mid;
            if(pm.y <= alturaTerreno(pm.x, pm.z)) t_hi = t_mid; else t_lo = t_mid;
            
            p_final = pos_dron + dir * t_hi;
            
            // Añadir ruido simulado
            float ruido = gauss(p_final.xz + pos_dron.xz) * 0.02; // sigma = 0.02
            p_final += dir * ruido;
            
            float final_terr = alturaTerreno(p_final.x, p_final.z);
            if(p_final.y < final_terr - 0.1) {
                p_final.y = final_terr - 0.1;
            }
            
            if(distance(p_final, pos_dron) <= max_rango) {
                impacto = true;
            }
            break;
        }
        t += dt_step;
    }
    
    if(impacto) {
        hits[idx].pos = p_final;
        hits[idx].hit = 1.0;
    } else {
        hits[idx].hit = 0.0;
    }
}
"""

def compilar_compute_shader(src):
    shader = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    
    prog = glCreateProgram()
    glAttachShader(prog, shader)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    
    glDeleteShader(shader)
    return prog

class SensorLidarReal:
    def __init__(self, num_canales=32, fov_v_deg=45.0, max_rango=80.0): 
        self.num_canales = num_canales
        self.max_rango = max_rango
        self.azimut = 0.0
        
        # ¡Ahora podemos volver a subir las RPM y los pasos a niveles locos!
        self.rpm = 12000
        self.ruido_sigma = 0.02
        
        self.elevaciones = [
            math.radians(-fov_v_deg / 2 + fov_v_deg * i / (num_canales - 1))
            for i in range(num_canales)
        ]
        self.cos_el = np.cos(self.elevaciones)
        self.sin_el = np.sin(self.elevaciones)

        # --- Setup OpenGL Compute Shader ---
        # Nota: El contexto OpenGL ya debe estar activo al instanciar esto.
        try:
            self.compute_prog = compilar_compute_shader(COMPUTE_SHADER_SRC)
            self.loc_pos_dron = glGetUniformLocation(self.compute_prog, "pos_dron")
            self.loc_max_rango = glGetUniformLocation(self.compute_prog, "max_rango")
            
            # Reservar buffers (ssbo) en GPU
            self.max_rayos_por_batch = 100000 # Podemos procesar 100k rayos de golpe sin despeinarnos
            
            # Buffer de rayos (input) - 3 floats por rayo (vec3 dir) + padding (vec4 align) = 16 bytes
            self.ssbo_rays = glGenBuffers(1)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_rays)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.max_rayos_por_batch * 16, None, GL_DYNAMIC_DRAW)
            
            # Buffer de impactos (output) - vec3 pos (12) + float hit (4) = 16 bytes
            self.ssbo_hits = glGenBuffers(1)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_hits)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.max_rayos_por_batch * 16, None, GL_DYNAMIC_READ)
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            self.gpu_accelerated = True
            print("[✓] Sensor LiDAR acelerado por GPU (Compute Shaders) inicializado.")
        except Exception as e:
            print(f"[!] Error inicializando Compute Shader: {e}")
            print("[!] Cayendo de vuelta a cálculo en CPU...")
            self.gpu_accelerated = False

    def emitir(self, pos_dron, dt):
        if not self.gpu_accelerated:
            return self._emitir_cpu(pos_dron, dt)
            
        puntos = []
        grados_frame = (self.rpm / 60.0) * 360.0 * dt
        
        # Podemos calcular miles de pasos por frame en GPU
        pasos = min(2500, max(1, int(grados_frame / 2.0)))
        d_az = grados_frame / pasos if pasos > 0 else 0
        
        total_rayos = pasos * self.num_canales
        if total_rayos > self.max_rayos_por_batch:
            pasos = self.max_rayos_por_batch // self.num_canales
            total_rayos = pasos * self.num_canales
            
        # 1. Preparar las direcciones de los rayos en CPU (rápido con numpy)
        # Array de vec4 para alineación std430 (x, y, z, padding)
        ray_dirs = np.zeros((total_rayos, 4), dtype=np.float32)
        
        idx = 0
        az_actual = self.azimut
        for _ in range(pasos):
            az_actual = (az_actual + d_az) % 360.0
            az_rad = math.radians(az_actual)
            cos_az = math.cos(az_rad)
            sin_az = math.sin(az_rad)
            
            for i in range(self.num_canales):
                dy = self.sin_el[i]
                if dy < 0.05: # Solo rayos hacia abajo
                    ray_dirs[idx, 0] = self.cos_el[i] * cos_az
                    ray_dirs[idx, 1] = dy
                    ray_dirs[idx, 2] = self.cos_el[i] * sin_az
                idx += 1
                
        self.azimut = az_actual
        
        # 2. Subir direcciones a la GPU
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_rays)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ray_dirs.nbytes, ray_dirs)
        
        # 3. Ejecutar Compute Shader
        glUseProgram(self.compute_prog)
        glUniform3f(self.loc_pos_dron, pos_dron[0], pos_dron[1], pos_dron[2])
        glUniform1f(self.loc_max_rango, self.max_rango)
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo_rays)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo_hits)
        
        # Lanzar hilos (work groups). Dividimos por 256 que es nuestro local_size_x
        num_grupos = int(math.ceil(total_rayos / 256.0))
        glDispatchCompute(num_grupos, 1, 1)
        
        # Esperar a que terminen
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        
        # 4. Leer resultados
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_hits)
        # Obtenemos los datos crudos (array de floats)
        # La estructura Hit es {vec3 pos, float hit}, es decir 4 floats seguidos por rayo.
        datos_crudos = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, total_rayos * 16)
        
        # Convertimos bytes a numpy array
        hits_np = np.frombuffer(datos_crudos, dtype=np.float32).reshape(-1, 4)
        
        # Filtramos los que sí impactaron (hit == 1.0)
        # hits_np[:, 3] es el campo 'hit'
        impactos = hits_np[hits_np[:, 3] > 0.5]
        
        # Devolvemos solo xyz
        return impactos[:, 0:3].tolist()

    def _emitir_cpu(self, pos_dron, dt):
        puntos = []
        grados_frame = (self.rpm / 60.0) * 360.0 * dt
        
        pasos = min(150, max(1, int(grados_frame / 2.0)))
        d_az = grados_frame / pasos if pasos > 0 else 0

        for _ in range(pasos):
            self.azimut = (self.azimut + d_az) % 360.0
            az_rad = math.radians(self.azimut)
            
            cos_az = math.cos(az_rad)
            sin_az = math.sin(az_rad)

            for i in range(self.num_canales):
                dy = self.sin_el[i]
                if dy >= 0.05:
                    continue

                dx = self.cos_el[i] * cos_az
                dz = self.cos_el[i] * sin_az
                
                dir_rayo_x = dx
                dir_rayo_y = dy
                dir_rayo_z = dz

                t = 0.5
                dt_step = 4.0 
                
                max_iters = int(self.max_rango / dt_step)
                
                for _ in range(max_iters):
                    px = pos_dron[0] + dir_rayo_x * t
                    pz = pos_dron[2] + dir_rayo_z * t
                    py = pos_dron[1] + dir_rayo_y * t
                    
                    y_terreno = altura_terreno(px, pz)
                    
                    if py <= y_terreno:
                        t_lo, t_hi = t - dt_step, t
                        
                        t_mid = (t_lo + t_hi) * 0.5
                        pm_x = pos_dron[0] + dir_rayo_x * t_mid
                        pm_z = pos_dron[2] + dir_rayo_z * t_mid
                        pm_y = pos_dron[1] + dir_rayo_y * t_mid
                        
                        if pm_y <= altura_terreno(pm_x, pm_z):
                            t_hi = t_mid
                        else:
                            t_lo = t_mid
                        
                        px = pos_dron[0] + dir_rayo_x * t_hi
                        pz = pos_dron[2] + dir_rayo_z * t_hi
                        py = pos_dron[1] + dir_rayo_y * t_hi
                        
                        ruido = random.gauss(0, self.ruido_sigma)
                        px += dir_rayo_x * ruido
                        py += dir_rayo_y * ruido
                        pz += dir_rayo_z * ruido
                        
                        alt_terr = altura_terreno(px, pz)
                        if py < alt_terr - 0.1:
                            py = alt_terr - 0.1
                        
                        dist_sq = (px - pos_dron[0])**2 + (py - pos_dron[1])**2 + (pz - pos_dron[2])**2
                        if dist_sq <= self.max_rango**2:
                            puntos.append([px, py, pz])
                        break

                    t += dt_step

        return puntos
