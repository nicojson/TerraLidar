"""
TerraLidar - Simulador de Escaneo Agrícola 3D
Versión basada en análisis de cultivos con LiDAR
"""

import os
import glfw
from OpenGL.GL import *
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import math
import time

from core.math3d import M3D
from core.camera import CamaraLibre
from simulation.drone_wireframe import DronWireframe
from simulation.visualizador_lidar import VisualizadorLidar
from simulation.analizador import AnalizadorAgricola
from simulation.escenario import Escenario3DAgricola
from simulation.procesador_nube import ProcesadorNubePuntos
from sensors.lidar import SensorLidar

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

        self.proyeccion = M3D.matriz_perspectiva(45.0, 1400 / 900, 0.1, 800.0)

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

                    # 2. Barrido láser rotatorio
                    barrido_hits, origins, dirs = self.sensor.emitir_barrido_rotatorio(self.pos_dron, delta_time)
                    self.visualizador.actualizar_barrido(barrido_hits, origins, dirs)

                    # 3. Dibujar algunos rayos aleatorios
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

            if self.escaneando:
                self.visualizador.dibujar_barrido_rotatorio(self.pos_dron)

            self.render_texto()

            glfw.swap_buffers(self.ventana)
            glfw.poll_events()

        glfw.terminate()

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