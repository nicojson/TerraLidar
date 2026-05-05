"""
TerraLidar - Simulador de Escaneo Agrícola 3D
Versión basada en análisis de cultivos con LiDAR
"""

import glfw
from OpenGL.GL import *
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
import numpy as np
import math

from core.math3d import M3D
from core.camera import CamaraLibre
from core.window import init_window
from simulation.drone_wireframe import DronWireframe
from simulation.visualizador_lidar import VisualizadorLidar
from simulation.analizador import AnalizadorAgricola
from simulation.escenario import Escenario3DAgricola
from sensors.lidar_simple import SensorLidar
from graphics.renderizador_texto import RenderizadorTexto


class SimuladorTerraLidar:
    def __init__(self):
        if not glfw.init():
            return

        glutInit()

        self.ventana = glfw.create_window(1400, 900, "TerraLidar - Escaneo Agricola 3D", None, None)
        glfw.make_context_current(self.ventana)
        glfw.set_input_mode(self.ventana, glfw.CURSOR, glfw.CURSOR_DISABLED)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.camara = CamaraLibre(self.ventana)
        self.dron = DronWireframe()
        self.sensor = SensorLidar()
        self.escenario = Escenario3DAgricola()
        self.visualizador = VisualizadorLidar()
        self.analizador = AnalizadorAgricola()
        self.renderizador_texto = RenderizadorTexto()

        self.escaneando = False
        self.tecla_l_presionada = False

        self.ancho_ventana = 1400
        self.alto_ventana = 900

        glfw.set_cursor_pos_callback(self.ventana, self.callback_movimiento_raton)
        glfw.set_scroll_callback(self.ventana, self.callback_scroll)

        VERTEX_SHADER = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        out vec3 WorldPos;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() { 
            WorldPos = vec3(model * vec4(aPos, 1.0));
            gl_Position = projection * view * vec4(WorldPos, 1.0); 
        }
        """
        FRAGMENT_SHADER = """
        #version 330 core
        in vec3 WorldPos;
        out vec4 FragColor;

        uniform vec3 colorBase;
        uniform int objectType;

        void main() {
            if (objectType == 2) {
                float surcos = mod(WorldPos.x, 1.8);
                float cultivos = sin(WorldPos.z * surcos * 2.0);

                vec3 brownDirt = vec3(0.4, 0.25, 0.1);
                vec3 greenCrop = vec3(0.1, 0.4, 0.1);

                vec3 finalColor = brownDirt;
                if(surcos < 1.1){ finalColor = greenCrop; }

                FragColor = vec4(finalColor, 1.0);
            } else if (objectType == 1) {
                FragColor = vec4(colorBase, 0.6); 
            } else {
                FragColor = vec4(colorBase, 1.0);
            }
        }
        """

        vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vs, VERTEX_SHADER)
        glCompileShader(vs)
        fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fs, FRAGMENT_SHADER)
        glCompileShader(fs)
        self.shader = glCreateProgram()
        glAttachShader(self.shader, vs)
        glAttachShader(self.shader, fs)
        glLinkProgram(self.shader)

        self.proyeccion = M3D.perspectiva(45.0, self.ancho_ventana / self.alto_ventana, 0.1, 300.0)

        print("╔══════════════════════════════════════════╗")
        print("║  TerraLidar - Escaneo Agrícola 3D        ║")
        print("╠══════════════════════════════════════════╣")
        print("║  LIDAR ACTIVADO - Presiona L para escanear║")
        print("║  W/A/S/D: Mover | ESPACIO/CTRL: Arriba  ║")
        print("║  Raton: Rotar | Rueda: Zoom             ║")
        print("║  ESC: Salir                              ║")
        print("╚══════════════════════════════════════════╝")

    def callback_movimiento_raton(self, window, xpos, ypos):
        if self.camara.primer_movimiento:
            self.camara.ultima_x = xpos
            self.camara.ultima_y = ypos
            self.camara.primer_movimiento = False
            return

        delta_x = xpos - self.camara.ultima_x
        delta_y = ypos - self.camara.ultima_y

        self.camara.ultima_x = xpos
        self.camara.ultima_y = ypos

        self.camara.rotar_con_raton(delta_x, delta_y)

    def callback_scroll(self, window, xoffset, yoffset):
        self.camara.zoom(-yoffset * 0.1)

    def procesar_inputs(self):
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
        if glfw.get_key(self.ventana, glfw.KEY_SPACE) == glfw.PRESS:
            self.camara.mover("ARRIBA")
        if glfw.get_key(self.ventana, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            self.camara.mover("ABAJO")

        estado_l = glfw.get_key(self.ventana, glfw.KEY_L)
        if estado_l == glfw.PRESS and not self.tecla_l_presionada:
            self.escaneando = not self.escaneando
            self.tecla_l_presionada = True
            print("DRON: " + ("ESCANEANDO" if self.escaneando else "PAUSADO"))
        elif estado_l == glfw.RELEASE:
            self.tecla_l_presionada = False

    def render_hud(self, resultados_analisis, altura, regularidad, densidad):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.ancho_ventana, self.alto_ventana, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)

        offset_y = 20
        linea_alto = 20

        glColor3f(0.95, 0.95, 0.95)
        glRasterPos2f(20, offset_y)

        info_texto = f"LIDAR: {len(self.visualizador.puntos)} puntos | Altura: {altura:.2f}m | Regularidad: {regularidad:.3f} | Densidad: {densidad:.2f}"
        for char in info_texto:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

        if resultados_analisis:
            offset_y += linea_alto * 2
            glColor3f(1.0, 1.0, 0.8)
            titulo = "=== ANALISIS DE CULTIVOS ==="
            glRasterPos2f(20, offset_y)
            for char in titulo:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

            offset_y += linea_alto
            cultivos_ordenados = sorted(resultados_analisis.items(), key=lambda x: x[1]['aptitud'], reverse=True)

            for cultivo_nombre, datos in cultivos_ordenados[:3]:
                glColor3f(0.95, 0.95, 0.95) if datos['aptitud'] > 70 else glColor3f(0.95, 0.95, 0.95)

                texto = f"{cultivo_nombre}: {datos['aptitud']}% | {datos['rendimiento_kg']}kg | {datos['dias_cosecha']} dias"
                glRasterPos2f(20, offset_y)
                for char in texto:
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
                offset_y += linea_alto

        glColor3f(0.7, 0.7, 0.7)
        ayuda = "W/A/S/D: Mover | ESPACIO/CTRL: Arriba/Abajo | Raton: Rotar | Rueda: Zoom | L: Escanear"
        glRasterPos2f(20, self.alto_ventana - 20)
        for char in ayuda:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def ejecutar(self):
        loc_color = glGetUniformLocation(self.shader, "colorBase")
        loc_vista = glGetUniformLocation(self.shader, "view")
        loc_proj = glGetUniformLocation(self.shader, "projection")
        loc_modelo = glGetUniformLocation(self.shader, "model")
        loc_tipo = glGetUniformLocation(self.shader, "objectType")

        print("LIDAR ACTIVADO - Presiona L para escanear")

        contador_frames = 0
        resultados_mostrados = None
        altura_mostrada = 0.0
        regularidad_mostrada = 0.0
        densidad_mostrada = 0.0

        while not glfw.window_should_close(self.ventana):
            self.procesar_inputs()
            self.dron.actualizar_vuelo()

            nuevos_impactos = []
            if self.escaneando:
                nuevos_impactos = self.sensor.emitir_rayos(self.dron.posicion)

            if contador_frames % 15 == 0 and len(self.visualizador.puntos) > 50:
                area_analisis = 2500.0
                resultados, altura, regularidad, densidad = self.analizador.evaluar_puntos_lidar(
                    self.visualizador.puntos, area_analisis
                )
                if resultados:
                    resultados_mostrados = resultados
                    altura_mostrada = altura
                    regularidad_mostrada = regularidad
                    densidad_mostrada = densidad

            glClearColor(0.5, 0.8, 0.9, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUseProgram(self.shader)

            glUniformMatrix4fv(loc_vista, 1, GL_TRUE, self.camara.obtener_matriz_vista())
            glUniformMatrix4fv(loc_proj, 1, GL_TRUE, self.proyeccion)
            glUniformMatrix4fv(loc_modelo, 1, GL_TRUE, np.eye(4, dtype=np.float32))

            self.escenario.dibujar(loc_tipo)
            self.visualizador.actualizar_y_dibujar(nuevos_impactos, self.dron.posicion, loc_color, loc_tipo)
            self.dron.dibujar(self.shader, loc_modelo, loc_color, loc_tipo)

            self.render_hud(resultados_mostrados, altura_mostrada, regularidad_mostrada, densidad_mostrada)

            glfw.swap_buffers(self.ventana)
            glfw.poll_events()
            contador_frames += 1

        glfw.terminate()
        print("LIDAR DESACTIVADO")


if __name__ == "__main__":
    SimuladorTerraLidar().ejecutar()
