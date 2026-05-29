from OpenGL.GL import *
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18

class RenderizadorTexto:
    def __init__(self):
        self.fuente_lista = glGenLists(256)
        self.compilar_fuente()

    def compilar_fuente(self):
        for i in range(256):
            glNewList(self.fuente_lista + i, GL_COMPILE)
            glEndList()

    def render_texto_2d(self, x, y, texto, ancho_ventana, alto_ventana):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, ancho_ventana, alto_ventana, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glColor3f(0.95, 0.95, 0.95)

        glRasterPos2f(x, y)
        for char in texto:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def render_texto_simple(self, x, y, texto, ancho_ventana, alto_ventana):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, ancho_ventana, alto_ventana, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glColor3f(0.95, 0.95, 0.95)
        glWindowPos2f(int(x), int(y))

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

