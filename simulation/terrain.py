import math
import random
from OpenGL.GL import *

class Perlin2D:
    def __init__(self, seed=None):
        if seed is None:
            seed = random.randint(0, 1000000)
        rng = random.Random(seed)
        self.perm = list(range(256))
        rng.shuffle(self.perm)
        self.perm += self.perm

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def _grad(h, x, y):
        h &= 3
        if h == 0: return x + y
        if h == 1: return -x + y
        if h == 2: return x - y
        return -x - y

    def noise(self, x, y):
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u = self._fade(xf)
        v = self._fade(yf)
        a = self.perm[xi] + yi
        b = self.perm[xi + 1] + yi
        return self._lerp(
            self._lerp(self._grad(self.perm[a], xf, yf),
                       self._grad(self.perm[b], xf - 1, yf), u),
            self._lerp(self._grad(self.perm[a + 1], xf, yf - 1),
                       self._grad(self.perm[b + 1], xf - 1, yf - 1), u), v)

    def fbm(self, x, y, octaves=1, lacunarity=2.0, gain=0.5): # Ultra optimizado: solo 1 octava
        val = 0.0
        amp = 0.5
        freq = 1.0
        for _ in range(octaves):
            val += self.noise(x * freq, y * freq) * amp
            freq *= lacunarity
            amp *= gain
        return val

# Instancia global por defecto
global_perlin = Perlin2D()

def set_terreno_seed(seed=None):
    global global_perlin
    global_perlin = Perlin2D(seed)

def altura_terreno(x, z):
    # Ultra optimizado: quitamos operaciones sin() complejas
    surco = 0.12 * math.sin(x * 3.1415)
    noise = global_perlin.fbm(x * 0.15, z * 0.15) * 0.6
    crop = 0.18 * max(0.0, math.sin(x * 3.1415))
    return surco + noise + crop

class TerrenoAgricola:
    def __init__(self):
        # En vez de tener todo el terreno gigante de golpe, no generamos nada.
        # Solo inicializamos OpenGL
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ibo = glGenBuffers(1)
        self.n_indices = 0

    def dibujar(self, loc_tipo):
        # Como no hay terreno base cargado, no dibujamos nada.
        # El terreno "lo genera" el dron en forma de puntos lidar.
        pass
