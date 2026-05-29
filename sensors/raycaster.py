from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import os

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

        shader_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'shaders', 'raytracing.comp')
        with open(shader_path, "r") as f:
            cs = f.read()
            
        self.compute_shader = compileShader(cs, GL_COMPUTE_SHADER)
        self.program = compileProgram(self.compute_shader)
        glDeleteShader(self.compute_shader)

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
        result_bytes = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, hit_data.nbytes)
        result = np.frombuffer(result_bytes, dtype=np.float32)
        result = result.reshape(-1, 4)
        
        hits = []
        for r in result:
            if r[3] > 0.5:
                hits.append(r[:3])
        return hits
