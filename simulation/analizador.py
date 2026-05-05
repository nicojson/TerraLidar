import numpy as np
from collections import deque

class AnalizadorAgricola:
    def __init__(self):
        self.cultivos = {
            'CEBOLLA': {
                'altura_ideal_min': -0.5,
                'altura_ideal_max': 0.3,
                'regularidad_min': 0.0,
                'regularidad_max': 0.4,
                'densidad_min': 0.3,
                'rendimiento_kg_ha': 45000,
                'dias_cosecha': 120,
            },
            'JITOMATE': {
                'altura_ideal_min': -0.2,
                'altura_ideal_max': 0.8,
                'regularidad_min': 0.1,
                'regularidad_max': 0.6,
                'densidad_min': 0.2,
                'rendimiento_kg_ha': 80000,
                'dias_cosecha': 70,
            },
            'CILANTRO': {
                'altura_ideal_min': -0.3,
                'altura_ideal_max': 0.4,
                'regularidad_min': 0.0,
                'regularidad_max': 0.5,
                'densidad_min': 0.25,
                'rendimiento_kg_ha': 15000,
                'dias_cosecha': 45,
            },
            'CHILES': {
                'altura_ideal_min': 0.0,
                'altura_ideal_max': 1.0,
                'regularidad_min': 0.2,
                'regularidad_max': 0.7,
                'densidad_min': 0.2,
                'rendimiento_kg_ha': 25000,
                'dias_cosecha': 90,
            },
            'LECHUGA': {
                'altura_ideal_min': -0.4,
                'altura_ideal_max': 0.2,
                'regularidad_min': 0.0,
                'regularidad_max': 0.3,
                'densidad_min': 0.35,
                'rendimiento_kg_ha': 50000,
                'dias_cosecha': 55,
            },
        }
        self.historial_analisis = deque(maxlen=500)

    def calcular_aptitud(self, altura_promedio, regularidad_terreno, densidad_puntos, area_m2):
        resultados = {}

        for cultivo, params in self.cultivos.items():
            altura_score = 0.0
            if params['altura_ideal_min'] <= altura_promedio <= params['altura_ideal_max']:
                altura_score = 1.0
            else:
                dist_min = abs(altura_promedio - params['altura_ideal_min'])
                dist_max = abs(altura_promedio - params['altura_ideal_max'])
                dist = min(dist_min, dist_max)
                altura_score = max(0.0, 1.0 - (dist * 0.5))

            regularidad_score = 0.0
            if params['regularidad_min'] <= regularidad_terreno <= params['regularidad_max']:
                regularidad_score = 1.0
            else:
                if regularidad_terreno < params['regularidad_min']:
                    regularidad_score = max(0.0, 1.0 - ((params['regularidad_min'] - regularidad_terreno) * 2.0))
                else:
                    regularidad_score = max(0.0, 1.0 - ((regularidad_terreno - params['regularidad_max']) * 2.0))

            densidad_score = 0.0
            if densidad_puntos >= params['densidad_min']:
                densidad_score = min(1.0, densidad_puntos / 0.8)
            else:
                densidad_score = (densidad_puntos / params['densidad_min']) * 0.7

            aptitud_final = (altura_score * 0.35) + (regularidad_score * 0.35) + (densidad_score * 0.30)
            aptitud_porcentaje = int(aptitud_final * 100)

            area_ha = area_m2 / 10000.0
            rendimiento_estimado = (aptitud_final * params['rendimiento_kg_ha'] * area_ha)

            resultados[cultivo] = {
                'aptitud': aptitud_porcentaje,
                'rendimiento_kg': int(rendimiento_estimado),
                'dias_cosecha': params['dias_cosecha'],
                'altura_score': altura_score,
                'regularidad_score': regularidad_score,
                'densidad_score': densidad_score,
            }

        return resultados

    def evaluar_puntos_lidar(self, puntos_lidar, area_m2):
        if len(puntos_lidar) == 0:
            return None, 0.0, 0.0, 0.0

        puntos_array = np.array(puntos_lidar)
        altura_promedio = np.mean(puntos_array[:, 1])
        altura_std = np.std(puntos_array[:, 1])
        regularidad_terreno = altura_std

        densidad_puntos = min(len(puntos_lidar) / 10000.0, 1.0)

        resultados = self.calcular_aptitud(altura_promedio, regularidad_terreno, densidad_puntos, area_m2)

        self.historial_analisis.append({
            'timestamp': len(self.historial_analisis),
            'altura_promedio': altura_promedio,
            'regularidad': regularidad_terreno,
            'densidad': densidad_puntos,
            'resultados': resultados
        })

        mejor_cultivo = max(resultados.items(), key=lambda x: x[1]['aptitud'])
        return resultados, altura_promedio, regularidad_terreno, densidad_puntos

