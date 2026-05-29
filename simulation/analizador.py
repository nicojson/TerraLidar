import numpy as np
import random

class AnalizadorAgricola:
    def __init__(self, config):
        self.tipo_analisis = config.get('analisis', 1)
        self.densidad_laser = config.get('densidad', 1000)

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
