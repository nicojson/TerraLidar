import numpy as np
import csv
from scipy.ndimage import grey_opening
from scipy.interpolate import griddata

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
