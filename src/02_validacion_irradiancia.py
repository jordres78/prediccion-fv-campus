"""Paso 2. Comparación de la irradiancia global horizontal (GHI) satelital con una estación local.

La estación registra cada ~2 min. Se calculan medias horarias solo cuando la hora tiene al menos
24 registros válidos (≈80 %). Se compara la GHI horaria en horas diurnas y la irradiación diaria,
y se verifica la alineación temporal buscando el desfase (−3 a +3 h) de máxima correlación.

Salidas en resultados/validacion_irradiancia/: resumen JSON y figura (dispersión y perfil horario).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from comun import cargar_config, ruta, carpeta_resultados, leer_nasa_power, metricas_validacion, guardar_json, coma

cfg = cargar_config()
salida = carpeta_resultados('validacion_irradiancia')

# Serie satelital (hora local, sin zona horaria para alinear con la estación)
sat = leer_nasa_power(ruta(cfg['archivos']['nasa_irradiancia_csv']), cfg['sitio']['zona_horaria'])['ALLSKY_SFC_SW_DWN']
sat.index = sat.index.tz_localize(None)

# Estación local: filtro físico (0–1.600 W/m²) y medias horarias con cobertura suficiente
est = pd.read_excel(ruta(cfg['archivos']['estacion_xlsx']))
est['date'] = pd.to_datetime(est['date'])
est = est.set_index('date').sort_index()
g = est['irradiance'].where((est['irradiance'] >= 0) & (est['irradiance'] < 1600))
conteo = g.resample('h').count()
estacion = g.resample('h').mean()[conteo >= 24]

resumen = dict(registros=int(len(est)), inicio=str(est.index.min()), fin=str(est.index.max()),
               horas_completas=int(len(estacion)))

# Alineación temporal: correlación para desfases de −3 a +3 h
corr = {}
for L in range(-3, 4):
    j = pd.concat([estacion, sat.shift(L)], axis=1, keys=['est', 'sat'], sort=True).dropna()
    corr[L] = float(np.corrcoef(j.est, j.sat)[0, 1])
resumen['correlacion_por_desfase_h'] = corr
resumen['desfase_optimo_h'] = max(corr, key=corr.get)

# Comparación horaria en horas diurnas (06–18 h con irradiancia > 5 W/m² en alguna de las dos series)
j = pd.concat([estacion, sat], axis=1, keys=['est', 'sat'], sort=True).dropna()
diurno = j[(j.index.hour >= 6) & (j.index.hour <= 18) & ((j.est > 5) | (j.sat > 5))]
resumen['horaria_diurna'] = metricas_validacion(diurno.est, diurno.sat)

# Comparación diaria (kWh/m²·día) con días que tienen las 13 horas diurnas completas
jd = j[(j.index.hour >= 6) & (j.index.hour <= 18)]
horas = jd.groupby(jd.index.date).size()
diaria = jd.groupby(jd.index.date).sum()[horas >= 12] / 1000
resumen['dias_completos'] = int(len(diaria))
if len(diaria) > 3:
    resumen['diaria_kWh_m2'] = metricas_validacion(diaria.est, diaria.sat)
guardar_json(resumen, f'{salida}/validacion_irradiancia.json')
print(resumen['horaria_diurna'])

# Figura: (a) dispersión horaria; (b) perfil horario medio
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
m = resumen['horaria_diurna']
ax[0].scatter(diurno.est, diurno.sat, s=6, alpha=0.4)
ax[0].plot([0, 1300], [0, 1300], 'k--', lw=1)
ax[0].set_title(f"(a) GHI horaria (n = {m['n']}); r = {coma(m['r'], 2)}; nRMSE = {coma(m['nRMSE_pct'], 0)} %", fontsize=10)
ax[0].set_xlabel('Estación local (W/m²)')
ax[0].set_ylabel('NASA POWER (W/m²)')
ax[0].grid(alpha=0.3)
perfil = j.groupby(j.index.hour).mean()
ax[1].plot(perfil.index, perfil.est, 'k-o', ms=3, label='Estación local')
ax[1].plot(perfil.index, perfil.sat, '-s', ms=3, label='NASA POWER')
ax[1].set_title('(b) Perfil horario medio del periodo común', fontsize=10)
ax[1].set_xlabel('Hora local')
ax[1].set_ylabel('W/m²')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{salida}/fig_validacion_irradiancia.png', dpi=300)
plt.close()
