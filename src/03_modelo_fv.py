"""Paso 3. Modelo fotovoltaico horario, validación con la producción medida y año meteorológico típico.

Cadena de cálculo (pvlib):
  1. Posición solar en el centro de cada hora.
  2. Descomposición de la GHI en componentes directa y difusa (modelo de Erbs).
  3. Transposición al plano del módulo (modelo de Hay-Davies) y pérdidas por ángulo de incidencia.
  4. Temperatura de celda (modelo de Faiman) con la temperatura ambiente y el viento.
  5. Potencia DC (PVWatts) con coeficiente de temperatura y pérdidas del sistema.
  6. Potencia AC con la eficiencia del inversor y recorte a su potencia nominal.

El modelo del sistema existente se compara con la energía diaria medida por su inversor; la relación
entre la energía medida y la modelada se usa como factor de calibración de la ampliación.

Salidas en resultados/fv/: resumen JSON, energía anual por año meteorológico, series horarias
calibradas (sistema existente y ampliación), serie diaria del sistema existente y figura de validación.
"""
import numpy as np
import pandas as pd
import pvlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from comun import (cargar_config, ruta, carpeta_resultados, leer_nasa_power, metricas_validacion,
                   guardar_json, coma, MESES_ES)

cfg = cargar_config()
s, f_cfg = cfg['sitio'], cfg['fv']
TZ = s['zona_horaria']
salida = carpeta_resultados('fv')
res = {}

# --------------------------------------------------------------------------- datos meteorológicos
w = leer_nasa_power(ruta(cfg['archivos']['nasa_irradiancia_csv']), TZ).join(
    leer_nasa_power(ruta(cfg['archivos']['nasa_meteorologia_csv']), TZ))
w = w.rename(columns={'ALLSKY_SFC_SW_DWN': 'ghi', 'T2M': 'tamb', 'WS10M': 'ws'})
w = w.interpolate(limit=3)                      # huecos de hasta 3 h
fin = w.ghi.last_valid_index()                  # último dato disponible de irradiancia
w = w.loc[:fin]

ubic = pvlib.location.Location(s['latitud'], s['longitud'], TZ, s['altitud_m'])
sol = ubic.get_solarposition(w.index + pd.Timedelta(minutes=30))   # centro de la hora
sol.index = w.index
# Las componentes directa y difusa del producto satelital no cierran con la GHI; por coherencia se
# obtienen de la GHI con el modelo de Erbs (el mismo procedimiento se aplica a la estación local).
comp = pvlib.irradiance.erbs(w.ghi.clip(lower=0), sol['apparent_zenith'], w.index)
w['dni'], w['dhi'] = comp['dni'].fillna(0), comp['dhi'].fillna(0)

PERDIDAS = 100 * (1 - np.prod([1 - v / 100 for v in f_cfg['perdidas_pct'].values()]))   # combinación multiplicativa


def potencia_ac(ghi, dni, dhi, tamb, viento, pdc_w, pac_w):
    """Potencia AC horaria media (W) de un generador de pdc_w (W en condiciones estándar)."""
    zen, azi = sol.loc[ghi.index, 'apparent_zenith'], sol.loc[ghi.index, 'azimuth']
    poa = pvlib.irradiance.get_total_irradiance(
        f_cfg['inclinacion_grados'], f_cfg['azimut_grados'], zen, azi, dni, ghi, dhi,
        dni_extra=pvlib.irradiance.get_extra_radiation(ghi.index), model='haydavies', albedo=f_cfg['albedo'])
    aoi = pvlib.irradiance.aoi(f_cfg['inclinacion_grados'], f_cfg['azimut_grados'], zen, azi)
    efectiva = (poa['poa_direct'] * pvlib.iam.physical(aoi) + poa['poa_diffuse']).fillna(0).clip(lower=0)
    # viento a 2 m estimado desde 10 m con ley potencial (exponente 0,2)
    tcelda = pvlib.temperature.faiman(efectiva, tamb, viento * (2 / 10) ** 0.2)
    pdc = pvlib.pvsystem.pvwatts_dc(efectiva, tcelda, pdc_w, f_cfg['coef_temperatura_potencia_por_C']) * (1 - PERDIDAS / 100)
    eta = f_cfg['eficiencia_inversor']
    pac = pvlib.inverter.pvwatts(pdc, pac_w / eta, eta_inv_nom=eta)
    return pac.clip(lower=0).fillna(0), poa['poa_global'].fillna(0)


res['supuestos'] = dict(inclinacion=f_cfg['inclinacion_grados'], azimut=f_cfg['azimut_grados'],
                        perdidas_pct=f_cfg['perdidas_pct'], perdidas_totales_pct=round(PERDIDAS, 2),
                        transposicion='Hay-Davies', temperatura_celda='Faiman', descomposicion='Erbs',
                        fin_datos_irradiancia=str(fin))

# --------------------------------------------------------------------------- producción medida
inv = pd.read_csv(ruta(cfg['archivos']['inversor_csv']), usecols=['date', 'acPower'], parse_dates=['date'])
inv = inv.set_index('date').sort_index()
inv.index = inv.index.tz_localize(TZ)
# energía de cada registro = potencia × intervalo hasta el siguiente registro (máximo 5 min)
dt = inv.index.to_series().diff().dt.total_seconds().shift(-1).fillna(120).clip(upper=300)
inv['Wh'] = inv.acPower.clip(lower=0) * dt / 3600
# día válido: registros entre 06:00 y 18:00 sin huecos mayores de 10 min
huecos = inv.index.to_series().diff().dt.total_seconds()
ventana = inv[(inv.index.hour >= 6) & (inv.index.hour < 18)]
hv = huecos.reindex(ventana.index)
cob = pd.DataFrame({'primero': ventana.index.to_series().groupby(ventana.index.date).min(),
                    'ultimo': ventana.index.to_series().groupby(ventana.index.date).max(),
                    'hueco_max': hv.groupby(ventana.index.date).apply(lambda x: x.iloc[1:].max() if len(x) > 1 else np.inf)})
validos = cob[(cob.primero.dt.hour <= 6) & (cob.primero.dt.minute <= 30) & (cob.ultimo.dt.hour >= 17)
              & (cob.ultimo.dt.minute >= 30) & (cob.hueco_max <= 600)].index
E_medida = inv.Wh.groupby(inv.index.date).sum()[validos] / 1000          # kWh/día
P_EX = f_cfg['sistema_existente']['potencia_dc_kw']
res['produccion_medida'] = dict(inicio=str(inv.index.min()), fin=str(inv.index.max()), dias_validos=int(len(E_medida)),
                                energia_media_kWh_dia=round(float(E_medida.mean()), 2),
                                rendimiento_kWh_kWp_dia=round(float(E_medida.mean() / P_EX), 3))

# --------------------------------------------------------------------------- validación del sistema existente
pac_ex, poa = potencia_ac(w.ghi, w.dni, w.dhi, w.tamb, w.ws, P_EX * 1000,
                          f_cfg['sistema_existente']['potencia_ac_kw'] * 1000)
E_modelo = pac_ex.groupby(pac_ex.index.date).sum() / 1000
H_poa = poa.groupby(poa.index.date).sum() / 1000
horas_ok = w.ghi.notna().groupby(w.index.date).sum()
dias = sorted(set(E_medida.index) & set(E_modelo.index) & set(horas_ok[horas_ok == 24].index))
med, mod = E_medida.loc[dias], E_modelo.loc[dias]
res['validacion_diaria'] = metricas_validacion(med, mod)
res['rendimiento_global_medido'] = round(float(med.sum() / (H_poa.loc[dias].sum() * P_EX)), 3)
res['rendimiento_global_modelo'] = round(float(mod.sum() / (H_poa.loc[dias].sum() * P_EX)), 3)
cal = float(med.sum() / mod.sum())
res['factor_calibracion'] = round(cal, 3)

# Comparación adicional: el mismo modelo alimentado con la estación local
est = pd.read_excel(ruta(cfg['archivos']['estacion_xlsx']))
est['date'] = pd.to_datetime(est['date'])
est = est.set_index('date').sort_index()
g = est['irradiance'].where((est['irradiance'] >= 0) & (est['irradiance'] < 1600))
gh = g.resample('h').mean()[g.resample('h').count() >= 24]
gh.index = gh.index.tz_localize(TZ)
gest = gh.reindex(w.index)
ok = gest.notna()
comp_e = pvlib.irradiance.erbs(gest[ok], sol['apparent_zenith'][ok], gest[ok].index)
pac_est, _ = potencia_ac(gest[ok], comp_e['dni'], comp_e['dhi'], w.tamb[ok], w.ws[ok], P_EX * 1000,
                         f_cfg['sistema_existente']['potencia_ac_kw'] * 1000)
E_est = pac_est.groupby(pac_est.index.date).sum() / 1000
horas_est = ok.groupby(ok.index.date).sum()
dias_est = [d for d in dias if d in E_est.index and horas_est.get(d, 0) >= 24]
if len(dias_est) > 5:
    res['validacion_diaria_estacion'] = metricas_validacion(E_medida.loc[dias_est], E_est.loc[dias_est])
    res['validacion_diaria_satelite_mismos_dias'] = metricas_validacion(E_medida.loc[dias_est], E_modelo.loc[dias_est])

# --------------------------------------------------------------------------- ampliación: años meteorológicos
P_AMP = f_cfg['ampliacion']['potencia_dc_kw']
pac_amp, poa_amp = potencia_ac(w.ghi, w.dni, w.dhi, w.tamb, w.ws, P_AMP * 1000,
                               f_cfg['ampliacion']['potencia_ac_kw'] * 1000)
anios = f_cfg['anios_meteorologicos']
anual = pd.DataFrame({'GHI_kWh_m2': [w.ghi[str(a)].sum() / 1000 for a in anios],
                      'POA_kWh_m2': [poa_amp[str(a)].sum() / 1000 for a in anios],
                      'E_modelo_MWh': [pac_amp[str(a)].sum() / 1e6 for a in anios]}, index=anios)
anual['E_calibrada_MWh'] = anual.E_modelo_MWh * cal
anual['rendimiento_kWh_kWp'] = 1000 * anual.E_calibrada_MWh / P_AMP
anual.index.name = 'anio'
anual.to_csv(f'{salida}/energia_anual_ampliacion.csv')
E = anual.E_calibrada_MWh
res['ampliacion'] = dict(potencia_kWp=P_AMP, energia_media_MWh=round(float(E.mean()), 1),
                         energia_min_MWh=round(float(E.min()), 1), energia_max_MWh=round(float(E.max()), 1),
                         desviacion_MWh=round(float(E.std()), 1), CV_pct=round(float(100 * E.std() / E.mean()), 2),
                         P90_MWh=round(float(E.mean() - 1.282 * E.std()), 1),        # supuesto de normalidad
                         rendimiento_kWh_kWp=round(float(anual.rendimiento_kWh_kWp.mean()), 0),
                         factor_capacidad_pct=round(float(100 * E.mean() * 1e3 / (P_AMP * 8760)), 2))

# Series para los pasos siguientes (potencia media horaria calibrada, kW)
(pac_amp[str(anios[0]):str(anios[-1])] * cal / 1000).to_frame('P_kW').to_csv(f'{salida}/potencia_horaria_ampliacion.csv')
(pac_ex * cal / 1000).to_frame('P_kW').to_csv(f'{salida}/potencia_horaria_existente.csv')
# Serie diaria del sistema existente: medición cuando existe y modelo calibrado en el resto
d = pd.DataFrame({'E_modelo_calibrado_kWh': E_modelo * cal})
d['E_medida_kWh'] = E_medida.reindex(d.index)
d['E_usada_kWh'] = d.E_medida_kWh.fillna(d.E_modelo_calibrado_kWh)
d.index.name = 'fecha'
d.to_csv(f'{salida}/energia_diaria_existente.csv')
guardar_json(res, f'{salida}/modelo_fv.json')
print(res['validacion_diaria'], res['factor_calibracion'], res['ampliacion'])

# --------------------------------------------------------------------------- figura de validación
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(med, mod, s=10, alpha=0.6, label='Modelo con NASA POWER')
if len(dias_est) > 5:
    ax[0].scatter(E_medida.loc[dias_est], E_est.loc[dias_est], s=10, alpha=0.6, marker='^', label='Modelo con estación local')
lim = float(max(med.max(), mod.max())) * 1.05
ax[0].plot([0, lim], [0, lim], 'k--', lw=1)
ax[0].set_xlabel('Energía medida (kWh/día)')
ax[0].set_ylabel('Energía modelada (kWh/día)')
v = res['validacion_diaria']
ax[0].set_title(f"(a) Sistema existente, {v['n']} días; nMBE = {'+' if v['nMBE_pct'] >= 0 else '−'}{coma(abs(v['nMBE_pct']))} %", fontsize=10)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
mm = pd.DataFrame({'medida': med, 'modelo': mod})
mm.index = pd.to_datetime(mm.index)
mm = mm.resample('ME').mean()
x = np.arange(len(mm))
ax[1].bar(x - 0.2, mm.medida, 0.4, label='Medida (inversor)')
ax[1].bar(x + 0.2, mm.modelo, 0.4, label='Modelo (NASA POWER)')
ax[1].set_xticks(x)
ax[1].set_xticklabels([MESES_ES[t.month - 1] for t in mm.index])
ax[1].set_ylabel('kWh/día (media mensual)')
ax[1].set_title('(b) Media mensual de la energía diaria', fontsize=10)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{salida}/fig_validacion_fv.png', dpi=300)
plt.close()
