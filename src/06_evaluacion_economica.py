"""Paso 6. Balance horario y evaluación financiera de la ampliación fotovoltaica.

Balance (esquema de autogeneración a pequeña escala con entrega de excedentes):
  * En cada hora la generación se destina primero al autoconsumo; el resto es excedente.
  * Al cierre de cada mes, los excedentes iguales o inferiores a la energía importada se reconocen como
    créditos de energía (valorados con el componente de generación del costo unitario) y el resto se
    vende al precio de bolsa del mes.
  * El autoconsumo evita el costo unitario más la contribución.

Consumo: serie horaria del medidor. Desde la fecha en que el generador existente quedó detrás del
medidor se reconstruye el consumo bruto; la ampliación enfrenta el consumo bruto menos la generación del
sistema existente, que sigue en operación.

Indicadores: VPN, TIR, periodos de retorno simple y descontado y costo nivelado de la energía (LCOE),
con degradación anual, operación y mantenimiento proporcional a la inversión y reemplazo de inversores.
Se incluyen la variabilidad interanual del recurso, un análisis de sensibilidad univariado y el
dimensionamiento (escala lineal de la planta y de la inversión).

Salidas en resultados/economia/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from comun import cargar_config, carpeta_resultados, guardar_json

cfg = cargar_config()
e, f_cfg = cfg['economia'], cfg['fv']
TZ = cfg['sitio']['zona_horaria']
ANIO = e['anio_base']
INICIO_NETO = cfg['consumo']['inicio_generacion_detras_del_medidor']
P_AMP = f_cfg['ampliacion']['potencia_dc_kw']
REL_AC_DC = f_cfg['ampliacion']['potencia_ac_kw'] / P_AMP
salida = carpeta_resultados('economia')
d_fv = carpeta_resultados('fv')

BASE = dict(
    costo_evitado=round(e['costo_unitario_cop_kwh'] * (1 + e['contribucion']), 2),   # COP/kWh autoconsumido
    valor_credito=e['componente_generacion_cop_kwh'],                                  # COP/kWh compensado
    bolsa=list(e['precio_bolsa_mensual_cop_kwh']),                                     # COP/kWh, enero–diciembre
    capex_usd=round(e['capex_usd_kw'] * P_AMP),
    trm=e['trm_cop_usd'],
    om_pct=e['om_pct_capex'],
    reemplazo_usd=e['reemplazo_inversores_usd'],
    anio_reemplazo=e['anio_reemplazo_inversores'],
    degradacion=e['degradacion_pct_anual'],
    escalamiento=e['escalamiento_real_pct_anual'],
    tasa=e['tasa_descuento_pct'],
    vida=e['vida_util_anios'],
)


def leer_potencia(archivo):
    """Serie horaria de potencia (kW) en hora local sin zona horaria."""
    p = pd.read_csv(archivo, index_col=0)['P_kW']
    p.index = pd.to_datetime(p.index, utc=True).tz_convert(TZ).tz_localize(None)
    return p


fv_amp = leer_potencia(f'{d_fv}/potencia_horaria_ampliacion.csv')
medidor = pd.read_csv(f"{carpeta_resultados('consumo')}/consumo_horario.csv", index_col=0, parse_dates=True)['kWh']
fv_ex = leer_potencia(f'{d_fv}/potencia_horaria_existente.csv').reindex(medidor.index).fillna(0.0)
bruto = medidor + fv_ex.where(medidor.index >= pd.Timestamp(INICIO_NETO), 0.0)
carga = (bruto - fv_ex).clip(lower=0)          # consumo que enfrenta la ampliación


def balance(gen, anio_gen, anio_carga, prm):
    """Balance mensual entre la generación del año anio_gen y el consumo del año anio_carga."""
    g = gen[str(anio_gen)].copy()
    l = carga[str(anio_carga)].copy()
    clave = lambda x: x.index.strftime('%m-%d %H')          # alinea por mes, día y hora (descarta 29-feb sin par)
    g.index, l.index = clave(g), clave(l)
    df = pd.concat([g.rename('gen'), l.rename('consumo')], axis=1).dropna()
    df['autoconsumo'] = np.minimum(df.gen, df.consumo)
    df['importacion'] = (df.consumo - df.gen).clip(lower=0)
    df['excedente'] = (df.gen - df.consumo).clip(lower=0)
    df['mes'] = [int(k[:2]) for k in df.index]
    m = df.groupby('mes')[['gen', 'consumo', 'autoconsumo', 'importacion', 'excedente']].sum()
    m['credito'] = np.minimum(m.excedente, m.importacion)
    m['venta_bolsa'] = m.excedente - m.credito
    m['precio_bolsa'] = prm['bolsa']
    m['ahorro_autoconsumo'] = m.autoconsumo * prm['costo_evitado']
    m['valor_creditos'] = m.credito * prm['valor_credito']
    m['ingreso_bolsa'] = m.venta_bolsa * m.precio_bolsa
    return m


def finanzas(m, prm):
    """Flujo de caja anual en términos reales y sus indicadores."""
    capex = prm['capex_usd'] * prm['trm']
    om = capex * prm['om_pct'] / 100
    energia = m.gen.sum()
    beneficio = (m.ahorro_autoconsumo + m.valor_creditos + m.ingreso_bolsa).sum()
    r = prm['tasa'] / 100
    flujo, e_desc, c_desc = [-capex], 0.0, capex
    for a in range(1, prm['vida'] + 1):
        factor = (1 - prm['degradacion'] / 100) ** (a - 1) * (1 + prm['escalamiento'] / 100) ** (a - 1)
        costo = om + (prm['reemplazo_usd'] * prm['trm'] if a == prm['anio_reemplazo'] else 0)
        flujo.append(beneficio * factor - costo)
        e_desc += energia * (1 - prm['degradacion'] / 100) ** (a - 1) / (1 + r) ** a
        c_desc += costo / (1 + r) ** a
    flujo = np.array(flujo)
    t = np.arange(len(flujo))
    vpn = float(np.sum(flujo / (1 + r) ** t))
    # TIR por bisección
    vpn_de = lambda x: np.sum(flujo / (1 + x) ** t)
    lo, hi, tir = -0.99, 1.0, np.nan
    if vpn_de(lo) * vpn_de(hi) < 0:
        for _ in range(200):
            mid = (lo + hi) / 2
            if vpn_de(lo) * vpn_de(mid) <= 0:
                hi = mid
            else:
                lo = mid
        tir = 100 * (lo + hi) / 2

    def retorno(acum):
        """Año (con interpolación lineal) en que el flujo acumulado se vuelve no negativo."""
        k = next((i for i in range(1, len(acum)) if acum[i] >= 0), None)
        return None if k is None else k - 1 + (-acum[k - 1]) / (acum[k] - acum[k - 1])

    return dict(inversion_MCOP=capex / 1e6, om_MCOP_anio=om / 1e6, beneficio_anio1_MCOP=beneficio / 1e6,
                VPN_MCOP=vpn / 1e6, TIR_pct=tir, retorno_simple_anios=retorno(np.cumsum(flujo)),
                retorno_descontado_anios=retorno(np.cumsum(flujo / (1 + r) ** t)),
                LCOE_COP_kWh=c_desc / e_desc, LCOE_USD_kWh=c_desc / e_desc / prm['trm'])


res = {'supuestos': BASE}

# Caso base: generación y consumo del mismo año (coincidencia horaria real)
mb = balance(fv_amp, ANIO, ANIO, BASE)
mb.round(1).to_csv(f'{salida}/balance_mensual_{ANIO}.csv')
tot = mb.sum(numeric_only=True)
energias = ['gen', 'consumo', 'autoconsumo', 'importacion', 'excedente', 'credito', 'venta_bolsa']
valores = ['ahorro_autoconsumo', 'valor_creditos', 'ingreso_bolsa']
res['balance_anual'] = {**{f'{k}_MWh': round(float(tot[k]) / 1e3, 2) for k in energias},
                        **{f'{k}_MCOP': round(float(tot[k]) / 1e6, 2) for k in valores},
                        'autoconsumo_pct_generacion': round(100 * tot.autoconsumo / tot.gen, 1),
                        'autosuficiencia_pct_consumo': round(100 * tot.autoconsumo / tot.consumo, 1)}
res['indicadores'] = finanzas(mb, BASE)
res['consumo_bruto_MWh'] = {str(a): round(float(bruto[str(a)].sum() / 1e3), 2) for a in cfg['consumo']['anios_horarios']}

# Variabilidad interanual del recurso solar (cada año meteorológico con el consumo del año base)
var = []
for a in f_cfg['anios_meteorologicos']:
    f = finanzas(balance(fv_amp, a, ANIO, BASE), BASE)
    var.append(dict(anio_meteorologico=a, beneficio_MCOP=f['beneficio_anio1_MCOP'], VPN_MCOP=f['VPN_MCOP'], TIR_pct=f['TIR_pct']))
var = pd.DataFrame(var)
var.round(2).to_csv(f'{salida}/variabilidad_recurso.csv', index=False)
res['variabilidad_recurso'] = dict(VPN_min=float(var.VPN_MCOP.min()), VPN_max=float(var.VPN_MCOP.max()),
                                   TIR_min=float(var.TIR_pct.min()), TIR_max=float(var.TIR_pct.max()))

# Sensibilidad univariada (valor bajo / valor alto de cada supuesto)
b = BASE
sens = {
    'Precio de bolsa (×0,5 / ×2)': ('bolsa', [list(np.array(b['bolsa']) * 0.5), list(np.array(b['bolsa']) * 2)]),
    'Inversión (−20 % / +20 %)': ('capex_usd', [b['capex_usd'] * 0.8, b['capex_usd'] * 1.2]),
    'TRM (3.000 / 4.200)': ('trm', [3000, 4200]),
    'Tasa de descuento (6 % / 14 %)': ('tasa', [6, 14]),
    'Costo evitado por kWh (−15 % / +15 %)': ('costo_evitado', [round(b['costo_evitado'] * 0.85, 2), round(b['costo_evitado'] * 1.15, 2)]),
    'Valor del crédito de energía (−15 % / +15 %)': ('valor_credito', [round(b['valor_credito'] * 0.85, 2), round(b['valor_credito'] * 1.15, 2)]),
    'O&M (0,5 % / 2 % de la inversión)': ('om_pct', [0.5, 2.0]),
    'Escalamiento real de precios (−1 % / +2 %)': ('escalamiento', [-1, 2]),
}
filas = []
for etiqueta, (clave, (bajo, alto)) in sens.items():
    pb, pa = {**BASE, clave: bajo}, {**BASE, clave: alto}
    rb, ra = finanzas(balance(fv_amp, ANIO, ANIO, pb), pb), finanzas(balance(fv_amp, ANIO, ANIO, pa), pa)
    filas.append(dict(variable=etiqueta, VPN_bajo=rb['VPN_MCOP'], VPN_alto=ra['VPN_MCOP'], TIR_bajo=rb['TIR_pct'],
                      TIR_alto=ra['TIR_pct'], LCOE_bajo=rb['LCOE_COP_kWh'], LCOE_alto=ra['LCOE_COP_kWh']))
sens_df = pd.DataFrame(filas)
sens_df.round(2).to_csv(f'{salida}/sensibilidad.csv', index=False)

# Dimensionamiento: escala lineal de la generación, la inversión y el reemplazo de inversores
dim = []
carga_base = carga[str(ANIO)]
for kw in e['potencias_dimensionamiento_kw']:
    k = kw / P_AMP
    gen_k = fv_amp * k
    prm = dict(BASE, capex_usd=BASE['capex_usd'] * k, reemplazo_usd=BASE['reemplazo_usd'] * k)
    m = balance(gen_k, ANIO, ANIO, prm)
    f = finanzas(m, prm)
    g = gen_k[str(ANIO)]
    pico = float((g - carga_base.reindex(g.index).fillna(0)).clip(lower=0).max())
    dim.append(dict(potencia_kWp=kw, potencia_AC_kW=round(kw * REL_AC_DC, 1), pico_exportacion_kW=round(pico, 1),
                    energia_MWh=m.gen.sum() / 1e3, autoconsumo_pct=100 * m.autoconsumo.sum() / m.gen.sum(),
                    autosuficiencia_pct=100 * m.autoconsumo.sum() / m.consumo.sum(), VPN_MCOP=f['VPN_MCOP'],
                    TIR_pct=f['TIR_pct'], LCOE_COP_kWh=f['LCOE_COP_kWh'], retorno_descontado_anios=f['retorno_descontado_anios']))
dim = pd.DataFrame(dim)
dim.round(2).to_csv(f'{salida}/dimensionamiento.csv', index=False)
guardar_json(res, f'{salida}/evaluacion_economica.json')
print(res['balance_anual'])
print(res['indicadores'])
print(sens_df.round(1).to_string())
print(dim.round(1).to_string())

# Figura: (a) perfiles horarios medios del año base; (b) sensibilidad del VPN
g, l = fv_amp[str(ANIO)], carga[str(ANIO)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].plot(range(24), g.groupby(g.index.hour).mean(), '-o', ms=3, label=f"Generación de la ampliación ({str(P_AMP).replace('.', ',')} kWp)")
ax[0].plot(range(24), l.groupby(l.index.hour).mean(), 'k-s', ms=3, label='Consumo neto del sistema existente')
ax[0].set_xlabel('Hora del día')
ax[0].set_ylabel('kW (media horaria)')
ax[0].set_title(f'(a) Perfiles horarios medios, {ANIO}', fontsize=10)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
vpn0 = res['indicadores']['VPN_MCOP']
orden = sens_df.assign(rango=(sens_df.VPN_alto - sens_df.VPN_bajo).abs()).sort_values('rango')
y = np.arange(len(orden))
ax[1].barh(y, orden.VPN_bajo - vpn0, left=vpn0, color='#c55a11', label='Valor bajo')
ax[1].barh(y, orden.VPN_alto - vpn0, left=vpn0, color='#2f5597', label='Valor alto')
ax[1].axvline(vpn0, color='k', lw=1)
ax[1].axvline(0, color='grey', lw=0.8, ls='--')
ax[1].set_yticks(y)
ax[1].set_yticklabels(orden.variable, fontsize=8)
ax[1].set_xlabel('VPN (millones de COP)')
ax[1].set_title(f'(b) Sensibilidad del VPN (caso base = {vpn0:,.0f} MCOP)'.replace(',', '.'), fontsize=10)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{salida}/fig_balance_sensibilidad.png', dpi=300)
plt.close()
