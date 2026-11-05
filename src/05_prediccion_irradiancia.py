"""Paso 5. Predicción de la irradiancia global horizontal (GHI) horaria.

Horizontes de 1 h y 24 h con origen móvil. Cada muestra se indexa por el instante objetivo; las
entradas usan solo información disponible en el instante de emisión, más variables astronómicas
(determinísticas) del instante objetivo.

Referencias: persistencia, persistencia de 24 h, climatología mes-hora del periodo de entrenamiento y
persistencia inteligente (conserva el índice de cielo despejado del instante de emisión). Las métricas
se calculan solo en horas diurnas (ángulo cenital aparente menor que el umbral configurado), porque las
horas nocturnas, con irradiancia nula, elevan artificialmente el R².

Salidas en resultados/irradiancia/: métricas y predicciones por horizonte, resumen JSON y figura.
"""
import os
import time
import warnings

os.environ.setdefault('KERAS_BACKEND', 'jax')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pvlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid

from comun import (cargar_config, ruta, carpeta_resultados, leer_nasa_power, metricas, ic_rmse_dias,
                   guardar_json, coma, MESES_ES)

cfg = cargar_config()
s, ci = cfg['sitio'], cfg['irradiancia']
TZ = s['zona_horaria']
SEMILLA = ci['semilla']
FIN_ENT = pd.Timestamp(cfg['particion']['fin_entrenamiento'] + ' 23:00', tz=TZ)
FIN_VAL = pd.Timestamp(cfg['particion']['fin_validacion'] + ' 23:00', tz=TZ)
salida = carpeta_resultados('irradiancia')
registro = {}

# --------------------------------------------------------------------------- datos
df = leer_nasa_power(ruta(cfg['archivos']['nasa_irradiancia_csv']), TZ).join(
    leer_nasa_power(ruta(cfg['archivos']['nasa_meteorologia_csv']), TZ), how='inner')
df = df.rename(columns={'ALLSKY_SFC_SW_DWN': 'ghi'})
registro['datos'] = dict(inicio=str(df.index[0]), fin=str(df.index[-1]), n_horas=int(len(df)),
                         ghi_faltantes=int(df.ghi.isna().sum()))
df['ghi'] = df.ghi.interpolate(limit=3).clip(lower=0)
for col in ['T2M', 'RH2M', 'WS10M']:
    df[col] = df[col].interpolate(limit=6)

# Posición solar y GHI de cielo despejado (Ineichen-Perez) en el centro de cada hora
ubic = pvlib.location.Location(s['latitud'], s['longitud'], TZ, s['altitud_m'])
centro = df.index + pd.Timedelta(minutes=30)
sol = ubic.get_solarposition(centro)
despejado = ubic.get_clearsky(centro, model='ineichen', solar_position=sol)
df['cenit'] = sol['apparent_zenith'].values
df['ghi_cs'] = despejado['ghi'].values
df['cos_cenit'] = np.cos(np.radians(df.cenit)).clip(lower=0)
diurno = df.cenit < ci['cenit_max_diurno_grados']
# índice de cielo despejado kc = GHI / GHI_cs (acotado entre 0 y 1,5; 0 cuando GHI_cs ≤ 20 W/m²)
df['kc'] = np.where(df.ghi_cs > 20, (df.ghi / df.ghi_cs.where(df.ghi_cs > 20)).clip(0, 1.5), np.nan)
df['kc'] = df.kc.fillna(0.0)
hora, dia_anio = df.index.hour, df.index.dayofyear
df['sen_h'], df['cos_h'] = np.sin(2 * np.pi * hora / 24), np.cos(2 * np.pi * hora / 24)
df['sen_d'], df['cos_d'] = np.sin(2 * np.pi * dia_anio / 365.25), np.cos(2 * np.pi * dia_anio / 365.25)
df['kc_media_24h'] = df.kc.where(diurno).rolling(24, min_periods=6).mean()   # media diurna de las últimas 24 h

ent = df[:FIN_ENT]
climatologia = ent.groupby([ent.index.month, ent.index.hour]).ghi.mean()


def disenar(h):
    """Matriz de entradas, objetivo y referencias para el horizonte h (índice = instante objetivo)."""
    X = pd.DataFrame(index=df.index)
    for L in [0, 1, 2, 3]:
        X[f'kc_l{L}'] = df.kc.shift(L)
        X[f'ghi_l{L}'] = df.ghi.shift(L)
    X['kc_obj_menos24'] = df.kc.shift(24 - h)        # misma hora del día anterior al instante objetivo
    X['ghi_obj_menos24'] = df.ghi.shift(24 - h)
    X['kc_media_24h'] = df.kc_media_24h
    for col in ['T2M', 'RH2M', 'WS10M']:
        X[col] = df[col]
    for col in ['ghi_cs', 'cos_cenit', 'sen_h', 'cos_h', 'sen_d', 'cos_d']:
        X[f'{col}_obj'] = df[col].shift(-h)
    y = df.ghi.shift(-h)
    objetivo = df.index + pd.Timedelta(hours=h)
    ref = pd.DataFrame(index=df.index)
    ref['Persistencia'] = df.ghi
    ref['Persistencia inteligente'] = df.kc * df.ghi_cs.shift(-h)
    ref['Persistencia 24 h'] = df.ghi.shift(24 - h)
    ref['Climatología (mes-hora)'] = [climatologia.get((t.month, t.hour), np.nan) for t in objetivo]
    ok = X.notna().all(axis=1) & y.notna() & (df.cenit.shift(-h) < ci['cenit_max_diurno_grados'])
    X, y, ref = X[ok], y[ok], ref[ok]
    X.index = y.index = ref.index = X.index + pd.Timedelta(hours=h)
    return X, y, ref


import keras
from keras import layers

VARS_SECUENCIA = ['kc', 'ghi', 'cos_cenit', 'T2M', 'RH2M']
VARS_OBJETIVO = ['ghi_cs_obj', 'cos_cenit_obj', 'sen_h_obj', 'cos_h_obj', 'sen_d_obj', 'cos_d_obj']


def evaluar(h):
    X, y, ref = disenar(h)
    m_tr = X.index <= FIN_ENT
    m_va = (X.index > FIN_ENT) & (X.index <= FIN_VAL)
    m_te = X.index > FIN_VAL
    Xtr, ytr, Xva, yva, Xte, yte = X[m_tr], y[m_tr], X[m_va], y[m_va], X[m_te], y[m_te]
    Xtv, ytv = X[m_tr | m_va], y[m_tr | m_va]
    pred = {k: ref.loc[m_te, k] for k in ref.columns}
    hiper = {}

    def ajustar(constructor, rejilla, nombre, submuestra=None):
        """Rejilla con validación temporal; submuestra aleatoria opcional para modelos costosos."""
        rng = np.random.default_rng(SEMILLA)
        itr = np.arange(len(Xtr)) if submuestra is None else rng.choice(len(Xtr), min(submuestra, len(Xtr)), replace=False)
        mejor = None
        for g in ParameterGrid(rejilla):
            m = constructor(**g).fit(Xtr.iloc[itr], ytr.iloc[itr])
            error = metricas(yva, np.clip(m.predict(Xva), 0, None))['RMSE']
            if mejor is None or error < mejor[0]:
                mejor = (error, g)
        itv = np.arange(len(Xtv)) if submuestra is None else rng.choice(len(Xtv), min(submuestra, len(Xtv)), replace=False)
        m = constructor(**mejor[1]).fit(Xtv.iloc[itv], ytv.iloc[itv])
        pred[nombre] = pd.Series(np.clip(m.predict(Xte), 0, None), index=Xte.index)
        hiper[nombre] = {**mejor[1], 'RMSE_validacion': round(mejor[0], 2), **({'submuestra': submuestra} if submuestra else {})}
        print(h, nombre, hiper[nombre], flush=True)

    ajustar(lambda alpha: Pipeline([('esc', StandardScaler()), ('m', Ridge(alpha=alpha))]), {'alpha': [0.1, 1, 10, 100]}, 'Ridge')
    ajustar(lambda alpha: Pipeline([('esc', StandardScaler()), ('pol', PolynomialFeatures(2, include_bias=False)),
                                    ('esc2', StandardScaler()), ('m', Ridge(alpha=alpha))]),
            {'alpha': [1, 10, 100, 1000]}, 'Regresión polinomial (grado 2)')
    ajustar(lambda C, epsilon: Pipeline([('esc', StandardScaler()), ('m', SVR(C=C, epsilon=epsilon, gamma='scale'))]),
            {'C': [100, 300, 1000], 'epsilon': [5, 20]}, 'SVR', submuestra=ci['submuestra_svr'])
    ajustar(lambda max_depth, min_samples_leaf: RandomForestRegressor(n_estimators=200, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=0.5, random_state=SEMILLA, n_jobs=-1),
            {'max_depth': [None, 15], 'min_samples_leaf': [5, 20]}, 'Random Forest')
    ajustar(lambda learning_rate, max_leaf_nodes: HistGradientBoostingRegressor(learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes, max_iter=500, random_state=SEMILLA),
            {'learning_rate': [0.03, 0.1], 'max_leaf_nodes': [15, 31, 63]}, 'Gradient Boosting')

    # LSTM: secuencia de las últimas W horas (en el instante de emisión) + variables del instante objetivo
    Z = df[VARS_SECUENCIA]
    Z = ((Z - Z[:FIN_ENT].mean()) / Z[:FIN_ENT].std()).values.astype('float32')
    posicion = {t: i for i, t in enumerate(df.index)}
    mu_o, sd_o = X.loc[m_tr, VARS_OBJETIVO].mean(), X.loc[m_tr, VARS_OBJETIVO].std()
    ymu, ysd = float(ytr.mean()), float(ytr.std())

    def tensores(Xp, W):
        orig = [posicion[t - pd.Timedelta(hours=h)] for t in Xp.index]
        seq = np.stack([Z[o - W + 1:o + 1] for o in orig])
        obj = ((Xp[VARS_OBJETIVO] - mu_o) / sd_o).values.astype('float32')
        return np.concatenate([seq, np.repeat(obj[:, None, :], W, axis=1)], axis=2)

    def red(W, n_var, unidades):
        m = keras.Sequential([layers.Input((W, n_var)), layers.LSTM(unidades), layers.Dropout(0.2),
                              layers.Dense(32, activation='relu'), layers.Dense(1)])
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
        return m

    con_historia = lambda Xp: Xp[[posicion[t - pd.Timedelta(hours=h)] >= 48 for t in Xp.index]]
    Xtr2, Xva2, Xte2, Xtv2 = con_historia(Xtr), con_historia(Xva), con_historia(Xte), con_historia(Xtv)
    mejor = None
    for W in [24, 48]:
        for unidades in [32, 64]:
            keras.utils.set_random_seed(SEMILLA)
            A, B = tensores(Xtr2, W), tensores(Xva2, W)
            m = red(W, A.shape[2], unidades)
            parada = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            hist = m.fit(A, ((y[Xtr2.index] - ymu) / ysd).values, validation_data=(B, ((y[Xva2.index] - ymu) / ysd).values),
                         epochs=80, batch_size=256, verbose=0, callbacks=[parada])
            error = metricas(y[Xva2.index], np.clip(m.predict(B, verbose=0).ravel() * ysd + ymu, 0, None))['RMSE']
            print(h, 'LSTM', W, unidades, round(error, 2), flush=True)
            if mejor is None or error < mejor[0]:
                mejor = (error, W, unidades, int(np.argmin(hist.history['val_loss']) + 1))
    error_val, W, unidades, epocas = mejor
    A, T = tensores(Xtv2, W), tensores(Xte2, W)
    corridas = []
    for semilla in [SEMILLA, SEMILLA + 1, SEMILLA + 2]:
        keras.utils.set_random_seed(semilla)
        m = red(W, A.shape[2], unidades)
        m.fit(A, ((y[Xtv2.index] - ymu) / ysd).values, epochs=epocas, batch_size=256, verbose=0)
        corridas.append(np.clip(m.predict(T, verbose=0).ravel() * ysd + ymu, 0, None))
    pred['LSTM'] = pd.Series(np.mean(corridas, axis=0), index=Xte2.index)
    hiper['LSTM'] = dict(ventana_h=W, unidades=unidades, epocas=epocas, dropout=0.2, optimizador='Adam (0,001)', perdida='MSE',
                         lote=256, RMSE_validacion=round(error_val, 2),
                         RMSE_prueba_por_semilla=[round(metricas(y[Xte2.index], r)['RMSE'], 2) for r in corridas])

    # Evaluación con las mismas horas para todos los modelos
    comunes = Xte2.index
    yc = yte.reindex(comunes)
    ref_skill = pred['Persistencia inteligente'].reindex(comunes)
    filas = []
    for nombre, p in pred.items():
        p = pd.Series(p).reindex(comunes)
        mt = metricas(yc, p, ref_skill)
        mt['IC95_RMSE'] = ic_rmse_dias(yc, p, semilla=SEMILLA)
        filas.append(dict(Modelo=nombre, **mt))
    tabla = pd.DataFrame(filas).sort_values('RMSE').reset_index(drop=True)
    tabla.to_csv(f'{salida}/metricas_prueba_h{h}.csv', index=False)
    P = pd.DataFrame({k: pd.Series(v).reindex(comunes) for k, v in pred.items()})
    P['Observado'] = yc
    P.to_csv(f'{salida}/predicciones_prueba_h{h}.csv')
    print(tabla[['Modelo', 'MAE', 'RMSE', 'nRMSE_pct', 'MBE', 'R2', 'habilidad_pct']].round(3).to_string(), flush=True)
    return tabla, hiper, P, dict(n_entrenamiento=int(m_tr.sum()), n_validacion=int(m_va.sum()), n_prueba=int(len(comunes)))


t0 = time.time()
resultados = {}
for h in ci['horizontes_h']:
    tabla, hiper, P, n = evaluar(h)
    resultados[h] = (tabla, P)
    registro[f'h{h}'] = dict(muestras_diurnas=n, hiperparametros=hiper, metricas_prueba=tabla.to_dict(orient='records'))
registro['tiempo_s'] = round(time.time() - t0)
guardar_json(registro, f'{salida}/prediccion_irradiancia.json')

# --------------------------------------------------------------------------- figura
mejor = {h: next(m for m in resultados[h][0].Modelo if not m.startswith(('Persistencia', 'Climatología'))) for h in resultados}
fig = plt.figure(figsize=(11, 8.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8])
for j, h in enumerate(list(resultados)[:2]):
    tabla, P = resultados[h]
    r = tabla.set_index('Modelo').loc[mejor[h]]
    ax = fig.add_subplot(gs[0, j])
    ax.scatter(P.Observado, P[mejor[h]], s=3, alpha=0.15)
    ax.plot([0, 1200], [0, 1200], 'k--', lw=1)
    ax.set_title(f"({'ab'[j]}) {mejor[h]}, h = {h} h\nRMSE = {coma(r.RMSE)} W/m²; R² = {coma(r.R2, 3)}; s = {coma(r.habilidad_pct)} %", fontsize=11)
    ax.set_xlabel('GHI observada (W/m²)')
    ax.set_ylabel('GHI predicha (W/m²)')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 1200)
# detalle de una semana (primer lunes del tercer mes del periodo de prueba)
P1 = resultados[ci['horizontes_h'][0]][1]
ini = (P1.index[0].tz_localize(None) + pd.offsets.MonthBegin(2)).normalize()
ini = ini + pd.Timedelta(days=(7 - ini.dayofweek) % 7)
semana = pd.date_range(ini, ini + pd.Timedelta(hours=167), freq='h', tz=TZ)
ax = fig.add_subplot(gs[1, :])
ax.plot(semana.tz_localize(None), df.ghi.reindex(semana).values, 'k-', lw=1.6, label='Observada (NASA POWER)')
for h, estilo in zip(resultados, ['-', '--']):
    P = resultados[h][1]
    ax.plot(semana.tz_localize(None), P[mejor[h]].reindex(semana).values, estilo, lw=1.2, label=f'{mejor[h]}, h = {h} h')
ax.plot(semana.tz_localize(None), P1['Persistencia inteligente'].reindex(semana).values, ':', lw=1.2,
        label=f"Persistencia inteligente, h = {ci['horizontes_h'][0]} h")
ax.set_title('(c) Detalle de una semana del periodo de prueba', fontsize=11)
ax.set_ylabel('GHI (W/m²)')
ax.set_ylim(0, 800)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{mdates.num2date(v).day} {MESES_ES[mdates.num2date(v).month - 1]}.'))
plt.tight_layout()
plt.savefig(f'{salida}/fig_prediccion_irradiancia.png', dpi=300)
plt.close()
