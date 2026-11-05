"""Paso 4. Predicción del consumo eléctrico diario con horizonte de un día.

Protocolo de evaluación:
  * Partición temporal: entrenamiento hasta fin_entrenamiento, validación hasta fin_validacion y prueba
    desde el día siguiente hasta el final de la serie.
  * Pronóstico con origen móvil: cada predicción usa solo información disponible el día anterior.
  * Los hiperparámetros se eligen con el error en validación; el modelo elegido se reentrena con
    entrenamiento + validación y se evalúa una sola vez en prueba.
  * Referencias: persistencia (t − 1) y persistencia semanal (t − 7).

Preparación de la serie:
  * Se eliminan fechas duplicadas, los valores no positivos se tratan como faltantes y se interpolan
    huecos de hasta tres días.
  * Desde la fecha en que el generador fotovoltaico existente quedó detrás del medidor, el medidor
    registra la importación neta; se reconstruye el consumo bruto sumando la energía diaria de ese
    generador (resultado del paso 3).

Uso: python src/04_prediccion_consumo.py [neto]
     Con el argumento 'neto' se evalúa la serie tal como la registra el medidor, sin reconstrucción.
Salidas en resultados/consumo/ (o resultados/consumo_neto/).
"""
import os
import sys
import time
import warnings

os.environ.setdefault('KERAS_BACKEND', 'jax')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import holidays
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from comun import cargar_config, ruta, carpeta_resultados, metricas, ic_rmse_bloques, guardar_json, MESES_ES

cfg = cargar_config()
c = cfg['consumo']
NETO = len(sys.argv) > 1 and sys.argv[1] == 'neto'
salida = carpeta_resultados('consumo_neto' if NETO else 'consumo')
SEMILLA = c['semilla']
FIN_ENT, FIN_VAL = cfg['particion']['fin_entrenamiento'], cfg['particion']['fin_validacion']
registro = {}

# --------------------------------------------------------------------------- serie diaria
crudo = pd.read_excel(ruta(cfg['archivos']['consumo_xlsx']), sheet_name=c['hoja_diaria'])
crudo[c['columna_fecha']] = pd.to_datetime(crudo[c['columna_fecha']])
duplicados = int(crudo.duplicated(c['columna_fecha']).sum())
d = crudo.drop_duplicates(c['columna_fecha']).set_index(c['columna_fecha']).sort_index()
s = pd.to_numeric(d[c['columna_energia']], errors='coerce')
no_positivos = int((s <= 0).sum())
s[s <= 0] = np.nan
calendario = pd.date_range(s.index.min(), s.index.max(), freq='D')
faltantes = int(len(calendario.difference(s.index)) + s.isna().sum())
s = s.reindex(calendario).interpolate(limit=3)

INICIO_NETO = c['inicio_generacion_detras_del_medidor']
if not NETO:
    fv = pd.read_csv(f"{carpeta_resultados('fv')}/energia_diaria_existente.csv", index_col=0, parse_dates=True)['E_usada_kWh']
    aporte = fv.reindex(s.index).where(s.index >= pd.Timestamp(INICIO_NETO)).fillna(0.0)
    s = s + aporte
    registro['reconstruccion_consumo_bruto'] = dict(desde=INICIO_NETO, energia_fv_media_kWh_dia=round(float(aporte[aporte > 0].mean()), 2))
registro['datos'] = dict(filas_originales=int(len(crudo)), fechas_duplicadas_eliminadas=duplicados,
                         valores_no_positivos=no_positivos, dias_faltantes_interpolados=faltantes,
                         inicio=str(s.index.min().date()), fin=str(s.index.max().date()), n_dias=int(len(s)),
                         media_kWh_dia=round(float(s.mean()), 2))
print(registro['datos'])

# --------------------------------------------------------------------------- variables de entrada
festivos = holidays.Colombia(years=range(s.index.min().year, s.index.max().year + 2))


def variables(serie):
    """Rezagos, estadísticos móviles (solo del pasado) y calendario del día objetivo."""
    X = pd.DataFrame(index=serie.index)
    for L in [1, 2, 7, 14]:
        X[f'rezago_{L}'] = serie.shift(L)
    for V in [7, 14]:
        X[f'media_{V}'] = serie.shift(1).rolling(V).mean()
    X['desv_7'] = serie.shift(1).rolling(7).std()
    dia = serie.index.dayofweek
    for k in range(7):
        X[f'dia_{k}'] = (dia == k).astype(float)
    mes = serie.index.month
    X['sen_mes'] = np.sin(2 * np.pi * mes / 12)
    X['cos_mes'] = np.cos(2 * np.pi * mes / 12)
    X['festivo'] = [1.0 if t in festivos else 0.0 for t in serie.index]
    X['festivo_siguiente'] = [1.0 if (t + pd.Timedelta(days=1)) in festivos else 0.0 for t in serie.index]
    return X


X_all = variables(s)
ok = X_all.notna().all(axis=1)
X_all, y_all = X_all[ok], s[ok]
Xtr, ytr = X_all[X_all.index <= FIN_ENT], y_all[y_all.index <= FIN_ENT]
mval = (X_all.index > FIN_ENT) & (X_all.index <= FIN_VAL)
Xva, yva = X_all[mval], y_all[mval]
Xte, yte = X_all[X_all.index > FIN_VAL], y_all[y_all.index > FIN_VAL]
Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
registro['particion'] = dict(entrenamiento=[str(Xtr.index[0].date()), str(Xtr.index[-1].date()), len(Xtr)],
                             validacion=[str(Xva.index[0].date()), str(Xva.index[-1].date()), len(Xva)],
                             prueba=[str(Xte.index[0].date()), str(Xte.index[-1].date()), len(Xte)])
pred, hiper = {}, {}

# --------------------------------------------------------------------------- referencias
pred['Persistencia (t − 1)'] = s.shift(1).reindex(yte.index)
pred['Persistencia semanal (t − 7)'] = s.shift(7).reindex(yte.index)

# --------------------------------------------------------------------------- Holt-Winters
# Parámetros estimados con entrenamiento + validación; en prueba se reajusta el estado con cada
# nuevo dato manteniendo fijos los parámetros de suavizado (pronóstico a un paso).
hw = ExponentialSmoothing(s[:FIN_VAL], trend=None, seasonal='add', seasonal_periods=7).fit(optimized=True)
alfa, gamma = hw.params['smoothing_level'], hw.params['smoothing_seasonal']
hw_pred = []
for t in yte.index:
    m = ExponentialSmoothing(s[:t - pd.Timedelta(days=1)], trend=None, seasonal='add', seasonal_periods=7)
    hw_pred.append(float(m.fit(smoothing_level=alfa, smoothing_seasonal=gamma, optimized=False).forecast(1).iloc[0]))
pred['Holt-Winters'] = pd.Series(hw_pred, index=yte.index)
hiper['Holt-Winters'] = dict(alfa=round(float(alfa), 4), gamma=round(float(gamma), 4))


# --------------------------------------------------------------------------- ARIMA y SARIMA
def seleccionar_sarimax(ordenes, estacionales):
    """Ajusta con entrenamiento y elige el orden con menor RMSE de pronóstico a un paso en validación."""
    mejor = None
    for o in ordenes:
        for so in estacionales:
            try:
                r = SARIMAX(s[:FIN_ENT], order=o, seasonal_order=so, trend='c', enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
                rv = r.append(s[FIN_ENT:FIN_VAL].iloc[1:], refit=False)   # actualiza el estado, no los parámetros
                error = metricas(yva, rv.predict(start=yva.index[0], end=yva.index[-1]))['RMSE']
                if mejor is None or error < mejor[0]:
                    mejor = (error, o, so)
            except Exception:
                pass
    return mejor


t0 = time.time()
sel = {'ARIMA': seleccionar_sarimax([(p, dd, q) for p in range(4) for dd in range(2) for q in range(4)], [(0, 0, 0, 0)]),
       'SARIMA': seleccionar_sarimax([(1, 0, 1), (2, 0, 1), (1, 1, 1), (2, 1, 2), (3, 0, 3)],
                                     [(1, 0, 1, 7), (1, 1, 1, 7), (0, 1, 1, 7)])}
for nombre, (error, o, so) in sel.items():
    r = SARIMAX(s[:FIN_VAL], order=o, seasonal_order=so, trend='c', enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False)
    rt = r.append(s[FIN_VAL:].iloc[1:], refit=False)
    pred[nombre] = rt.predict(start=yte.index[0], end=yte.index[-1])
    hiper[nombre] = dict(orden=o, orden_estacional=so, RMSE_validacion=round(error, 2))
print('ARIMA/SARIMA', round(time.time() - t0), 's', hiper)


# --------------------------------------------------------------------------- aprendizaje automático
def ajustar(constructor, rejilla, nombre):
    """Búsqueda en rejilla con validación temporal y reentrenamiento final con entrenamiento + validación."""
    mejor = None
    for g in ParameterGrid(rejilla):
        error = metricas(yva, constructor(**g).fit(Xtr, ytr).predict(Xva))['RMSE']
        if mejor is None or error < mejor[0]:
            mejor = (error, g)
    modelo = constructor(**mejor[1]).fit(Xtv, ytv)
    pred[nombre] = pd.Series(modelo.predict(Xte), index=yte.index)
    hiper[nombre] = dict(mejor[1], RMSE_validacion=round(mejor[0], 2))


ajustar(lambda alpha: Pipeline([('esc', StandardScaler()), ('m', Ridge(alpha=alpha))]),
        {'alpha': [0.1, 1, 10, 100]}, 'Ridge')
ajustar(lambda alpha: Pipeline([('esc', StandardScaler()), ('pol', PolynomialFeatures(2, include_bias=False)),
                                ('esc2', StandardScaler()), ('m', Ridge(alpha=alpha))]),
        {'alpha': [1, 10, 100, 1000]}, 'Regresión polinomial (grado 2)')
ajustar(lambda C, epsilon, gamma: Pipeline([('esc', StandardScaler()), ('m', SVR(C=C, epsilon=epsilon, gamma=gamma))]),
        {'C': [10, 50, 100, 300], 'epsilon': [1, 5, 10], 'gamma': ['scale', 0.01]}, 'SVR')
ajustar(lambda n_estimators, max_depth, min_samples_leaf: RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEMILLA, n_jobs=-1),
        {'n_estimators': [300], 'max_depth': [None, 10], 'min_samples_leaf': [1, 5]}, 'Random Forest')
ajustar(lambda learning_rate, max_depth: HistGradientBoostingRegressor(learning_rate=learning_rate, max_depth=max_depth,
                                                                      random_state=SEMILLA),
        {'learning_rate': [0.03, 0.1], 'max_depth': [3, None]}, 'Gradient Boosting')

# --------------------------------------------------------------------------- LSTM
import keras
from keras import layers

keras.utils.set_random_seed(SEMILLA)
mu, sd = float(s[:FIN_ENT].mean()), float(s[:FIN_ENT].std())
cols_cal = [k for k in X_all.columns if k.startswith('dia_') or k in ('sen_mes', 'cos_mes', 'festivo', 'festivo_siguiente')]
cal = variables(s)[cols_cal]


def ventanas(w, indices):
    """Secuencias de w días: consumo normalizado y calendario, terminando en el día objetivo."""
    z, C, idx = ((s - mu) / sd).values, cal.values, s.index
    X, y, t = [], [], []
    for i in range(w, len(z)):
        if idx[i] in indices:
            X.append(np.column_stack([z[i - w:i], C[i - w + 1:i + 1]]))
            y.append(z[i])
            t.append(idx[i])
    return np.array(X, dtype='float32'), np.array(y, dtype='float32'), pd.DatetimeIndex(t)


def red(w, n_var, unidades):
    m = keras.Sequential([layers.Input((w, n_var)), layers.LSTM(unidades), layers.Dropout(0.2),
                          layers.Dense(32, activation='relu'), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mae')
    return m


mejor = None
for w in [7, 14, 20, 28]:
    for unidades in [32, 64]:
        Xa, ya, _ = ventanas(w, ytr.index)
        Xv, yv, _ = ventanas(w, yva.index)
        m = red(w, Xa.shape[2], unidades)
        parada = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        h = m.fit(Xa, ya, validation_data=(Xv, yv), epochs=200, batch_size=32, verbose=0, callbacks=[parada])
        error = metricas(yv * sd + mu, m.predict(Xv, verbose=0).ravel() * sd + mu)['RMSE']
        if mejor is None or error < mejor[0]:
            mejor = (error, w, unidades, int(np.argmin(h.history['val_loss']) + 1))
error_val, w, unidades, epocas = mejor
corridas = []
Xa, ya, _ = ventanas(w, ytv.index)
Xt, yt, tt = ventanas(w, yte.index)
for semilla in [SEMILLA, SEMILLA + 1, SEMILLA + 2]:      # promedio de tres semillas
    keras.utils.set_random_seed(semilla)
    m = red(w, Xa.shape[2], unidades)
    m.fit(Xa, ya, epochs=epocas, batch_size=32, verbose=0)
    corridas.append(m.predict(Xt, verbose=0).ravel() * sd + mu)
pred['LSTM'] = pd.Series(np.mean(corridas, axis=0), index=tt).reindex(yte.index)
hiper['LSTM'] = dict(ventana_dias=w, unidades=unidades, epocas=epocas, dropout=0.2, optimizador='Adam (0,001)', perdida='MAE',
                     RMSE_validacion=round(error_val, 2),
                     RMSE_prueba_por_semilla=[round(metricas(yte.reindex(tt), r)['RMSE'], 2) for r in corridas])

# --------------------------------------------------------------------------- evaluación en prueba
ref = pred['Persistencia (t − 1)']
filas = []
for nombre, p in pred.items():
    p = pd.Series(p).reindex(yte.index)
    m = metricas(yte, p, ref)
    m['IC95_RMSE'] = ic_rmse_bloques(yte.values, p.values, bloque=7, semilla=SEMILLA)
    filas.append(dict(Modelo=nombre, **m))
tabla = pd.DataFrame(filas).sort_values('RMSE').reset_index(drop=True)
tabla.to_csv(f'{salida}/metricas_prueba.csv', index=False)
pd.DataFrame({k: pd.Series(v).reindex(yte.index) for k, v in pred.items()}).assign(Observado=yte).to_csv(f'{salida}/predicciones_prueba.csv')
registro['hiperparametros'] = hiper
registro['metricas_prueba'] = tabla.to_dict(orient='records')
guardar_json(registro, f'{salida}/prediccion_consumo.json')
print(tabla[['Modelo', 'MAE', 'RMSE', 'nRMSE_pct', 'MBE', 'R2', 'habilidad_pct']].round(3).to_string())

# --------------------------------------------------------------------------- figura
mejor_modelo = next(n for n in tabla.Modelo if not n.startswith('Persistencia'))
pb = pd.Series(pred[mejor_modelo]).reindex(yte.index)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(yte, pb, s=8, alpha=0.6)
lo, hi = float(min(yte.min(), pb.min())), float(max(yte.max(), pb.max()))
ax[0].plot([lo, hi], [lo, hi], 'k--', lw=1)
ax[0].set_xlabel('Consumo observado (kWh/día)')
ax[0].set_ylabel('Consumo predicho (kWh/día)')
ax[0].set_title(f'(a) Observado frente a predicho – {mejor_modelo}', fontsize=10)
ax[0].grid(alpha=0.3)
# detalle de las cuatro primeras semanas completas del tercer mes del periodo de prueba
ini = (yte.index[0] + pd.offsets.MonthBegin(2)).normalize()
ini = ini + pd.Timedelta(days=(7 - ini.dayofweek) % 7)
det = yte[ini:ini + pd.Timedelta(days=27)]
ax[1].plot(det.index, det.values, 'k-o', ms=3, label='Observado')
ax[1].plot(det.index, pb.reindex(det.index), '-s', ms=3, label=mejor_modelo)
ax[1].plot(det.index, ref.reindex(det.index), ':', label='Persistencia (t − 1)')
ax[1].set_title('(b) Detalle de cuatro semanas', fontsize=10)
ax[1].set_ylabel('kWh/día')
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
ax[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{mdates.num2date(v).day} {MESES_ES[mdates.num2date(v).month - 1]}.'))
ax[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f'{salida}/fig_prediccion_consumo.png', dpi=300)
plt.close()
print('Mejor modelo:', mejor_modelo)
