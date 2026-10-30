"""Funciones compartidas por los scripts del flujo de análisis.

Contiene la lectura de la configuración y de los archivos de NASA POWER, las métricas de error,
los intervalos de confianza por remuestreo y utilidades de formato para las figuras.
"""
import json
import os

import numpy as np
import pandas as pd

# Raíz del repositorio (carpeta que contiene config/, datos/, src/ y resultados/)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MESES_ES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']


def cargar_config(ruta=None):
    """Lee config/parametros.json y devuelve un diccionario."""
    ruta = ruta or os.path.join(RAIZ, 'config', 'parametros.json')
    with open(ruta, encoding='utf-8') as f:
        return json.load(f)


def ruta(relativa):
    """Convierte una ruta relativa a la raíz del repositorio en una ruta absoluta."""
    return os.path.join(RAIZ, relativa)


def carpeta_resultados(nombre):
    """Crea (si no existe) y devuelve la carpeta resultados/<nombre>."""
    p = os.path.join(RAIZ, 'resultados', nombre)
    os.makedirs(p, exist_ok=True)
    return p


def leer_nasa_power(archivo, zona_horaria):
    """Lee un CSV horario de NASA POWER (formato 'Point', hora solar local LST).

    El encabezado termina en la línea '-END HEADER-'. El índice resultante corresponde al inicio de
    cada hora; el valor horario representa la media de la hora. El código -999 (dato faltante) se
    reemplaza por NaN.
    """
    with open(archivo, encoding='utf-8', errors='ignore') as f:
        lineas = f.read().splitlines()
    k = lineas.index('-END HEADER-')
    d = pd.read_csv(archivo, skiprows=k + 1)
    d.index = pd.to_datetime(dict(year=d.YEAR, month=d.MO, day=d.DY, hour=d.HR)).dt.tz_localize(zona_horaria)
    return d.drop(columns=['YEAR', 'MO', 'DY', 'HR']).replace(-999, np.nan)


def metricas(obs, pred, ref=None):
    """Métricas de error de un pronóstico.

    MAE, RMSE, nRMSE (RMSE dividido por la media observada, en %), MBE (positivo si el modelo
    sobrestima) y R². Si se entrega la predicción de referencia, se calcula además el índice de
    habilidad s = 1 − RMSE/RMSE_ref (en %).
    """
    y = np.asarray(obs, float)
    p = np.asarray(pred, float)
    e = p - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    out = dict(MAE=float(np.mean(np.abs(e))), RMSE=rmse, nRMSE_pct=100 * rmse / float(np.mean(y)),
               MBE=float(np.mean(e)), R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))
    if ref is not None:
        rmse_ref = float(np.sqrt(np.mean((np.asarray(ref, float) - y) ** 2)))
        out['habilidad_pct'] = 100 * (1 - rmse / rmse_ref)
    return out


def metricas_validacion(real, modelo):
    """Comparación entre una serie medida y una modelada (sesgo, error y correlación)."""
    e = modelo - real
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return dict(n=int(len(real)), media_real=float(real.mean()), media_modelo=float(modelo.mean()),
                MBE=float(e.mean()), nMBE_pct=100 * float(e.mean()) / float(real.mean()),
                MAE=float(e.abs().mean()), RMSE=rmse, nRMSE_pct=100 * rmse / float(real.mean()),
                r=float(np.corrcoef(real, modelo)[0, 1]))


def ic_rmse_bloques(obs, pred, bloque=7, n=1000, semilla=42):
    """Intervalo de confianza del 95 % del RMSE por remuestreo de bloques consecutivos.

    Se usa en series diarias para conservar la autocorrelación dentro de cada bloque.
    """
    rng = np.random.default_rng(semilla)
    y = np.asarray(obs, float)
    p = np.asarray(pred, float)
    N = len(y)
    nb = int(np.ceil(N / bloque))
    valores = []
    for _ in range(n):
        inicios = rng.integers(0, N - bloque + 1, nb)
        idx = np.concatenate([np.arange(a, a + bloque) for a in inicios])[:N]
        valores.append(np.sqrt(np.mean((p[idx] - y[idx]) ** 2)))
    return [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]


def ic_rmse_dias(obs, pred, n=500, semilla=42):
    """Intervalo de confianza del 95 % del RMSE por remuestreo de días completos (series horarias)."""
    rng = np.random.default_rng(semilla)
    d = pd.DataFrame({'e2': (np.asarray(pred, float) - np.asarray(obs, float)) ** 2}, index=obs.index)
    g = d.groupby(d.index.date).e2.agg(['sum', 'count'])
    S, C = g['sum'].values, g['count'].values
    N = len(g)
    valores = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        valores.append(np.sqrt(S[i].sum() / C[i].sum()))
    return [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]


def guardar_json(obj, archivo):
    """Guarda un diccionario como JSON legible (UTF-8)."""
    with open(archivo, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def coma(x, d=1):
    """Número con coma decimal (para títulos de figuras en español)."""
    return f'{x:.{d}f}'.replace('.', ',')
