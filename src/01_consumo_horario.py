"""Paso 1. Extracción de la serie horaria de consumo.

El archivo del operador de red contiene una hoja por año con bloques mensuales; cada fila es un día y
las columnas HORA 1 … HORA 24 contienen la energía activa (kWh) de cada hora. HORA 1 se asigna al
intervalo 00:00–01:00 de la hora local.

Salida: resultados/consumo/consumo_horario.csv (índice horario, columna kWh).
"""
import pandas as pd

from comun import cargar_config, ruta, carpeta_resultados

cfg = cargar_config()
archivo = ruta(cfg['archivos']['consumo_xlsx'])
filas = []
for anio in cfg['consumo']['anios_horarios']:
    hoja = pd.read_excel(archivo, sheet_name=str(anio), header=None)
    for _, fila in hoja.iterrows():
        # Solo las filas cuya primera celda es una fecha contienen datos horarios
        if isinstance(fila[0], pd.Timestamp) or (hasattr(fila[0], 'year') and not isinstance(fila[0], str)):
            filas.append([pd.Timestamp(fila[0])] + list(pd.to_numeric(fila[1:25], errors='coerce').values))

h = pd.DataFrame(filas).set_index(0)
h.columns = range(24)
h = h[~h.index.duplicated()].sort_index()   # se descartan fechas repetidas
h.index = pd.DatetimeIndex(h.index)

serie = h.stack()
serie.index = [dia + pd.Timedelta(hours=hora) for dia, hora in serie.index]
serie = serie.rename('kWh')
serie.index.name = 'fecha_hora'
salida = carpeta_resultados('consumo')
serie.to_frame().to_csv(f'{salida}/consumo_horario.csv')
print(f'{len(serie)} horas, de {serie.index.min()} a {serie.index.max()}')
