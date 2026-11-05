# Predicción de consumo e irradiancia y evaluación técnico-económica de un sistema fotovoltaico en un campus universitario

Este repositorio contiene los datos, el código y los resultados de un flujo de análisis reproducible para:

1. Predecir el consumo eléctrico diario de una sede universitaria con modelos estadísticos, de aprendizaje automático y redes LSTM.
2. Predecir la irradiancia global horizontal (GHI) horaria con horizontes de 1 h y 24 h.
3. Validar la irradiancia satelital con una estación local y modelar un generador fotovoltaico calibrado con la producción medida de un sistema existente.
4. Evaluar económicamente una ampliación fotovoltaica con un balance horario entre generación y consumo.

El caso de estudio es una sede universitaria ubicada en Pasto (Colombia), a 2.553 m s. n. m. Todos los parámetros del sitio, del sistema y de la evaluación económica están en `config/parametros.json`. Para aplicar el flujo a otro caso basta con reemplazar los datos y ajustar ese archivo.

## Estructura

```
config/parametros.json          Parámetros del sitio, de los modelos y de la evaluación económica
datos/                          Datos de entrada (ver la tabla de datos)
src/comun.py                    Funciones compartidas: lectura de datos, métricas e intervalos de confianza
src/01_consumo_horario.py       Extracción de la serie horaria de consumo
src/02_validacion_irradiancia.py  Comparación de la GHI satelital con la estación local
src/03_modelo_fv.py             Modelo fotovoltaico, validación con la producción medida y año meteorológico típico
src/04_prediccion_consumo.py    Predicción del consumo diario (horizonte de un día)
src/05_prediccion_irradiancia.py  Predicción de la GHI horaria (horizontes de 1 h y 24 h)
src/06_evaluacion_economica.py  Balance horario, VPN, TIR, retornos, LCOE, sensibilidad y dimensionamiento
ejecutar_todo.py                Ejecuta los pasos en orden
resultados/                     Tablas (CSV y JSON) y figuras (PNG, 300 ppp) generadas por el flujo
```

## Datos

| Archivo | Contenido | Periodo | Resolución |
|---|---|---|---|
| `consumo_medidor.xlsx` | Energía activa del medidor de la sede (hojas anuales con columnas HORA 1–24 y hoja `Dias` con totales diarios) | 2019-01-01 a 2025-04-30 | Horaria hasta 2024 y diaria hasta 2025-04-30 |
| `nasa_power_irradiancia.csv` | GHI, DNI y DHI de NASA POWER (CERES SYN1deg), punto 1,1981° N, 77,2781° O, hora solar local | 2015-01-01 a 2025-09-09 (sin datos de irradiancia desde 2025-05-31) | Horaria |
| `nasa_power_meteorologia.csv` | Temperatura, humedad y viento de NASA POWER (MERRA-2) | 2015-01-01 a 2025-09-09 | Horaria |
| `estacion_meteorologica.xlsx` | Irradiancia y variables meteorológicas de la estación local | 2025-02-03 a 2025-05-27 | ~2 min |
| `inversor_sistema_existente.csv` | Potencia AC del sistema fotovoltaico existente (15 kWp) | 2025-02-05 a 2025-10-17 | ~2 min |

Consideraciones sobre los datos:

* En la hoja `Dias`, el año 2023 aparece duplicado, por lo que el código elimina las fechas repetidas.
* Desde el 5 de mayo de 2024, el sistema fotovoltaico existente opera detrás del medidor, por lo que este registra la importación neta. El paso 4 reconstruye el consumo bruto sumando la energía diaria del sistema existente: la medida por el inversor cuando existe y la modelada y calibrada en el resto del periodo.
* Las componentes DNI y DHI de NASA POWER no cierran con la GHI, por lo que el modelo fotovoltaico obtiene ambas componentes de la GHI con el modelo de Erbs.

## Protocolo de evaluación de los pronósticos

* **Partición temporal sin mezcla:**
  * entrenamiento hasta 2022-12-31
  * validación durante 2023
  * prueba desde 2024-01-01
* **Selección de modelos:** los hiperparámetros se eligen solo con el error en validación. El modelo elegido se reentrena con entrenamiento y validación y se evalúa una sola vez en prueba.
* **Origen móvil:** cada pronóstico usa únicamente información disponible en el instante de emisión.
* **Referencias:**
  * consumo: persistencia (t − 1) y persistencia semanal (t − 7)
  * irradiancia: persistencia, persistencia de 24 h, climatología mes-hora y persistencia inteligente
* **Métricas:** MAE, RMSE, nRMSE (RMSE/media observada), MBE, R² e índice de habilidad s = 1 − RMSE/RMSE_ref. Las de irradiancia se calculan solo en horas diurnas (ángulo cenital < 85°).
* **Intervalos de confianza del 95 % del RMSE:** remuestreo por bloques de 7 días para el consumo y por días completos para la irradiancia.
* **Redes LSTM:** se entrenan con tres semillas y se reporta el promedio de sus predicciones.

## Instalación y ejecución

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
python ejecutar_todo.py
```

Las redes LSTM usan Keras con el motor JAX (`KERAS_BACKEND=jax`, valor por defecto en los scripts). Los pasos también pueden ejecutarse por separado, en orden, por ejemplo `python src/03_modelo_fv.py`. El paso 4 acepta el argumento `neto` para evaluar la serie del medidor sin reconstruir el consumo bruto.

Versiones con las que se generaron los resultados incluidos: Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, statsmodels 0.15, pvlib 0.16, Keras 3.15, JAX 0.12, holidays 0.107 y matplotlib 3.11. Con otras versiones, los modelos con componentes aleatorios pueden dar diferencias menores.

## Supuestos de la evaluación económica

Los valores están en `config/parametros.json`:

* **Tarifa:** costo unitario de prestación del servicio y componente de generación, promedios ponderados de tres facturas del operador de red, con contribución del 20 %.
* **Precio de bolsa:** promedio mensual de 2025 publicado por el operador del mercado.
* **Inversión:** costo de referencia de 867 USD/kW y TRM de 3.158,43 COP/USD.
* **Operación y mantenimiento:** 1 % anual de la inversión.
* **Vida útil y operación:** reemplazo de inversores en el año 13, degradación de 0,5 % anual, vida útil de 25 años y tasa de descuento real del 10 %.
* **Sensibilidad:** precio de bolsa, inversión, TRM, tasa de descuento, costo evitado, valor de los créditos de energía, operación y mantenimiento, degradación anual y escalamiento real de precios.

## Licencias

* Código: MIT (archivo `LICENSE`).
* Datos y resultados: Creative Commons Atribución 4.0 Internacional (CC BY 4.0).

## Versiones

* **1.2:** se añade la degradación anual de los módulos al análisis de sensibilidad y se unifica el formato de las fechas en las figuras.
* **1.1:** primera versión completa del flujo, con datos, código y resultados.

## Cómo citar

Si usa este material, cite el artículo asociado y este repositorio. El DOI siguiente identifica todas las versiones y remite siempre a la más reciente:

> Repositorio *prediccion-fv-campus*. Zenodo. https://doi.org/10.5281/zenodo.23810715

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23810715.svg)](https://doi.org/10.5281/zenodo.23810715)
