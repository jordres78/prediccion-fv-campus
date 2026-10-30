"""Ejecuta el flujo completo en orden. Uso: python ejecutar_todo.py

Los pasos 4 y 5 entrenan redes LSTM y pueden tardar varios minutos.
"""
import os
import subprocess
import sys

PASOS = [
    ('01_consumo_horario.py', []),
    ('02_validacion_irradiancia.py', []),
    ('03_modelo_fv.py', []),
    ('04_prediccion_consumo.py', []),
    ('04_prediccion_consumo.py', ['neto']),
    ('05_prediccion_irradiancia.py', []),
    ('06_evaluacion_economica.py', []),
]
raiz = os.path.dirname(os.path.abspath(__file__))
entorno = dict(os.environ, KERAS_BACKEND=os.environ.get('KERAS_BACKEND', 'jax'))
for script, args in PASOS:
    print(f'\n=== {script} {" ".join(args)} ===', flush=True)
    subprocess.run([sys.executable, os.path.join(raiz, 'src', script), *args], check=True, env=entorno)
print('\nFlujo terminado. Resultados en la carpeta resultados/.')
