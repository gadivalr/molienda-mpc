#!/usr/bin/env bash
# Reproduce todas las figuras y métricas del proyecto.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

echo "== Pruebas de cordura =="
$PY -m unittest discover -s tests -v

echo "== Modelo y estado estacionario =="
$PY src/model.py

echo "== Sensor virtual de P80 =="
$PY src/soft_sensor.py

echo "== Simulación, control y figuras =="
$PY src/run_simulation.py

echo "Listo. Figuras en figures/, métricas en reports/."
