"""Generación de datos, escenarios y ruido de medición."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from model import GrindingCircuit, PlantParams


# ----------------------------------------------------------------------------
# Ruido de medición (instrumentos típicos de planta)
# ----------------------------------------------------------------------------
MEAS_SIGMA = {
    "p80": 1.5,          # um (analizador de tamaño)
    "mill_load": 2.0,    # t
    "density_cf": 0.004, # fracción másica
    "power_kw": 25.0,    # kW
    "dp_kpa": 1.5,       # kPa
    "qf": 1.5,           # t/h
    "qw_mill": 1.0,      # m3/h
    "qw_dil": 1.5,       # m3/h
    "qs_cf": 8.0,        # t/h
    "circulating_load": 0.08,
}


def add_noise(values: Dict[str, float], rng: np.random.Generator | None = None,
              scale: float = 1.0) -> Dict[str, float]:
    rng = rng or np.random.default_rng()
    out = dict(values)
    for k, s in MEAS_SIGMA.items():
        if k in out:
            out[k] = float(out[k] + scale * s * rng.standard_normal())
    return out


# ----------------------------------------------------------------------------
# Excitación tipo PRBS multi-entrada
# ----------------------------------------------------------------------------
def make_excitation(
    n_steps: int,
    p: PlantParams,
    seed: int = 7,
    step_min: int = 90,
    step_max: int = 300,
    du_frac: Tuple[float, float, float] = (0.12, 0.18, 0.20),
    hardness_frac: float = 0.15,
    f80_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera entradas y disturbios por tramos aleatorios.

    Returns
    -------
    u : (n, 3)
    hardness : (n,)
    f80_factor : (n,)
    """
    rng = np.random.default_rng(seed)
    u0 = np.array([p.qf_nom, p.qw_mill_nom, p.qw_dil_nom])
    u = np.zeros((n_steps, 3))
    hardness = np.ones(n_steps)
    f80 = np.ones(n_steps)

    k = 0
    u_cur = u0.copy()
    h_cur, f_cur = 1.0, 1.0
    while k < n_steps:
        dwell = int(rng.integers(step_min, step_max + 1))
        dwell = min(dwell, n_steps - k)
        u[k : k + dwell] = u_cur
        hardness[k : k + dwell] = h_cur
        f80[k : k + dwell] = f_cur

        # nuevos valores para el siguiente tramo
        u_cur = u0 * (1.0 + rng.uniform(-1, 1, 3) * np.array(du_frac))
        h_cur = 1.0 + rng.uniform(-hardness_frac, hardness_frac)
        f_cur = 1.0 + rng.uniform(-f80_frac, f80_frac)
        k += dwell

    return u, hardness, f80


# ----------------------------------------------------------------------------
# Simulación
# ----------------------------------------------------------------------------
def simulate(
    plant: GrindingCircuit,
    u: np.ndarray,
    hardness: np.ndarray | None = None,
    f80_factor: np.ndarray | None = None,
    noise: bool = False,
    seed: int = 1,
    record_state: bool = False,
) -> pd.DataFrame:
    """Simula una secuencia de entradas de 1 minuto."""
    n = len(u)
    hardness = np.ones(n) if hardness is None else hardness
    f80_factor = np.ones(n) if f80_factor is None else f80_factor
    rng = np.random.default_rng(seed)

    rows = []
    for k in range(n):
        plant.p.feed_size_factor = float(f80_factor[k])
        out = plant.step(u[k], hardness=float(hardness[k]))
        row = {
            "t_min": k,
            "qf": float(u[k, 0]),
            "qw_mill": float(u[k, 1]),
            "qw_dil": float(u[k, 2]),
            "hardness": float(hardness[k]),
            "f80_factor": float(f80_factor[k]),
        }
        row.update(out)
        if noise:
            meas = add_noise(out, rng)
            row.update({f"meas_{k}": v for k, v in meas.items()})
        if record_state:
            for i, v in enumerate(plant.state):
                row[f"x{i}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def steady_state(p: PlantParams | None = None, max_hours: float = 800.0):
    plant = GrindingCircuit(p)
    out = plant.settle(max_hours=max_hours)
    return plant, out


# ----------------------------------------------------------------------------
# Escenarios de control
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    n_steps: int
    sp: np.ndarray              # (n, 3) setpoints [p80, mill_load, density]
    hardness: np.ndarray
    f80_factor: np.ndarray
    u0: np.ndarray


def scenario_setpoints(p: PlantParams, n_steps: int = 900) -> Scenario:
    """Cambios de setpoint de P80, carga y densidad."""
    t = np.arange(n_steps)
    sp = np.tile([120.0, 280.0, 0.665], (n_steps, 1))
    sp[t >= 120, 0] = 110.0      # P80 más fino a las 2 h
    sp[t >= 480, 1] = 300.0      # más carga a las 8 h
    sp[t >= 720, 2] = 0.645      # más dilución a las 12 h
    return Scenario(
        name="setpoints",
        n_steps=n_steps,
        sp=sp,
        hardness=np.ones(n_steps),
        f80_factor=np.ones(n_steps),
        u0=np.array([p.qf_nom, p.qw_mill_nom, p.qw_dil_nom]),
    )


def scenario_disturbance(p: PlantParams, n_steps: int = 900) -> Scenario:
    """Mineral más duro y alimentación más gruesa."""
    t = np.arange(n_steps)
    sp = np.tile([120.0, 280.0, 0.665], (n_steps, 1))
    hardness = np.ones(n_steps)
    f80 = np.ones(n_steps)
    hardness[t >= 120] = 0.82    # 18% más duro
    f80[t >= 600] = 1.15         # 15% más grueso
    return Scenario(
        name="disturbance",
        n_steps=n_steps,
        sp=sp,
        hardness=hardness,
        f80_factor=f80,
        u0=np.array([p.qf_nom, p.qw_mill_nom, p.qw_dil_nom]),
    )


def scenario_throughput(p: PlantParams, n_steps: int = 900) -> Scenario:
    """Aumento de throughput con restricción de P80."""
    t = np.arange(n_steps)
    sp = np.tile([120.0, 280.0, 0.665], (n_steps, 1))
    # no cambia setpoint; se usa el término económico del MPC
    return Scenario(
        name="throughput",
        n_steps=n_steps,
        sp=sp,
        hardness=np.ones(n_steps),
        f80_factor=np.ones(n_steps),
        u0=np.array([p.qf_nom, p.qw_mill_nom, p.qw_dil_nom]),
    )
