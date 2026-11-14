"""Sensor virtual de P80 a partir de mediciones rutinarias de planta.

La idea: el P80 se mide con un analizador de tamaño caro y lento. Con variables
disponibles (potencia del molino, presión del ciclón, densidades, flujos y carga)
se entrena un modelo de regresión para estimar el P80 en línea.

Se comparan tres familias:
    * PLS (clásico en procesos, interpretable)
    * Random Forest
    * Gradient Boosting (con intervalos de cuantiles)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model import GrindingCircuit, PlantParams
from simulation import make_excitation, simulate, steady_state


BASE_COLS = [
    "qf",
    "qw_mill",
    "qw_dil",
    "power_kw",
    "dp_kpa",
    "density_cf",
    "mill_load",
    "qs_cf",
    "circulating_load",
]

SIGMA = {
    "qf": 1.5,
    "qw_mill": 1.0,
    "qw_dil": 1.5,
    "power_kw": 25.0,
    "dp_kpa": 1.5,
    "density_cf": 0.004,
    "mill_load": 2.0,
    "qs_cf": 8.0,
    "circulating_load": 0.08,
}


def add_measurement_noise(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Agrega ruido, sesgo y deriva lenta a las mediciones."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    t = np.linspace(0.0, 1.0, n)
    for col in BASE_COLS:
        out[f"{col}_meas"] = out[col] + SIGMA[col] * rng.standard_normal(n)
    # deriva lenta de instrumentos (más realista que ruido puro)
    out["power_kw_meas"] += 0.02 * out["power_kw"].mean() * t
    out["density_cf_meas"] += 0.003 * t
    out["dp_kpa_meas"] += 0.01 * out["dp_kpa"].mean() * t
    return out


def build_features(df: pd.DataFrame, lags=(0, 2, 5, 10, 20)) -> Tuple[pd.DataFrame, pd.Series]:
    """Construye features con rezagos y variables derivadas."""
    out = pd.DataFrame(index=df.index)
    for col in BASE_COLS:
        c = f"{col}_meas"
        for lag in lags:
            out[f"{c}_lag{lag}"] = df[c].shift(lag)

    # variables derivadas típicas de operación
    out["power_per_t"] = df["power_kw_meas"] / df["qf_meas"].clip(lower=1e-6)
    out["water_total"] = df["qw_mill_meas"] + df["qw_dil_meas"]
    out["water_ratio"] = out["water_total"] / df["qf_meas"].clip(lower=1e-6)
    out["dil_ratio"] = df["qw_dil_meas"] / df["qf_meas"].clip(lower=1e-6)
    out["load_per_t"] = df["mill_load_meas"] / df["qf_meas"].clip(lower=1e-6)
    out["circ_x_power"] = df["circulating_load_meas"] * df["power_kw_meas"]
    out["dp_x_density"] = df["dp_kpa_meas"] * df["density_cf_meas"]

    y = df["p80"].copy()
    mask = out.notna().all(axis=1) & y.notna()
    return out.loc[mask], y.loc[mask]


LAGS = (0, 2, 5, 10, 20)


def feature_names(lags=LAGS) -> list[str]:
    """Nombres y orden exacto de las features usadas por el modelo."""
    names = [f"{c}_meas_lag{lag}" for c in BASE_COLS for lag in lags]
    names += [
        "power_per_t", "water_total", "water_ratio", "dil_ratio",
        "load_per_t", "circ_x_power", "dp_x_density",
    ]
    return names


class OnlineSoftSensor:
    """Sensor virtual en línea: mantiene el historial y entrega P80 ± banda."""

    def __init__(self, model, q80_um: float, lags=LAGS):
        from collections import deque

        self.model = model
        self.q80 = float(q80_um)
        self.lags = lags
        self.window = max(lags) + 1
        self.hist = {c: deque(maxlen=self.window) for c in BASE_COLS}
        self.ready = False

    def update(self, meas: Dict[str, float]):
        for c in BASE_COLS:
            v = float(meas[c])
            if not self.hist[c]:
                for _ in range(self.window):
                    self.hist[c].append(v)
            else:
                self.hist[c].append(v)

        row = []
        for c in BASE_COLS:
            h = list(self.hist[c])
            for lag in self.lags:
                row.append(h[-1 - lag])

        cur = {c: list(self.hist[c])[-1] for c in BASE_COLS}
        qf = max(cur["qf"], 1e-6)
        water_total = cur["qw_mill"] + cur["qw_dil"]
        row += [
            cur["power_kw"] / qf,
            water_total,
            water_total / qf,
            cur["qw_dil"] / qf,
            cur["mill_load"] / qf,
            cur["circulating_load"] * cur["power_kw"],
            cur["dp_kpa"] * cur["density_cf"],
        ]
        X = pd.DataFrame([row], columns=feature_names(self.lags))
        pred = float(np.ravel(self.model.predict(X))[0])
        self.ready = True
        return pred, pred - self.q80, pred + self.q80


def generate_dataset(
    n_steps: int = 22000,
    seed: int = 21,
    noise_seed: int = 99,
    settle_hours: float = 800.0,
) -> pd.DataFrame:
    p = PlantParams()
    plant, _ = steady_state(p, settle_hours)
    u, h, f = make_excitation(
        n_steps, p, seed=seed, step_min=60, step_max=240,
        du_frac=(0.15, 0.22, 0.24), hardness_frac=0.20, f80_frac=0.12,
    )
    df = simulate(plant, u, h, f, noise=False, seed=seed)
    return add_measurement_noise(df, seed=noise_seed)


def metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
    }


def train_models(X_train, y_train, X_test, y_test):
    models = {
        "PLS": Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=8)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=300, max_depth=12, min_samples_leaf=4,
                n_jobs=-1, random_state=0,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingRegressor(
                n_estimators=500, max_depth=3, learning_rate=0.04,
                subsample=0.8, random_state=0,
            )),
        ]),
    }

    results = {}
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = metrics(y_test, pred)
        preds[name] = pred
    return models, results, preds


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    fig_dir = root / "figures"
    report_dir = root / "reports"
    for d in (data_dir, fig_dir, report_dir):
        d.mkdir(exist_ok=True)

    print("Generando datos...")
    df = generate_dataset()
    df.to_csv(data_dir / "soft_sensor_data.csv", index=False)

    X, y = build_features(df)
    print("features:", X.shape, "target:", y.shape)

    # split temporal 70/30
    n = len(X)
    n_train = int(0.7 * n)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    print("Entrenando modelos...")
    models, results, preds = train_models(X_train, y_train, X_test, y_test)
    for name, m in results.items():
        print(name, {k: round(v, 4) for k, v in m.items()})

    best_name = max(results, key=lambda k: results[k]["R2"])
    print("Mejor modelo:", best_name)

    # Calibración conformal: se reserva un 20% del train para estimar el
    # cuantil 80% de los residuos absolutos. El intervalo resultante tiene
    # cobertura marginal ~80% sin suponer normalidad.
    from sklearn.base import clone
    n_cal = int(0.2 * n_train)
    X_fit, X_cal = X_train.iloc[:-n_cal], X_train.iloc[-n_cal:]
    y_fit, y_cal = y_train.iloc[:-n_cal], y_train.iloc[-n_cal:]
    final_model = clone(models[best_name])
    final_model.fit(X_fit, y_fit)

    cal_res = y_cal.to_numpy() - final_model.predict(X_cal)
    q80 = float(np.quantile(np.abs(cal_res), 0.80))
    pred_test = final_model.predict(X_test)
    lo = pred_test - q80
    hi = pred_test + q80
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    width = float(np.mean(hi - lo))
    final_metrics = metrics(y_test, pred_test)
    print("Modelo final:", {k: round(v, 4) for k, v in final_metrics.items()})
    print(f"Intervalo conformal 80%: cobertura={coverage:.3f}, ancho medio={width:.2f} um")

    # guardar artefactos
    joblib.dump(final_model, root / "data" / "soft_sensor_model.joblib")
    with open(report_dir / "soft_sensor_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "models": results,
            "best": best_name,
            "final_model": final_metrics,
            "conformal_q80_um": q80,
            "interval_80_coverage": coverage,
            "interval_80_width_um": width,
            "n_train": int(n_train),
            "n_calibration": int(n_cal),
            "n_test": int(n - n_train),
            "n_features": int(X.shape[1]),
        }, f, indent=2, ensure_ascii=False)

    # gráficos
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # 1) serie temporal (ventana del test)
    idx = np.arange(len(y_test))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(idx, y_test.to_numpy(), color="black", lw=1.4, label="P80 real")
    ax.plot(idx, pred_test, color="#0F766E", lw=1.2, label=f"{best_name} (estimación)")
    ax.fill_between(idx, lo, hi, color="#0F766E", alpha=0.15, label="Intervalo 80%")
    ax.set_xlim(0, min(1500, len(idx)))
    ax.set_xlabel("Muestra (1 min)")
    ax.set_ylabel("P80 (µm)")
    ax.set_title("Sensor virtual de P80 — validación fuera de muestra")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_soft_sensor_timeseries.png")
    plt.close(fig)

    # 2) paridad
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(y_test, pred_test, s=5, alpha=0.25, color="#0F766E")
    lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("P80 real (µm)")
    ax.set_ylabel("P80 estimado (µm)")
    ax.set_title(f"Paridad — R²={final_metrics['R2']:.3f}")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_soft_sensor_parity.png")
    plt.close(fig)

    # 3) residuos
    res = y_test.to_numpy() - pred_test
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(res, bins=50, color="#0F766E", alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Residuo (µm)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Residuos — RMSE={final_metrics['RMSE']:.2f} µm")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_soft_sensor_residuals.png")
    plt.close(fig)

    # 4) importancia por permutación (aplicable a cualquier modelo)
    sub = min(2500, len(X_test))
    X_imp = X_test.iloc[:sub]
    y_imp = y_test.iloc[:sub]
    perm = permutation_importance(
        final_model, X_imp, y_imp, n_repeats=5, random_state=0,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    imp[::-1].plot.barh(ax=ax, color="#0F766E")
    ax.set_xlabel("Aumento de RMSE al permutar (µm)")
    ax.set_title("Importancia por permutación — sensor virtual de P80")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_soft_sensor_importance.png")
    plt.close(fig)

    print("Listo. Figuras en", fig_dir)


if __name__ == "__main__":
    main()
