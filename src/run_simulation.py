"""
Genera las figuras y métricas del gemelo digital de molienda-clasificación.

Salidas:
    figures/fig_psd.png                 distribuciones y curva de partición
    figures/fig_open_loop.png           respuestas a escalones en lazo abierto
    figures/fig_closed_loop.png         MPC vs PID ante cambios de setpoint
    figures/fig_disturbance.png         MPC vs PID ante mineral más duro/grueso
    figures/fig_soft_sensor_loop.png    MPC usando el sensor virtual de P80
    figures/fig_performance.png         índices de desempeño
    reports/closed_loop_metrics.json    métricas numéricas
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controllers import DecentralizedPID, KalmanObserver, LinearModel, MPC
from model import GrindingCircuit, PlantParams, linearize, lynch_partition
from simulation import MEAS_SIGMA, add_noise
from soft_sensor import OnlineSoftSensor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
REP = ROOT / "reports"
DATA = ROOT / "data"
for d in (FIG, REP, DATA):
    d.mkdir(exist_ok=True)

C_MPC = "#0F766E"
C_PID = "#B45309"
C_SP = "#111827"
C_TRUE = "#111827"
C_EST = "#7C3AED"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def build_plant(p: PlantParams, hours: float = 800.0):
    plant = GrindingCircuit(p)
    out = plant.settle(max_hours=hours)
    return plant, out


def nominal(p: PlantParams, out):
    u0 = np.array([p.qf_nom, p.qw_mill_nom, p.qw_dil_nom], dtype=float)
    y0 = np.array([out["p80"], out["mill_load"], out["density_cf"]], dtype=float)
    return u0, y0


def make_mpc(plant: GrindingCircuit, u0: np.ndarray) -> MPC:
    Ad, Bd, C, D, x0, y0 = linearize(plant, u0)
    model = LinearModel(Ad, Bd, C, x0, u0, y0)
    obs = KalmanObserver(model, q_rel=5e-4)
    mpc = MPC(
        model,
        u_min=[150.0, 30.0, 120.0],
        u_max=[250.0, 110.0, 360.0],
        du_max=[1.2, 2.5, 3.0],
        y_min=[95.0, 220.0, 0.60],
        y_max=[145.0, 330.0, 0.72],
        Np=75,
        Nc=8,
        q_scales=(6.0, 20.0, 0.02),
        r_scales=(5.0, 10.0, 30.0),
        rho=1e4,
        observer=obs,
    )
    mpc.reset(y0, u0)
    return mpc


def scenario_setpoints(n: int = 540):
    sp = np.tile([120.0, 280.0, 0.665], (n, 1))
    sp[120:, 0] = 112.0     # producto más fino
    sp[300:, 1] = 290.0     # mayor carga
    return sp, np.ones(n), np.ones(n)


def scenario_disturbance(n: int = 600):
    sp = np.tile([120.0, 280.0, 0.665], (n, 1))
    hardness = np.ones(n)
    f80 = np.ones(n)
    hardness[120:] = 0.88    # 12 % más duro
    f80[360:] = 1.10         # 10 % más grueso
    return sp, hardness, f80


# ---------------------------------------------------------------------------
# Lazo cerrado
# ---------------------------------------------------------------------------
def run_loop(
    kind: str,
    p: PlantParams,
    sp: np.ndarray,
    hardness: np.ndarray,
    f80: np.ndarray,
    seed: int = 5,
    analyzer_fail: int | None = None,
):
    plant, out0 = build_plant(p)
    u0, y0 = nominal(p, out0)

    if kind == "mpc":
        ctrl = make_mpc(plant, u0)
    elif kind == "pid":
        ctrl = DecentralizedPID(p, u0)
    else:
        raise ValueError(kind)

    soft = None
    q80 = 0.0
    if analyzer_fail is not None:
        model = joblib.load(DATA / "soft_sensor_model.joblib")
        with open(REP / "soft_sensor_metrics.json", encoding="utf-8") as fh:
            q80 = float(json.load(fh)["conformal_q80_um"])
        soft = OnlineSoftSensor(model, q80)

    rng = np.random.default_rng(seed)
    u_prev = u0.copy()
    rows = []
    n = len(sp)

    for k in range(n):
        plant.p.feed_size_factor = float(f80[k])
        true = dict(plant.last_outputs)
        true["qf"] = u_prev[0]
        true["qw_mill"] = u_prev[1]
        true["qw_dil"] = u_prev[2]
        meas = add_noise(true, rng)

        y_meas = np.array([meas["p80"], meas["mill_load"], meas["density_cf"]])
        p80_est = np.nan
        p80_lo = np.nan
        p80_hi = np.nan

        if soft is not None:
            ss_meas = {c: meas[c] for c in (
                "qf", "qw_mill", "qw_dil", "power_kw", "dp_kpa",
                "density_cf", "mill_load", "qs_cf", "circulating_load",
            )}
            p80_est, p80_lo, p80_hi = soft.update(ss_meas)
            if k >= analyzer_fail:
                y_meas[0] = p80_est

        if kind == "mpc":
            u = ctrl.step(y_meas, u_prev, sp[k])
        else:
            u = ctrl.step(y_meas, sp[k])

        out = plant.step(u, hardness=float(hardness[k]))
        u_prev = u
        rows.append({
            "t": k,
            "p80": out["p80"],
            "p80_meas": meas["p80"],
            "p80_est": p80_est,
            "p80_lo": p80_lo,
            "p80_hi": p80_hi,
            "mill_load": out["mill_load"],
            "density_cf": out["density_cf"],
            "qf": u[0],
            "qw_mill": u[1],
            "qw_dil": u[2],
            "power_kw": out["power_kw"],
            "specific_energy": out["specific_energy"],
            "sp_p80": sp[k, 0],
            "sp_load": sp[k, 1],
            "sp_density": sp[k, 2],
        })

    import pandas as pd
    return pd.DataFrame(rows)


def loop_metrics(df, t_change=120, t_next=300, band=1.5):
    t = df["t"].to_numpy(dtype=float)
    out = {}
    for key, sp_key in (("p80", "sp_p80"), ("mill_load", "sp_load"),
                        ("density_cf", "sp_density")):
        out[f"iae_{key}"] = float(np.trapezoid(np.abs(df[key] - df[sp_key]), t) / len(t))

    m = (t >= t_change) & (t < t_next)
    err = np.abs(df.loc[m, "p80"] - df.loc[m, "sp_p80"]).to_numpy()
    tt = t[m]
    outside = np.where(err > band)[0]
    out["settling_p80_min"] = float(
        0.0 if len(outside) == 0 else tt[outside[-1]] - t_change + 1
    )
    out["energy_kwh_per_t"] = float(df["specific_energy"].mean())
    out["throughput_tph"] = float(df["qf"].mean())
    out["dilution_m3h"] = float(df["qw_dil"].mean())
    return out


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figure_psd(plant: GrindingCircuit, out):
    psd = plant.psd_summary()
    x = psd["x"]
    order = np.argsort(x)
    xs = x[order]

    def cumulative(f):
        fs = np.clip(np.asarray(f, float)[order], 0.0, None)
        return 100.0 * np.cumsum(fs / fs.sum())

    # La descarga del molino y la alimentación al ciclón son la misma
    # distribución: el sump solo agrega agua de dilución.
    curves = [
        ("Alimentación fresca", "feed", "#9CA3AF"),
        ("Descarga molino = alimentación ciclón", "mill", "#2563EB"),
        ("Underflow (carga circulante)", "underflow", C_PID),
        ("Overflow (producto)", "overflow", C_MPC),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    for name, key, color in curves:
        ax.semilogx(xs, cumulative(psd[key]), color=color, lw=1.7,
                    label=f"{name} — P80={psd['p80_' + key]:.0f} µm")
        ax.plot([psd["p80_" + key]], [80], "o", color=color, ms=5)
    ax.axhline(80, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Tamaño de partícula (µm)")
    ax.set_ylabel("Pasante acumulado (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_title("(a) Distribución de tamaño en el circuito")
    ax.legend(fontsize=7.4, loc="lower right")

    ax = axes[1]
    y = lynch_partition(x, out["d50"], plant.p.alpha_c)[order]
    ax.semilogx(xs, 100.0 * y, color=C_MPC, lw=2.0,
                label=f"Lynch (d50={out['d50']:.0f} µm)")
    ax.axhline(50, color="black", lw=0.8, ls=":")
    ax.axvline(out["d50"], color="black", lw=0.8, ls=":")
    ax.set_xlabel("Tamaño de partícula (µm)")
    ax.set_ylabel("Fracción al underflow (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_title("(b) Curva de partición del hidrociclón")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG / "fig_psd.png")
    plt.close(fig)


def figure_open_loop(p: PlantParams, u0: np.ndarray, out0):
    steps = [
        ("Alimentación +20 t/h", np.array([20.0, 0.0, 0.0]), "#2563EB"),
        ("Agua al molino +10 m³/h", np.array([0.0, 10.0, 0.0]), "#7C3AED"),
        ("Agua de dilución +40 m³/h", np.array([0.0, 0.0, 40.0]), C_MPC),
    ]
    keys = [("p80", "P80 (µm)"), ("mill_load", "Carga del molino (t)"),
            ("density_cf", "Densidad alimentación ciclón (–)")]
    base = np.array([out0["p80"], out0["mill_load"], out0["density_cf"]])
    data = {}
    gains = {}

    for name, du, _ in steps:
        plant, _ = build_plant(p)
        hist = []
        for _ in range(240):
            o = plant.step(u0 + du)
            hist.append([o["p80"], o["mill_load"], o["density_cf"]])
        hist = np.asarray(hist)
        data[name] = hist
        idx = int(np.argmax(np.abs(du)))
        gains[name] = {
            "input": ["qf", "qw_mill", "qw_dil"][idx],
            "step": float(du[idx]),
            "dp80": float(hist[-1, 0] - base[0]),
            "dload": float(hist[-1, 1] - base[1]),
            "ddensity": float(hist[-1, 2] - base[2]),
        }

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    t = np.arange(240)
    for ax, (key, label) in zip(axes, keys):
        j = keys.index((key, label))
        for name, _, color in steps:
            ax.plot(t, data[name][:, j], color=color, lw=1.6, label=name)
        ax.axhline(base[j], color="black", lw=0.8, ls=":")
        ax.set_xlabel("Tiempo (min)")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(fontsize=7.6, loc="best")
    fig.suptitle("Respuestas a escalones en lazo abierto (desde el punto de operación)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_open_loop.png", bbox_inches="tight")
    plt.close(fig)
    return gains


def plot_closed_loop(df_mpc, df_pid, title, fname):
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.4), sharex=True)
    t = df_mpc["t"]

    outs = [
        ("p80", "sp_p80", "P80 (µm)"),
        ("mill_load", "sp_load", "Carga del molino (t)"),
        ("density_cf", "sp_density", "Densidad (–)"),
    ]
    for ax, (key, sp_key, label) in zip(axes[0], outs):
        ax.plot(t, df_mpc[key], color=C_MPC, lw=1.6, label="MPC")
        ax.plot(t, df_pid[key], color=C_PID, lw=1.3, ls="--", label="PID")
        ax.plot(t, df_mpc[sp_key], color=C_SP, lw=1.0, ls=":", label="Setpoint")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0, 0].legend(fontsize=8, loc="best")

    ins = [("qf", "Alimentación (t/h)"), ("qw_mill", "Agua molino (m³/h)"),
           ("qw_dil", "Agua dilución (m³/h)")]
    for ax, (key, label) in zip(axes[1], ins):
        ax.plot(t, df_mpc[key], color=C_MPC, lw=1.6, label="MPC")
        ax.plot(t, df_pid[key], color=C_PID, lw=1.3, ls="--", label="PID")
        ax.set_xlabel("Tiempo (min)")
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def figure_soft_sensor_loop(df, title="MPC con sensor virtual de P80"):
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.4), sharex=True)
    t = df["t"]

    ax = axes[0, 0]
    ax.plot(t, df["p80"], color=C_TRUE, lw=1.4, label="P80 real (analizador)")
    ax.plot(t, df["p80_est"], color=C_EST, lw=1.2, label="P80 estimado (PLS)")
    ax.fill_between(t, df["p80_lo"], df["p80_hi"], color=C_EST, alpha=0.12,
                    label="Intervalo conformal 80 %")
    ax.plot(t, df["sp_p80"], color=C_SP, lw=1.0, ls=":", label="Setpoint")
    ax.set_ylabel("P80 (µm)")
    ax.set_title("P80: medición vs estimación")
    ax.legend(fontsize=7.2, loc="best")

    ax = axes[0, 1]
    ax.plot(t, df["p80"] - df["p80_est"], color=C_EST, lw=1.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Error de estimación (µm)")
    ax.set_title("Residuo del sensor virtual")

    ax = axes[0, 2]
    ax.plot(t, df["mill_load"], color=C_MPC, lw=1.5, label="Carga")
    ax.plot(t, df["sp_load"], color=C_SP, lw=1.0, ls=":")
    ax2 = ax.twinx()
    ax2.plot(t, df["density_cf"], color=C_PID, lw=1.2, label="Densidad")
    ax2.set_ylabel("Densidad (–)", color=C_PID)
    ax2.tick_params(axis="y", colors=C_PID)
    ax.set_ylabel("Carga (t)")
    ax.set_title("Carga y densidad")

    ins = [("qf", "Alimentación (t/h)"), ("qw_mill", "Agua molino (m³/h)"),
           ("qw_dil", "Agua dilución (m³/h)")]
    for ax, (key, label) in zip(axes[1], ins):
        ax.plot(t, df[key], color=C_MPC, lw=1.5)
        ax.set_xlabel("Tiempo (min)")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[1, 0].axvline(120, color=C_PID, lw=1.0, ls="--")
    axes[1, 1].axvline(120, color=C_PID, lw=1.0, ls="--")
    axes[1, 2].axvline(120, color=C_PID, lw=1.0, ls="--")
    axes[1, 0].text(124, axes[1, 0].get_ylim()[0] + 2, "falla del analizador",
                    color=C_PID, fontsize=7.5)

    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / "fig_soft_sensor_loop.png")
    plt.close(fig)


def figure_performance(metrics):
    labels = ["P80", "Carga", "Densidad"]
    keys = ["iae_p80", "iae_mill_load", "iae_density_cf"]
    scen = [("Cambios de setpoint", "setpoints"), ("Rechazo de disturbios", "disturbance")]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=False)
    xpos = np.arange(len(labels))
    width = 0.36
    for ax, (title, key) in zip(axes, scen):
        mpc_vals = [metrics[key]["mpc"][k] for k in keys]
        pid_vals = [metrics[key]["pid"][k] for k in keys]
        # normaliza cada variable respecto del PID para comparar formas
        norm = np.array(pid_vals)
        norm[norm <= 0] = 1.0
        ax.bar(xpos - width / 2, np.array(mpc_vals) / norm, width, color=C_MPC, label="MPC")
        ax.bar(xpos + width / 2, np.array(pid_vals) / norm, width, color=C_PID, label="PID")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("IAE normalizado (–)")
        ax.set_title(title)
        ax.axhline(1.0, color="black", lw=0.7, ls=":")
    axes[0].legend(fontsize=8)
    fig.suptitle("Índice de error absoluto integrado (menor es mejor; PID = 1)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_performance.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    p = PlantParams()
    plant, out0 = build_plant(p)
    u0, y0 = nominal(p, out0)

    print("Estado estacionario:")
    for k in ("p80", "mill_load", "density_cf", "power_kw", "dp_kpa",
              "circulating_load", "specific_energy"):
        print(f"  {k:18s} {out0[k]:10.3f}")

    figure_psd(plant, out0)
    print("Figura PSD lista")

    gains = figure_open_loop(p, u0, out0)
    print("Figura lazo abierto lista")

    # --- cambios de setpoint ---
    sp, h, f = scenario_setpoints()
    print("Simulando MPC (setpoints)...")
    df_mpc = run_loop("mpc", p, sp, h, f, seed=5)
    print("Simulando PID (setpoints)...")
    df_pid = run_loop("pid", p, sp, h, f, seed=5)
    plot_closed_loop(df_mpc, df_pid,
                     "Cambios de setpoint: MPC vs PID descentralizado",
                     "fig_closed_loop.png")
    print("Figura lazo cerrado lista")

    # --- disturbios ---
    sp_d, h_d, f_d = scenario_disturbance()
    print("Simulando MPC (disturbios)...")
    df_mpc_d = run_loop("mpc", p, sp_d, h_d, f_d, seed=7)
    print("Simulando PID (disturbios)...")
    df_pid_d = run_loop("pid", p, sp_d, h_d, f_d, seed=7)
    plot_closed_loop(df_mpc_d, df_pid_d,
                     "Mineral más duro (t=120 min) y más grueso (t=360 min): MPC vs PID",
                     "fig_disturbance.png")
    print("Figura disturbios lista")

    # --- sensor virtual en el lazo ---
    print("Simulando MPC con sensor virtual...")
    df_ss = run_loop("mpc", p, sp, h, f, seed=5, analyzer_fail=120)
    figure_soft_sensor_loop(df_ss)
    print("Figura sensor virtual en lazo lista")

    # --- métricas ---
    metrics = {
        "setpoints": {
            "mpc": loop_metrics(df_mpc),
            "pid": loop_metrics(df_pid),
        },
        "disturbance": {
            "mpc": loop_metrics(df_mpc_d, t_change=120, t_next=600),
            "pid": loop_metrics(df_pid_d, t_change=120, t_next=600),
        },
    }
    ss_err = df_ss.loc[df_ss["t"] >= 120]
    metrics["soft_sensor_loop"] = {
        "mpc_with_soft_sensor": loop_metrics(df_ss),
        "p80_rmse_um": float(np.sqrt(np.mean((ss_err["p80"] - ss_err["p80_est"]) ** 2))),
        "p80_mae_um": float(np.mean(np.abs(ss_err["p80"] - ss_err["p80_est"]))),
        "coverage_80": float(np.mean(
            (ss_err["p80"] >= ss_err["p80_lo"]) & (ss_err["p80"] <= ss_err["p80_hi"])
        )),
    }

    # ganancias linealizadas
    Ad, Bd, C, D, x0, y0l = linearize(plant, u0)
    G = C @ np.linalg.inv(np.eye(len(Ad)) - Ad) @ Bd + D
    report = {
        "steady_state": {k: float(out0[k]) for k in (
            "p80", "mill_load", "density_cf", "power_kw", "dp_kpa",
            "circulating_load", "specific_energy",
        )},
        "operating_point": {"u0": u0.tolist(), "y0": y0.tolist()},
        "dc_gain": {
            "rows": ["p80", "mill_load", "density_cf"],
            "cols": ["qf", "qw_mill", "qw_dil"],
            "values": G.tolist(),
        },
        "open_loop_steps": gains,
        "closed_loop": metrics,
        "mpc": {
            "Np": 75, "Nc": 8,
            "u_min": [150, 30, 120], "u_max": [250, 110, 360],
            "du_max": [1.2, 2.5, 3.0],
        },
    }
    with open(REP / "closed_loop_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    figure_performance(metrics)
    print("Métricas guardadas en", REP / "closed_loop_metrics.json")


if __name__ == "__main__":
    main()
