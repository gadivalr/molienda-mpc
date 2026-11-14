"""
Gemelo digital de un circuito de molienda-clasificación.

Configuración:
    mineral fresco + underflow del ciclón -> molino de bolas -> sump (dilución)
    -> hidrociclón -> overflow = producto (P80) / underflow = carga circulante

Modelo:
    * Molino: modelo de mezcla perfecta con balance poblacional (PBM) discreto.
      - Cinética de molienda de primer orden: r_i = k_i * m_i
      - Función de selección de Austin: k_i = a * (x_i / 1000)^alpha
      - Función de ruptura acumulada de Austin:
            B(x) = phi*(x/x_j)^gamma + (1-phi)*(x/x_j)^beta
    * Hidrociclón: curva de partición de Lynch:
            y_i = (exp(alpha_c*x_i/d50) - 1) / (exp(alpha_c*x_i/d50) + exp(alpha_c) - 2)
      con d50 dependiente de flujo y densidad de alimentación.
    * Sump: CSTR con tiempo de residencia tau_s.
    * Potencia de molino: ley de Bond (1961) + término de carga.
    * Presión de ciclón: proporcional a Q^2 * rho.

Referencias:
    Austin, L.G. (1971). A review: introduction to the mathematical description of
        grinding as a rate process. Powder Technology.
    Bond, F.C. (1961). Crushing and grinding calculations. British Chemical Engineering.
    Lynch, A.J. (1977). Mineral Crushing and Grinding Circuits.
    Plitt, L.R. (1976). A mathematical model of the hydrocyclone classifier.
    Wills, B.A. & Napier-Munn, T. (2006). Wills' Mineral Processing Technology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Utilidades de tamaño de partícula
# ----------------------------------------------------------------------------
def build_size_classes(
    top_um: float = 16000.0, bottom_um: float = 10.0, n: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye clases geométricas descendentes.

    Returns
    -------
    edges : (n+1,) bordes descendentes [top ... bottom] en um
    x : (n,) tamaño representativo (media geométrica) en um
    """
    ratio = (top_um / bottom_um) ** (1.0 / n)
    edges = top_um / ratio ** np.arange(n + 1)
    x = np.sqrt(edges[:-1] * edges[1:])
    return edges, x, ratio


def rosin_rammler_psd(scale_um: float, edges: np.ndarray, m: float = 0.9) -> np.ndarray:
    """Distribución Rosin-Rammler acumulada pasante evaluada en los bordes.

    F(x) = 1 - exp(-ln(2) * (x / scale_um)^m)

    `scale_um` es el parámetro de escala de la distribución sin truncar (el
    tamaño con 50 % pasante para m=0.9). La malla superior corta la cola
    gruesa y las fracciones se renormalizan, por lo que el P80 efectivo de la
    alimentación queda en ~8.4 mm para una malla de 16 mm.
    """
    F = 1.0 - np.exp(-np.log(2.0) * (edges / scale_um) ** m)
    F = np.clip(F, 0.0, 1.0)
    frac = F[:-1] - F[1:]  # bordes descendentes
    frac = np.clip(frac, 0.0, None)
    return frac / frac.sum()


def p80_from_psd(x: np.ndarray, frac: np.ndarray) -> float:
    """P80 (um) interpolando log(x) vs. acumulado pasante."""
    order = np.argsort(x)
    xs = x[order]
    fs = np.clip(frac[order], 0.0, None)
    fs = fs / fs.sum()
    cum = np.cumsum(fs)
    if cum[-1] < 0.8:
        return float(xs[-1])
    # interpola en log(x)
    return float(np.exp(np.interp(0.8, cum, np.log(xs))))


def breakage_matrix(
    edges: np.ndarray, x: np.ndarray, phi: float = 0.35, gamma: float = 0.8, beta: float = 3.0
) -> np.ndarray:
    """Matriz de ruptura b[i, j]: fracción de la clase j que cae en la clase i.

    Se usa la función acumulada de Austin. La diagonal se fija en cero para
    garantizar que toda fractura reduzca tamaño; luego se renormaliza cada
    columna.
    """
    n = len(x)
    b = np.zeros((n, n))
    for j in range(n):
        xj = x[j]
        ratio = np.clip(edges / xj, 0.0, 1.0)
        B = phi * ratio ** gamma + (1.0 - phi) * ratio ** beta
        # B(edge_j) = 1, B(edge_{j+1}) = fracción menor al borde inferior
        for i in range(j + 1, n):
            b[i, j] = B[i] - B[i + 1]
        b[j, j] = 0.0
        s = b[:, j].sum()
        if s > 0:
            b[:, j] /= s
    return b


def selection_function(x: np.ndarray, a: float = 0.6, alpha: float = 0.7) -> np.ndarray:
    """Función de selección de Austin k_i [1/h]."""
    return a * (x / 1000.0) ** alpha


def lynch_partition(x: np.ndarray, d50: float, alpha_c: float = 3.0) -> np.ndarray:
    """Curva de partición de Lynch: fracción de la clase i que va al underflow.

    Implementación numéricamente estable: para z grande la función tiende a 1.
    """
    x = np.asarray(x, dtype=float)
    z = alpha_c * x / d50
    y = np.ones_like(z)
    mask = z < 50.0
    y[mask] = np.expm1(z[mask]) / (np.exp(z[mask]) + np.exp(alpha_c) - 2.0)
    return np.clip(y, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------------
@dataclass
class PlantParams:
    # Tamaños
    n_classes: int = 14
    top_um: float = 16000.0
    bottom_um: float = 10.0
    feed_rr_scale_um: float = 12000.0   # escala Rosin-Rammler (ver nota de truncamiento)
    feed_m: float = 0.9

    # Densidades (t/m3)
    rho_s: float = 2.70
    rho_w: float = 1.00

    # Molino
    tau_m: float = 0.40          # h, tiempo de residencia nominal
    v_m: float = 156.3           # m3, volumen de pulpa del molino (calibrado a tau_m)
    a_grind: float = 4.00        # 1/h
    alpha_grind: float = 0.70
    phi_b: float = 0.35
    gamma_b: float = 0.80
    beta_b: float = 3.00

    # Sump
    tau_s: float = 0.08          # h (~5 min)

    # Hidrociclón
    d50_ref: float = 150.0       # um
    q_ref: float = 600.0         # m3/h de alimentación
    c_ref: float = 0.60          # fracción másica de sólidos
    alpha_c: float = 3.0
    r_w: float = 0.20            # fracción de agua que va al underflow

    # Potencia / presión
    wi: float = 15.0             # kWh/t
    p_no_load: float = 300.0     # kW
    p_load_ref: float = 300.0    # t
    p_load_exp: float = 0.20
    dp_ref: float = 120.0        # kPa
    q_dp_ref: float = 600.0      # m3/h
    rho_dp_ref: float = 1.55     # t/m3

    # Entradas nominales
    qf_nom: float = 200.0        # t/h mineral fresco
    qw_mill_nom: float = 60.0    # m3/h agua al molino
    qw_dil_nom: float = 220.0    # m3/h agua de dilución al sump

    # Disturbios
    hardness: float = 1.0        # multiplicador de k_i (>1 = más blando)
    feed_size_factor: float = 1.0  # multiplicador del tamaño de alimentación

    # Integración
    dt: float = 1.0 / 60.0       # h (1 min)

    # Derivados
    edges: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self):
        self.edges, self.x, _ = build_size_classes(self.top_um, self.bottom_um, self.n_classes)
        self.b = breakage_matrix(self.edges, self.x, self.phi_b, self.gamma_b, self.beta_b)


# ----------------------------------------------------------------------------
# Planta
# ----------------------------------------------------------------------------
class GrindingCircuit:
    """Circuito molino de bolas + sump + hidrociclón."""

    def __init__(self, p: PlantParams | None = None):
        self.p = p or PlantParams()
        self.n = self.p.n_classes
        self.nx = 2 * self.n + 2

        # Estado: [m_solidos (n), Mw, ms_sump (n), Mws_sump]
        self.state = np.zeros(self.nx)
        self.state[: self.n] = 100.0 * rosin_rammler_psd(
            self.p.feed_rr_scale_um, self.p.edges, self.p.feed_m
        )
        self.state[self.n] = 30.0
        self.state[self.n + 1 : 2 * self.n + 1] = self.state[: self.n] * (
            self.p.tau_s / self.p.tau_m
        )
        self.state[-1] = 25.0
        self.inputs = np.array([self.p.qf_nom, self.p.qw_mill_nom, self.p.qw_dil_nom])
        self.last_outputs: Dict[str, float] = {}

    # -- división del estado -------------------------------------------------
    def _split(self, state: np.ndarray | None = None):
        s = self.state if state is None else state
        n = self.n
        m = s[:n]
        Mw = s[n]
        ms = s[n + 1 : 2 * n + 1]
        Mws = s[-1]
        return m, Mw, ms, Mws

    # -- hidrociclón ---------------------------------------------------------
    def cyclone(self, ms: np.ndarray, Mws: float):
        p = self.p
        Mss = max(ms.sum(), 1e-9)
        f_cf = np.clip(ms / Mss, 0.0, None)

        qs_cf = Mss / p.tau_s
        qw_cf = max(Mws, 0.0) / p.tau_s
        q_cf = qs_cf + qw_cf
        c_cf = qs_cf / max(q_cf, 1e-9)

        qvol_cf = qs_cf / p.rho_s + qw_cf / p.rho_w
        qvol_cf = max(qvol_cf, 1e-6)
        d50 = p.d50_ref * (p.q_ref / qvol_cf) ** 0.5 * (max(c_cf, 1e-3) / p.c_ref) ** 0.5
        d50 = float(np.clip(d50, 20.0, 500.0))

        y = lynch_partition(p.x, d50, p.alpha_c)
        under_frac = float(np.clip(np.sum(f_cf * y), 1e-9, 1.0 - 1e-9))

        qs_u = qs_cf * under_frac
        qs_o = qs_cf - qs_u
        f_u = f_cf * y / under_frac
        f_o = f_cf * (1.0 - y) / (1.0 - under_frac)

        qw_u = p.r_w * qw_cf
        qw_o = (1.0 - p.r_w) * qw_cf

        rho_slurry = 1.0 / (c_cf / p.rho_s + (1.0 - c_cf) / p.rho_w)
        dp_kpa = p.dp_ref * (qvol_cf / p.q_dp_ref) ** 2 * (rho_slurry / p.rho_dp_ref)

        return {
            "f_cf": f_cf,
            "f_u": f_u,
            "f_o": f_o,
            "qs_cf": qs_cf,
            "qw_cf": qw_cf,
            "qvol_cf": qvol_cf,
            "c_cf": c_cf,
            "d50": d50,
            "qs_u": qs_u,
            "qs_o": qs_o,
            "qw_u": qw_u,
            "qw_o": qw_o,
            "dp_kpa": dp_kpa,
            "rho_slurry": rho_slurry,
        }

    # -- derivadas -----------------------------------------------------------
    def derivatives(self, state: np.ndarray, inputs: np.ndarray, hardness: float | None = None):
        p = self.p
        qf, qw_mill, qw_dil = np.clip(inputs, 0.0, None)
        h = p.hardness if hardness is None else hardness

        m, Mw, ms, Mws = self._split(state)
        Ms = max(m.sum(), 1e-9)
        f_md = np.clip(m / Ms, 0.0, None)

        cyc = self.cyclone(ms, Mws)

        # Alimentación al molino = fresco + underflow recirculado
        qs_mf = qf + cyc["qs_u"]
        f_mf = (qf * rosin_rammler_psd(p.feed_rr_scale_um * p.feed_size_factor, p.edges, p.feed_m)
                + cyc["qs_u"] * cyc["f_u"]) / max(qs_mf, 1e-9)

        k = selection_function(p.x, p.a_grind, p.alpha_grind) * h
        G = p.b * k[None, :]
        G[np.diag_indices_from(G)] = -k

        # Tiempo de residencia efectivo: molino de volumen de pulpa constante.
        # Al aumentar el flujo, la pulpa permanece menos tiempo y el producto
        # se vuelve más grueso (comportamiento típico de un molino de rebalse).
        qw_mf = qw_mill + cyc["qw_u"]
        qvol_mf = qs_mf / p.rho_s + qw_mf / p.rho_w
        tau_eff = float(np.clip(p.v_m / max(qvol_mf, 1e-6), 0.08, 1.0))

        dm = qs_mf * f_mf - m / tau_eff + G @ m
        dMw = qw_mf - Mw / tau_eff

        dms = m / tau_eff - ms / p.tau_s
        dMws = Mw / tau_eff + qw_dil - Mws / p.tau_s

        return np.concatenate([dm, [dMw], dms, [dMws]], dtype=float)

    # -- paso de integración -------------------------------------------------
    def step(self, inputs: np.ndarray, dt: float | None = None, hardness: float | None = None):
        dt = self.p.dt if dt is None else dt
        self.inputs = np.asarray(inputs, dtype=float).copy()
        # RK4
        k1 = self.derivatives(self.state, self.inputs, hardness)
        k2 = self.derivatives(self.state + 0.5 * dt * k1, self.inputs, hardness)
        k3 = self.derivatives(self.state + 0.5 * dt * k2, self.inputs, hardness)
        k4 = self.derivatives(self.state + dt * k3, self.inputs, hardness)
        self.state = self.state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        self.state = np.clip(self.state, 0.0, None)
        return self.outputs(self.state, self.inputs, hardness)

    # -- salidas -------------------------------------------------------------
    def outputs(self, state: np.ndarray | None = None, inputs: np.ndarray | None = None,
                hardness: float | None = None):
        p = self.p
        s = self.state if state is None else state
        u = self.inputs if inputs is None else inputs
        h = p.hardness if hardness is None else hardness

        m, Mw, ms, Mws = self._split(s)
        Ms = max(m.sum(), 1e-9)
        cyc = self.cyclone(ms, Mws)

        p80 = p80_from_psd(p.x, cyc["f_o"])
        p80_feed = p80_from_psd(p.x, rosin_rammler_psd(p.feed_rr_scale_um * p.feed_size_factor,
                                                        p.edges, p.feed_m))

        # Bond + efecto de carga
        bond = 10.0 * p.wi * (1.0 / np.sqrt(max(p80, 1.0)) - 1.0 / np.sqrt(max(p80_feed, 1.0)))
        power = p.p_no_load + bond * max(u[0], 0.0) * (Ms / p.p_load_ref) ** p.p_load_exp
        power = max(power, p.p_no_load)

        out = {
            "p80": p80,
            "p80_feed": p80_feed,
            "mill_load": Ms,
            "mill_water": Mw,
            "sump_load": ms.sum(),
            "sump_water": Mws,
            "density_cf": cyc["c_cf"],
            "d50": cyc["d50"],
            "power_kw": power,
            "dp_kpa": cyc["dp_kpa"],
            "qs_cf": cyc["qs_cf"],
            "qw_cf": cyc["qw_cf"],
            "qvol_cf": cyc["qvol_cf"],
            "qs_u": cyc["qs_u"],
            "qs_o": cyc["qs_o"],
            "circulating_load": cyc["qs_u"] / max(u[0], 1e-9),
            "specific_energy": power / max(u[0], 1e-9),
        }
        self.last_outputs = out
        return out

    def psd_summary(self):
        """Distribuciones de tamaño en los puntos clave del circuito."""
        p = self.p
        m, Mw, ms, Mws = self._split()
        Ms = max(m.sum(), 1e-9)
        Mss = max(ms.sum(), 1e-9)
        cyc = self.cyclone(ms, Mws)
        f_feed = rosin_rammler_psd(p.feed_rr_scale_um * p.feed_size_factor, p.edges, p.feed_m)
        f_mill = np.clip(m / Ms, 0.0, None)
        return {
            "x": p.x,
            "edges": p.edges,
            "feed": f_feed,
            "mill": f_mill,
            "cyclone_feed": cyc["f_cf"],
            "underflow": cyc["f_u"],
            "overflow": cyc["f_o"],
            "p80_feed": p80_from_psd(p.x, f_feed),
            "p80_mill": p80_from_psd(p.x, f_mill),
            "p80_cyclone_feed": p80_from_psd(p.x, cyc["f_cf"]),
            "p80_underflow": p80_from_psd(p.x, cyc["f_u"]),
            "p80_overflow": p80_from_psd(p.x, cyc["f_o"]),
        }

    # -- simulación ----------------------------------------------------------
    def simulate(self, inputs: np.ndarray, n_steps: int, hardness: float | None = None):
        """Simula n_steps con entradas constantes. Devuelve arrays."""
        rows = []
        for _ in range(n_steps):
            out = self.step(inputs, hardness=hardness)
            rows.append(out)
        return rows

    def settle(self, inputs: np.ndarray | None = None, max_hours: float = 200.0,
               tol: float = 1e-7):
        """Integra hasta estado estacionario."""
        u = self.inputs if inputs is None else np.asarray(inputs, dtype=float)
        n_steps = int(max_hours / self.p.dt)
        last = self.state.copy()
        for k in range(n_steps):
            self.step(u)
            if k % 100 == 0 and k > 0:
                delta = np.max(np.abs(self.state - last) / (np.abs(last) + 1e-6))
                last = self.state.copy()
                if delta < tol:
                    break
        return self.outputs(self.state, u)


def linearize(plant: GrindingCircuit, u0: np.ndarray, hardness: float = 1.0,
              eps_x: float = 1e-6, eps_u: float = 1e-5):
    """Linealiza el modelo no lineal en un punto de operación y discretiza.

    Devuelve (Ad, Bd, C, D, x0, y0) con Ts = plant.p.dt.
    """
    from scipy.linalg import expm

    p = plant.p
    x0 = plant.state.copy()
    u0 = np.asarray(u0, float)
    nx = len(x0)
    nu = len(u0)

    def g(x, u):
        out = plant.outputs(x, u, hardness)
        return np.array([out["p80"], out["mill_load"], out["density_cf"]])

    A = np.zeros((nx, nx))
    for i in range(nx):
        dx = eps_x * max(abs(x0[i]), 1.0)
        xp = x0.copy(); xm = x0.copy()
        xp[i] += dx; xm[i] -= dx
        A[:, i] = (plant.derivatives(xp, u0, hardness)
                   - plant.derivatives(xm, u0, hardness)) / (2.0 * dx)

    B = np.zeros((nx, nu))
    for j in range(nu):
        du = eps_u * max(abs(u0[j]), 1.0)
        up = u0.copy(); um = u0.copy()
        up[j] += du; um[j] -= du
        B[:, j] = (plant.derivatives(x0, up, hardness)
                   - plant.derivatives(x0, um, hardness)) / (2.0 * du)

    ny = 3
    C = np.zeros((ny, nx))
    for i in range(nx):
        dx = eps_x * max(abs(x0[i]), 1.0)
        xp = x0.copy(); xm = x0.copy()
        xp[i] += dx; xm[i] -= dx
        C[:, i] = (g(xp, u0) - g(xm, u0)) / (2.0 * dx)

    D = np.zeros((ny, nu))
    for j in range(nu):
        du = eps_u * max(abs(u0[j]), 1.0)
        up = u0.copy(); um = u0.copy()
        up[j] += du; um[j] -= du
        D[:, j] = (g(x0, up) - g(x0, um)) / (2.0 * du)

    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = A
    M[:nx, nx:] = B
    M *= p.dt
    E = expm(M)
    Ad = E[:nx, :nx]
    Bd = E[:nx, nx:]

    return Ad, Bd, C, D, x0, g(x0, u0)


if __name__ == "__main__":
    p = PlantParams()
    plant = GrindingCircuit(p)
    out = plant.settle()
    for k, v in out.items():
        print(f"{k:20s} {v:12.4f}")
    print("state sum", plant.state.sum())

    u0 = np.array([p.qf_nom, p.qw_mill_nom, p.qw_dil_nom])
    Ad, Bd, C, D, x0, y0 = linearize(plant, u0)
    print("linearized y0", y0)
    print("max |eig(Ad)|", np.max(np.abs(np.linalg.eigvals(Ad))))
    G = C @ np.linalg.inv(np.eye(len(Ad)) - Ad) @ Bd + D
    print("DC gain (outputs x inputs):\n", np.round(G, 4))
