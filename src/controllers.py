"""Controladores: PID descentralizado y MPC lineal con corrección de sesgo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np


# ----------------------------------------------------------------------------
# PID
# ----------------------------------------------------------------------------
class PID:
    def __init__(self, kp: float, ti: float, td: float, u_min: float, u_max: float,
                 dt: float = 1.0):
        self.kp = kp
        self.ti = max(ti, 1e-9)
        self.td = td
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt
        self.integral = 0.0
        self.e_prev: Optional[float] = None
        self.u_prev: Optional[float] = None

    def reset(self, u0: float):
        # el integrador arranca en el valor nominal para que el PID opere
        # alrededor del punto de operación (no desde cero)
        self.integral = u0
        self.e_prev = None
        self.u_prev = u0

    def update(self, sp: float, pv: float) -> float:
        e = sp - pv
        p = self.kp * e
        if self.e_prev is None:
            d = 0.0
        else:
            d = self.kp * self.td / self.dt * (e - self.e_prev)

        # integral con anti-windup condicional
        i_new = self.integral + self.kp * self.dt / self.ti * e
        u_unsat = p + i_new + d
        u = float(np.clip(u_unsat, self.u_min, self.u_max))

        # si satura y el error empuja más allá del límite, no integrar
        if (u_unsat > self.u_max and e > 0) or (u_unsat < self.u_min and e < 0):
            i_new = self.integral
            u_unsat = p + i_new + d
            u = float(np.clip(u_unsat, self.u_min, self.u_max))
        self.integral = i_new
        self.e_prev = e
        self.u_prev = u
        return u


# ----------------------------------------------------------------------------
# Modelo ARX MIMO y espacio de estados
# ----------------------------------------------------------------------------
class ARXModel:
    """Modelo ARX MIMO:

        y_k = A1 y_{k-1} + ... + A_na y_{k-na}
              + B1 u_{k-1} + ... + B_nb u_{k-nb} + c + e_k
    """

    def __init__(self, na: int = 8, nb: int = 8):
        self.na = na
        self.nb = nb
        self.ny: Optional[int] = None
        self.nu: Optional[int] = None
        self.A_list: list[np.ndarray] = []
        self.B_list: list[np.ndarray] = []
        self.c: Optional[np.ndarray] = None
        self._build_state_space()

    # -- identificación ------------------------------------------------------
    def fit(self, u: np.ndarray, y: np.ndarray, ridge: float = 1e-6):
        u = np.asarray(u, float)
        y = np.asarray(y, float)
        N, nu = u.shape
        ny = y.shape[1]
        self.nu, self.ny = nu, ny
        start = max(self.na, self.nb)

        X, Y = [], []
        for k in range(start, N):
            reg = []
            for i in range(1, self.na + 1):
                reg.append(y[k - i])
            for j in range(1, self.nb + 1):
                reg.append(u[k - j])
            X.append(np.concatenate(reg))
            Y.append(y[k])
        X = np.asarray(X)
        Y = np.asarray(Y)

        # estandarización de regresores + intercepto
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xs = (X - mu) / sd
        Xs = np.column_stack([Xs, np.ones(len(Xs))])

        n_reg = Xs.shape[1]
        Theta_s = np.linalg.solve(Xs.T @ Xs + ridge * np.eye(n_reg), Xs.T @ Y)
        Theta = Theta_s[:-1] / sd[:, None]
        c = Theta_s[-1] - mu @ Theta

        self.A_list = [Theta[(i - 1) * ny : i * ny, :].T.copy() for i in range(1, self.na + 1)]
        self.B_list = [
            Theta[self.na * ny + (j - 1) * nu : self.na * ny + j * nu, :].T.copy()
            for j in range(1, self.nb + 1)
        ]
        self.c = c
        self._build_state_space()
        return self

    # -- espacio de estados --------------------------------------------------
    def _build_state_space(self):
        if not self.A_list:
            return
        na, nb, ny, nu = self.na, self.nb, self.ny, self.nu
        nx = na * ny + (nb - 1) * nu

        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))
        C = np.zeros((ny, nx))

        # primera fila de bloques: y_{k+1} = A1 y_k + ... + B1 u_k + ...
        C[0:ny, 0:ny] = self.A_list[0]
        for i in range(1, na):
            C[0:ny, i * ny : (i + 1) * ny] = self.A_list[i]
        for j in range(1, nb):
            C[0:ny, na * ny + (j - 1) * nu : na * ny + j * nu] = self.B_list[j]

        A[0:ny, :] = C
        B[0:ny, :] = self.B_list[0]

        # desplazamiento de bloques y
        for i in range(1, na):
            A[i * ny : (i + 1) * ny, (i - 1) * ny : i * ny] = np.eye(ny)

        # desplazamiento de bloques u
        for j in range(1, nb - 1):
            row = na * ny + j * nu
            col = na * ny + (j - 1) * nu
            A[row : row + nu, col : col + nu] = np.eye(nu)
        # el primer bloque u (u_k) entra por B
        if nb - 1 > 0:
            B[na * ny : na * ny + nu, :] = np.eye(nu)

        self.A, self.B, self.C = A, B, C
        self.nx = nx

    # -- estado a partir de historia ----------------------------------------
    def initial_state(self, y_hist: np.ndarray, u_hist: np.ndarray) -> np.ndarray:
        """y_hist: (na, ny) con y_k primero; u_hist: (nb-1, nu) con u_{k-1} primero."""
        return np.concatenate([y_hist.reshape(-1), u_hist.reshape(-1)])

    def one_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.C @ (self.A @ x + self.B @ u) + self.c

    def predict_future(self, x: np.ndarray, U: np.ndarray) -> np.ndarray:
        """U: (Np, nu). Devuelve Y: (Np, ny)."""
        y = []
        xk = x.copy()
        for k in range(len(U)):
            xk = self.A @ xk + self.B @ U[k]
            y.append(self.C @ xk + self.c)
        return np.asarray(y)

    def build_prediction_matrices(self, Np: int, Nc: int) -> Tuple[np.ndarray, np.ndarray]:
        """Y = F x + Phi z + c, con z: (Nc, nu) y bloqueo de movimientos."""
        A, B, C = self.A, self.B, self.C
        nx, nu, ny = self.nx, self.nu, self.ny
        F = np.zeros((Np * ny, nx))
        Phi_full = np.zeros((Np * ny, Np * nu))

        Ak = np.eye(nx)
        powers = [np.eye(nx)]
        for _ in range(Np):
            Ak = A @ Ak
            powers.append(Ak)

        for j in range(Np):
            F[j * ny : (j + 1) * ny, :] = C @ powers[j + 1]
            for i in range(j + 1):
                Phi_full[j * ny : (j + 1) * ny, i * nu : (i + 1) * nu] = (
                    C @ powers[j - i] @ B
                )

        M = np.zeros((Np * nu, Nc * nu))
        for j in range(Np):
            i = min(j, Nc - 1)
            M[j * nu : (j + 1) * nu, i * nu : (i + 1) * nu] = np.eye(nu)

        return F, Phi_full @ M


# ----------------------------------------------------------------------------
# Modelo lineal discretizado + observador Kalman
# ----------------------------------------------------------------------------
class LinearModel:
    """Modelo lineal discreto en variables de desviación.

        dx_{k+1} = A dx_k + B du_k
        dy_k     = C dx_k
    con dx = x - x0, du = u - u0, dy = y - y0.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                 x0: np.ndarray | None = None, u0: np.ndarray | None = None,
                 y0: np.ndarray | None = None):
        self.A = np.asarray(A, float)
        self.B = np.asarray(B, float)
        self.C = np.asarray(C, float)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        self.x0 = np.zeros(self.nx) if x0 is None else np.asarray(x0, float)
        self.u0 = np.zeros(self.nu) if u0 is None else np.asarray(u0, float)
        self.y0 = np.zeros(self.ny) if y0 is None else np.asarray(y0, float)
        self.c = np.zeros(self.ny)
        self.na, self.nb = 1, 1

    def one_step(self, dx: np.ndarray, du: np.ndarray) -> np.ndarray:
        return self.C @ (self.A @ dx + self.B @ du)

    def build_prediction_matrices(self, Np: int, Nc: int) -> Tuple[np.ndarray, np.ndarray]:
        A, B, C = self.A, self.B, self.C
        nx, nu, ny = self.nx, self.nu, self.ny
        F = np.zeros((Np * ny, nx))
        Phi_full = np.zeros((Np * ny, Np * nu))
        powers = [np.eye(nx)]
        for _ in range(Np):
            powers.append(A @ powers[-1])
        for j in range(Np):
            F[j * ny : (j + 1) * ny, :] = C @ powers[j + 1]
            for i in range(j + 1):
                Phi_full[j * ny : (j + 1) * ny, i * nu : (i + 1) * nu] = (
                    C @ powers[j - i] @ B
                )
        M = np.zeros((Np * nu, Nc * nu))
        for j in range(Np):
            i = min(j, Nc - 1)
            M[j * nu : (j + 1) * nu, i * nu : (i + 1) * nu] = np.eye(nu)
        return F, Phi_full @ M


class KalmanObserver:
    """Observador aumentado con perturbaciones de salida (offset-free).

    Estado aumentado: [dx, d], donde d son perturbaciones constantes que
    representan el error de modelo en cada salida.

        dx_{k+1} = A dx_k + B du_k
        d_{k+1}  = d_k
        dy_k     = C dx_k + d_k

    El filtro de Kalman estacionario entrega dx y d en cada paso. Usar d
    dentro del MPC garantiza seguimiento sin offset sin un integrador
    sintonizado a mano.
    """

    def __init__(self, model: LinearModel,
                 q_rel: float = 5e-4, r_sigma=(1.5, 2.0, 0.004),
                 q_dist=(0.05, 0.20, 0.0008)):
        from scipy.linalg import solve_discrete_are

        m = model
        nx, ny, nu = m.nx, m.ny, m.nu
        Aa = np.zeros((nx + ny, nx + ny))
        Aa[:nx, :nx] = m.A
        Aa[nx:, nx:] = np.eye(ny)
        Ba = np.vstack([m.B, np.zeros((ny, nu))])
        Ca = np.hstack([m.C, np.eye(ny)])

        Qx = np.diag((q_rel * np.maximum(np.abs(m.x0), 1.0)) ** 2)
        Q = np.zeros_like(Aa)
        Q[:nx, :nx] = Qx
        Q[nx:, nx:] = np.diag(np.asarray(q_dist, float) ** 2)
        R = np.diag(np.asarray(r_sigma, float) ** 2)

        P = solve_discrete_are(Aa.T, Ca.T, Q, R)
        S = Ca @ P @ Ca.T + R
        self.K = P @ Ca.T @ np.linalg.inv(S)
        self.Aa, self.Ba, self.Ca = Aa, Ba, Ca
        self.model = model
        self.z_hat = np.zeros(nx + ny)

    def reset(self):
        self.z_hat = np.zeros(self.model.nx + self.model.ny)

    def update(self, y: np.ndarray, u_prev: np.ndarray):
        m = self.model
        z_pred = self.Aa @ self.z_hat + self.Ba @ (u_prev - m.u0)
        innovation = (y - m.y0) - self.Ca @ z_pred
        self.z_hat = z_pred + self.K @ innovation
        dx_hat = self.z_hat[: m.nx]
        d_hat = self.z_hat[m.nx :]
        return dx_hat, d_hat, innovation


# ----------------------------------------------------------------------------
# MPC
# ----------------------------------------------------------------------------
class MPC:
    def __init__(
        self,
        model: ARXModel | LinearModel,
        u_min: np.ndarray,
        u_max: np.ndarray,
        du_max: np.ndarray,
        y_min: np.ndarray,
        y_max: np.ndarray,
        Np: int = 45,
        Nc: int = 8,
        q_scales: Tuple[float, float, float] = (8.0, 15.0, 0.02),
        r_scales: Tuple[float, float, float] = (4.0, 8.0, 12.0),
        rho: float = 1e4,
        econ_weight: float = 0.0,
        observer: Optional[KalmanObserver] = None,
        bias_gain: float = 0.04,
        bias_limits: Tuple[float, float, float] = (12.0, 25.0, 0.05),
    ):
        self.model = model
        self.linear = isinstance(model, LinearModel)
        self.observer = observer
        self.bias_gain = bias_gain
        self.bias_limits = np.asarray(bias_limits, float)
        self.u_min = np.asarray(u_min, float)
        self.u_max = np.asarray(u_max, float)
        self.du_max = np.asarray(du_max, float)
        self.y_min = np.asarray(y_min, float)
        self.y_max = np.asarray(y_max, float)
        self.Np, self.Nc = Np, Nc
        self.q_scales = np.asarray(q_scales, float)
        self.r_scales = np.asarray(r_scales, float)
        self.rho = rho
        self.econ_weight = econ_weight
        self.F, self.Phi = model.build_prediction_matrices(Np, Nc)

        self.y_hist: Optional[np.ndarray] = None
        self.u_hist: Optional[np.ndarray] = None
        self.x_prev: Optional[np.ndarray] = None
        self.u_prev_pred: Optional[np.ndarray] = None
        self.last_solve_time = 0.0
        self.last_status = ""

    def reset(self, y0: np.ndarray, u0: np.ndarray):
        self.y_hist = np.tile(y0, (self.model.na, 1))
        self.u_hist = np.tile(u0, (self.model.nb - 1, 1))
        self.x_prev = None
        self.u_prev_pred = None
        self.bias = np.zeros(self.model.ny)
        if self.observer is not None:
            self.observer.reset()

    def step(
        self,
        y_meas: np.ndarray,
        u_prev: np.ndarray,
        y_sp: np.ndarray,
        econ_weight: Optional[float] = None,
    ) -> np.ndarray:
        y_meas = np.asarray(y_meas, float)
        u_prev = np.asarray(u_prev, float)
        y_sp = np.asarray(y_sp, float)

        if self.linear:
            # variables de desviación; el observador aumentado entrega
            # directamente la perturbación de salida estimada
            x, d_hat, innovation = self.observer.update(y_meas, u_prev)
            self.bias = d_hat
            bias = d_hat
            u_ref = self.model.u0
            y_ref = self.model.y0
        else:
            # ARX: estado construido con la historia de y/u (variables absolutas)
            self.y_hist = np.vstack([y_meas[None, :], self.y_hist[:-1, :]])
            self.u_hist = np.vstack([u_prev[None, :], self.u_hist[:-1, :]])
            x = self.model.initial_state(self.y_hist, self.u_hist)
            u_ref = np.zeros(self.model.nu)
            y_ref = np.zeros(self.model.ny)

            # corrección de sesgo (offset-free tipo DMC)
            bias = np.zeros(self.model.ny)
            if self.x_prev is not None and self.u_prev_pred is not None:
                y_pred = self.model.one_step(self.x_prev, self.u_prev_pred)
                bias = y_meas - y_pred
                bias = np.clip(bias, -self.bias_limits, self.bias_limits)

        nu, ny = self.model.nu, self.model.ny
        Np, Nc = self.Np, self.Nc
        z = cp.Variable((Nc, nu))
        s = cp.Variable((Np, ny), nonneg=True)
        Z = cp.reshape(z, (Nc * nu,), order="C")
        Y = self.F @ x + self.Phi @ Z + np.tile(self.model.c + bias, Np)

        q_sqrt = np.tile(1.0 / self.q_scales, Np)
        R = np.tile(y_sp - y_ref, Np)
        cost = cp.sum_squares(cp.multiply(q_sqrt, Y - R))

        u_prev_dev = u_prev - u_ref
        dz = cp.diff(cp.vstack([u_prev_dev.reshape(1, nu), z]), axis=0)
        r_sqrt = np.tile(1.0 / self.r_scales, Nc)
        cost += cp.sum_squares(cp.multiply(r_sqrt, cp.reshape(dz, (Nc * nu,), order="C")))
        cost += self.rho * cp.sum(s)

        ew = self.econ_weight if econ_weight is None else econ_weight
        if ew:
            cost += -ew * cp.sum(z[:, 0])

        Ymat = cp.reshape(Y, (Np, ny), order="C")
        constraints = [
            z >= self.u_min - u_ref,
            z <= self.u_max - u_ref,
            dz >= -self.du_max,
            dz <= self.du_max,
            Ymat >= (self.y_min - y_ref) - s,
            Ymat <= (self.y_max - y_ref) + s,
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints)

        import time
        t0 = time.perf_counter()
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=20000)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
            except Exception:
                pass
        self.last_solve_time = time.perf_counter() - t0
        self.last_status = prob.status

        if z.value is None:
            u_opt = u_prev.copy()
        else:
            u_opt = u_ref + np.asarray(z.value[0]).flatten()
        u_opt = np.clip(u_opt, self.u_min, self.u_max)

        # guardar para el sesgo del siguiente paso
        self.x_prev = x
        self.u_prev_pred = u_opt.copy()
        return u_opt


# ----------------------------------------------------------------------------
# PID descentralizado para comparación
# ----------------------------------------------------------------------------
class DecentralizedPID:
    """PID 3x3 con pareo:
        P80       -> agua de dilución (acción inversa)
        carga     -> alimentación (acción inversa)
        densidad  -> agua al molino (acción inversa)
    """

    def __init__(self, p, u0: np.ndarray):
        self.pid_p80 = PID(kp=-2.5, ti=45.0, td=0.0, u_min=120.0, u_max=360.0)
        self.pid_load = PID(kp=0.50, ti=30.0, td=0.0, u_min=150.0, u_max=250.0)
        self.pid_dens = PID(kp=350.0, ti=30.0, td=0.0, u_min=30.0, u_max=110.0)
        self.pid_p80.reset(u0[2])
        self.pid_load.reset(u0[0])
        self.pid_dens.reset(u0[1])
        self.u0 = u0.copy()

    def step(self, y_meas: np.ndarray, y_sp: np.ndarray) -> np.ndarray:
        p80, load, dens = y_meas
        qw_dil = self.pid_p80.update(y_sp[0], p80)
        qf = self.pid_load.update(y_sp[1], load)
        qw_mill = self.pid_dens.update(y_sp[2], dens)
        return np.array([qf, qw_mill, qw_dil])
