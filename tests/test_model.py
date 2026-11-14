"""Pruebas de cordura del modelo y del controlador.

Ejecutar con:
    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers import KalmanObserver, LinearModel, MPC  # noqa: E402
from model import (  # noqa: E402
    GrindingCircuit,
    PlantParams,
    breakage_matrix,
    build_size_classes,
    linearize,
    lynch_partition,
    p80_from_psd,
    rosin_rammler_psd,
)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = PlantParams()
        cls.plant = GrindingCircuit(cls.p)
        cls.out = cls.plant.settle(max_hours=800)
        cls.u0 = np.array([cls.p.qf_nom, cls.p.qw_mill_nom, cls.p.qw_dil_nom])

    def test_psd_normalized(self):
        edges, x, _ = build_size_classes(self.p.top_um, self.p.bottom_um, self.p.n_classes)
        frac = rosin_rammler_psd(self.p.feed_rr_scale_um, edges, self.p.feed_m)
        self.assertAlmostEqual(float(frac.sum()), 1.0, places=9)
        self.assertTrue(np.all(frac >= 0))
        # la malla trunca la cola gruesa: el P80 efectivo es cercano a 8.4 mm
        p80_eff = p80_from_psd(x, frac)
        self.assertGreater(p80_eff, 7000.0)
        self.assertLess(p80_eff, 9500.0)

    def test_breakage_matrix(self):
        edges, x, _ = build_size_classes(self.p.top_um, self.p.bottom_um, self.p.n_classes)
        b = breakage_matrix(edges, x)
        col_sums = b.sum(axis=0)
        np.testing.assert_allclose(col_sums[col_sums > 0], 1.0, rtol=1e-9)
        self.assertTrue(np.all(np.diag(b) == 0.0))
        # la fractura solo produce tamaños menores
        for j in range(len(x)):
            self.assertTrue(np.all(b[: j + 1, j] == 0.0))

    def test_partition_curve(self):
        x = np.array([10.0, 50.0, 150.0, 400.0])
        y = lynch_partition(x, 150.0, 3.0)
        self.assertEqual(y[0], y.min())
        self.assertLess(y[0], y[1])
        self.assertLess(y[1], y[2])
        self.assertLess(y[2], y[3])
        self.assertLessEqual(y.max(), 1.0)

    def test_steady_state_ranges(self):
        self.assertGreater(self.out["p80"], 90.0)
        self.assertLess(self.out["p80"], 150.0)
        self.assertGreater(self.out["mill_load"], 200.0)
        self.assertLess(self.out["mill_load"], 350.0)
        self.assertGreater(self.out["density_cf"], 0.55)
        self.assertLess(self.out["density_cf"], 0.75)
        self.assertGreater(self.out["specific_energy"], 5.0)
        self.assertLess(self.out["specific_energy"], 25.0)

    def test_mass_balance_at_steady_state(self):
        m, Mw, ms, Mws = self.plant._split()
        qf = self.u0[0]
        cyc = self.plant.cyclone(ms, Mws)

        # En estado estacionario el agua del molino fija tau_eff
        qw_mf = self.u0[1] + cyc["qw_u"]
        tau_eff = float(Mw) / qw_mf

        # el sólido que entra al molino debe salir de él
        inflow = qf + cyc["qs_u"]
        self.assertAlmostEqual(float(m.sum()) / tau_eff, inflow, delta=2.0)

        # el agua del sump debe balancear: entrada = salida
        self.assertAlmostEqual(float(Mws) / self.p.tau_s, qw_mf + self.u0[2], delta=5.0)

    def test_linearization(self):
        Ad, Bd, C, D, x0, y0 = linearize(self.plant, self.u0)
        self.assertLess(float(np.max(np.abs(np.linalg.eigvals(Ad)))), 1.0)
        G = C @ np.linalg.inv(np.eye(len(Ad)) - Ad) @ Bd + D
        # alimentación sube carga; dilución baja P80; agua de molino sube carga
        self.assertGreater(G[1, 0], 0.2)
        self.assertLess(G[0, 2], 0.0)
        self.assertGreater(G[0, 0], 0.0)

    def test_mpc_respects_bounds(self):
        Ad, Bd, C, D, x0, y0 = linearize(self.plant, self.u0)
        model = LinearModel(Ad, Bd, C, x0, self.u0, y0)
        obs = KalmanObserver(model)
        mpc = MPC(
            model,
            u_min=[150, 30, 120], u_max=[250, 110, 360],
            du_max=[1.2, 2.5, 3.0],
            y_min=[95, 220, 0.6], y_max=[145, 330, 0.72],
            Np=20, Nc=5, observer=obs,
        )
        mpc.reset(y0, self.u0)
        u = mpc.step(y0, self.u0, np.array([118.0, 285.0, 0.66]))
        self.assertEqual(u.shape, (3,))
        self.assertTrue(np.all(u >= np.array([150, 30, 120]) - 1e-6))
        self.assertTrue(np.all(u <= np.array([250, 110, 360]) + 1e-6))
        self.assertTrue(np.all(np.abs(u - self.u0) <= np.array([1.2, 2.5, 3.0]) + 1e-6))


if __name__ == "__main__":
    unittest.main()
