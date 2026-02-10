import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.inviscid import get_inviscid_core
from python.run_inviscid_sweep import build_inviscid_context


class TestPhase2EulerRegression(unittest.TestCase):
    def _panel_inviscid(self, alpha_deg):
        ctx = build_inviscid_context(12, minf=0.0, alfa_rad=math.radians(alpha_deg), quiet=True)
        ctx.INVISCID_MODEL = "panel"
        cosa = math.cos(ctx.ALFA)
        sina = math.sin(ctx.ALFA)
        for i in range(1, ctx.N + 1):
            ctx.GAM[i] = cosa * ctx.GAMU[i][1] + sina * ctx.GAMU[i][2]
        get_inviscid_core(ctx).update_force_coefficients(ctx)
        return ctx

    def _euler_inviscid(self, alpha_deg):
        ctx = build_inviscid_context(12, minf=0.0, alfa_rad=math.radians(alpha_deg), quiet=True)
        ctx.INVISCID_MODEL = "euler"
        get_inviscid_core(ctx).update_force_coefficients(ctx)
        return ctx

    def test_residual_decrease_and_physical_state(self):
        ctx = self._euler_inviscid(4.0)
        diag = ctx.EULER_DIAGNOSTICS

        self.assertTrue(diag["states_ok"])
        self.assertTrue(diag["te_quality"]["ok"])
        self.assertGreater(len(diag["residual_history"]), 5)
        self.assertLess(diag["residual_history"][-1], diag["residual_history"][0])

        self.assertTrue(math.isfinite(ctx.CL))
        self.assertTrue(math.isfinite(ctx.CM))
        self.assertGreater(diag["cp_max"], diag["cp_min"])

    def test_cl_alpha_trend_vs_panel_attached_regime(self):
        alphas = [0.0, 4.0, 8.0]
        panel_cls = []
        euler_cls = []

        for a in alphas:
            panel_ctx = self._panel_inviscid(a)
            euler_ctx = self._euler_inviscid(a)
            panel_cls.append(panel_ctx.CL)
            euler_cls.append(euler_ctx.CL)

        # trend: CL should increase with alpha in attached regime
        self.assertLessEqual(euler_cls[0], euler_cls[1])
        self.assertLessEqual(euler_cls[1], euler_cls[2])

        # magnitude consistency: Euler MVP should stay in same order as panel
        for pe, ee in zip(panel_cls[1:], euler_cls[1:]):
            self.assertGreater(ee, 0.0)
            ratio = ee / pe if pe != 0.0 else 1.0
            self.assertGreater(ratio, 0.6)
            self.assertLess(ratio, 1.4)


if __name__ == "__main__":
    unittest.main()
