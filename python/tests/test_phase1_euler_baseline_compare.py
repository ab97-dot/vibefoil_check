import contextlib
import io
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.inviscid import get_inviscid_core
from python.run_inviscid_sweep import build_inviscid_context
from python.run_viscous_sweep import _patch_gauss_for_python_solver, build_viscal_context
from python.xbl import XBlState, blpini
from python.xoper import viscal


class TestPhase1EulerBaselineCompare(unittest.TestCase):
    def test_naca0012_panel_viscous_and_euler_at_4_and_8_deg(self):
        _patch_gauss_for_python_solver()
        alphas = [4.0, 8.0]
        minf = 0.0
        reinf = 1.0e6

        print("\nNACA0012 comparison: Re=1e6 Mach=0.0")
        print("mode,regime,alpha_deg,CL,CD,CM")

        for alpha_deg in alphas:
            alfa = alpha_deg * math.pi / 180.0

            # Panel inviscid
            ctx_pi = build_inviscid_context(12, minf=minf, alfa_rad=alfa, quiet=True)
            ctx_pi.INVISCID_MODEL = "panel"
            cosa = math.cos(ctx_pi.ALFA)
            sina = math.sin(ctx_pi.ALFA)
            for i in range(1, ctx_pi.N + 1):
                ctx_pi.GAM[i] = cosa * ctx_pi.GAMU[i][1] + sina * ctx_pi.GAMU[i][2]
            get_inviscid_core(ctx_pi).update_force_coefficients(ctx_pi)
            ctx_pi.CD = 0.0
            print(f"panel,inviscid,{alpha_deg:.1f},{ctx_pi.CL:.6f},{ctx_pi.CD:.6f},{ctx_pi.CM:.6f}")

            # Euler inviscid
            ctx_ei = build_inviscid_context(12, minf=minf, alfa_rad=alfa, quiet=True)
            ctx_ei.INVISCID_MODEL = "euler"
            get_inviscid_core(ctx_ei).update_force_coefficients(ctx_ei)
            ctx_ei.CD = 0.0
            print(f"euler,inviscid,{alpha_deg:.1f},{ctx_ei.CL:.6f},{ctx_ei.CD:.6f},{ctx_ei.CM:.6f}")

            # Panel viscous
            ctx_pv = build_viscal_context(12, minf=minf, reinf=reinf, alfa_rad=alfa, quiet=True)
            ctx_pv.INVISCID_MODEL = "panel"
            cosa_v = math.cos(ctx_pv.ALFA)
            sina_v = math.sin(ctx_pv.ALFA)
            for i in range(1, ctx_pv.N + 1):
                ctx_pv.GAM[i] = cosa_v * ctx_pv.GAMU[i][1] + sina_v * ctx_pv.GAMU[i][2]
            bl = XBlState()
            blpini(bl)
            with contextlib.redirect_stdout(io.StringIO()):
                viscal(ctx_pv, bl, 10)
            print(f"panel,viscous,{alpha_deg:.1f},{ctx_pv.CL:.6f},{ctx_pv.CD:.6f},{ctx_pv.CM:.6f}")

            # Euler viscous (Phase 3 loose coupling)
            ctx_ev = build_viscal_context(12, minf=minf, reinf=reinf, alfa_rad=alfa, quiet=True)
            ctx_ev.INVISCID_MODEL = "euler"
            bl_ev = XBlState()
            blpini(bl_ev)
            with contextlib.redirect_stdout(io.StringIO()):
                viscal(ctx_ev, bl_ev, 10)
            print(f"euler,viscous,{alpha_deg:.1f},{ctx_ev.CL:.6f},{ctx_ev.CD:.6f},{ctx_ev.CM:.6f}")

            # Core sanity constraints for requested report.
            self.assertEqual(ctx_pi.CD, 0.0)
            self.assertEqual(ctx_ei.CD, 0.0)
            self.assertGreater(ctx_pv.CD, 0.0)
            self.assertGreater(ctx_ev.CD, 0.0)
            self.assertTrue(math.isfinite(ctx_pi.CL) and math.isfinite(ctx_pi.CM))
            self.assertTrue(math.isfinite(ctx_ei.CL) and math.isfinite(ctx_ei.CM))
            self.assertTrue(math.isfinite(ctx_pv.CL) and math.isfinite(ctx_pv.CM))
            self.assertTrue(math.isfinite(ctx_ev.CL) and math.isfinite(ctx_ev.CM))


if __name__ == "__main__":
    unittest.main()
