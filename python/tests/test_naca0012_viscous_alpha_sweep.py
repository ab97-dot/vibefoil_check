import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import python.xbl as xbl_mod
from python.xbl import XBlState, blpini
from python.xoper import viscal
from python.xsolve import gauss as gauss_base
from python.tests.test_compare_viscal import build_viscal_context


class TestNaca0012ViscousAlphaSweep(unittest.TestCase):
    def test_naca0012_viscous_alpha_sweep(self):
        def gauss1(nsiz, nn, z, r, nrhs):
            rmat = [[0.0] * (nrhs + 1) for _ in range(nn + 1)]
            for i in range(1, nn + 1):
                rmat[i][1] = r[i]
            gauss_base(nsiz, nn, z, rmat, nrhs)
            for i in range(1, nn + 1):
                r[i] = rmat[i][1]

        xbl_mod.gauss = gauss1

        reinf = 1.0e6
        minf = 0.0
        waklen = 1.0

        cls = []
        cds = []
        cms = []

        for alpha_deg in range(0, 11):
            alfa = alpha_deg * math.pi / 180.0
            ctx = build_viscal_context(12, minf, reinf, alfa, waklen)
            cosa = math.cos(ctx.ALFA)
            sina = math.sin(ctx.ALFA)
            for i in range(1, ctx.N + 1):
                ctx.GAM[i] = cosa * ctx.GAMU[i][1] + sina * ctx.GAMU[i][2]

            bl = XBlState()
            blpini(bl)
            viscal(ctx, bl, 10)

            with self.subTest(alpha_deg=alpha_deg):
                self.assertTrue(math.isfinite(ctx.CL), "CL must be finite")
                self.assertTrue(math.isfinite(ctx.CD), "CD must be finite")
                self.assertTrue(math.isfinite(ctx.CM), "CM must be finite")
                self.assertGreater(ctx.CD, 0.0, "CD must be positive")

            cls.append(ctx.CL)
            cds.append(ctx.CD)
            cms.append(ctx.CM)

        self.assertLess(abs(cls[0]), 0.2, "CL at 0 deg should be near zero for NACA 0012")

        for i in range(1, len(cls)):
            self.assertGreaterEqual(
                cls[i],
                cls[i - 1] - 0.05,
                msg=f"CL should not drop sharply between {i-1} and {i} deg",
            )


if __name__ == "__main__":
    unittest.main()
