import contextlib
import io
import math
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import python.run_inviscid_sweep as sweep
from python.xbl import XFoilState
from python.xfoil import naca


class TestRunInviscidSweep(unittest.TestCase):
    def test_build_inviscid_context_naca(self):
        ctx = sweep.build_inviscid_context(12, minf=0.0, alfa_rad=0.0, quiet=True)
        self.assertEqual(ctx.N, 160)

    def test_build_inviscid_context_dat(self):
        source = XFoilState()
        source.NPAN = 160
        with contextlib.redirect_stdout(io.StringIO()):
            naca(source, 12)

        with tempfile.TemporaryDirectory() as tmpdir:
            dat_path = pathlib.Path(tmpdir) / "naca0012.dat"
            with dat_path.open("w", encoding="utf-8") as f:
                f.write("NACA0012 via DAT\n")
                for i in range(1, source.NB + 1):
                    f.write(f"{source.XB[i]:.8f} {source.YB[i]:.8f}\n")

            ctx = sweep.build_inviscid_context(
                None,
                minf=0.0,
                alfa_rad=0.0 * math.pi / 180.0,
                quiet=True,
                airfoil_dat_path=dat_path,
            )

            self.assertEqual(ctx.N, 160)
            self.assertEqual(ctx.NAME, "NACA0012 via DAT")


if __name__ == "__main__":
    unittest.main()
