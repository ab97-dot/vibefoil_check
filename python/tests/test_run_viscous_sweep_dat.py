import contextlib
import io
import math
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.run_viscous_sweep import build_viscal_context, enforce_blunt_te, parse_selig_dat
from python.xbl import XFoilState
from python.xfoil import naca


class TestRunViscousSweepDat(unittest.TestCase):
    def test_parse_selig_dat_reads_header_and_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dat_path = pathlib.Path(tmpdir) / "sample.dat"
            dat_path.write_text("Sample Airfoil\n1.0 0.0\n0.0 0.1\n1.0 -0.0\n", encoding="utf-8")

            name, coords = parse_selig_dat(dat_path)

            self.assertEqual(name, "Sample Airfoil")
            self.assertEqual(len(coords), 3)
            self.assertEqual(coords[1], (0.0, 0.1))

    def test_enforce_blunt_te(self):
        coords = [(1.0, 0.10), (0.0, 0.0), (1.0, -0.05)]
        out = enforce_blunt_te(coords, 0.02)
        self.assertAlmostEqual(out[0][1] - out[-1][1], 0.02, places=12)

    def test_build_viscal_context_from_dat(self):
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

            ctx = build_viscal_context(
                None,
                minf=0.0,
                reinf=1.0e6,
                alfa_rad=0.0 * math.pi / 180.0,
                waklen=1.0,
                quiet=True,
                airfoil_dat_path=dat_path,
            )

            self.assertEqual(ctx.N, 160)
            self.assertEqual(ctx.NAME, "NACA0012 via DAT")
            chord = max(source.XB[1:source.NB + 1]) - min(source.XB[1:source.NB + 1])
            target = 0.002 * chord
            self.assertAlmostEqual(ctx.YB[1] - ctx.YB[ctx.NB], target, places=7)


if __name__ == "__main__":
    unittest.main()
