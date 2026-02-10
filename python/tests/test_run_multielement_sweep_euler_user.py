import contextlib
import io
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import python.run_multielement_sweep_euler_user as multi


class TestRunMultielementSweepEulerUser(unittest.TestCase):
    def test_main_prints_per_element_and_total_rows(self):
        old = {
            "ELEMENT_DAT_PATHS": multi.ELEMENT_DAT_PATHS,
            "ALPHAS_DEG_LIST": multi.ALPHAS_DEG_LIST,
            "NITER": multi.NITER,
            "VERBOSE": multi.VERBOSE,
        }
        try:
            multi.ELEMENT_DAT_PATHS = [
                "python/airfoils/s414_element1.dat",
                "python/airfoils/s414_element2.dat",
            ]
            multi.ALPHAS_DEG_LIST = [4.0, 8.0]
            multi.NITER = 6
            multi.VERBOSE = False

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                multi.main()

            lines = [ln.strip() for ln in out.getvalue().splitlines() if ln.strip()]
            self.assertEqual(lines[0], "alpha_deg,element,CL,CD,CDp,CM")

            # 2 alphas * (2 elements + total) = 6 data lines
            data = lines[1:]
            self.assertEqual(len(data), 6)

            by_alpha = {4.0: [], 8.0: []}
            for row in data:
                a, element, cl, cd, cdp, cm = row.split(",")
                a = float(a)
                vals = (element, float(cl), float(cd), float(cdp), float(cm))
                by_alpha[a].append(vals)

            for a in (4.0, 8.0):
                rows = by_alpha[a]
                self.assertEqual(len(rows), 3)
                elem_rows = [r for r in rows if r[0].startswith("element_")]
                total_row = [r for r in rows if r[0] == "total"]
                self.assertEqual(len(elem_rows), 2)
                self.assertEqual(len(total_row), 1)

                sum_cl = sum(r[1] for r in elem_rows)
                sum_cd = sum(r[2] for r in elem_rows)
                sum_cdp = sum(r[3] for r in elem_rows)
                sum_cm = sum(r[4] for r in elem_rows)

                t = total_row[0]
                self.assertAlmostEqual(t[1], sum_cl, places=5)
                self.assertAlmostEqual(t[2], sum_cd, places=5)
                self.assertAlmostEqual(t[3], sum_cdp, places=5)
                self.assertAlmostEqual(t[4], sum_cm, places=5)
        finally:
            for k, v in old.items():
                setattr(multi, k, v)


if __name__ == "__main__":
    unittest.main()
