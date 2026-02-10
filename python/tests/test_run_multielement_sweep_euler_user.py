import contextlib
import io
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import python.run_multielement_sweep_euler_user as multi

S414_ELEMENT_PATHS = [
    "python/airfoils/s414_element1.dat",
    "python/airfoils/s414_element2.dat",
]


class TestRunMultielementSweepEulerUser(unittest.TestCase):
    def _run_main_capture(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            multi.main()
        return [ln.strip() for ln in out.getvalue().splitlines() if ln.strip()]

    def test_main_prints_per_element_and_total_rows(self):
        self.assertEqual(multi.ELEMENT_DAT_PATHS, S414_ELEMENT_PATHS)
        old = {
            "ELEMENT_DAT_PATHS": multi.ELEMENT_DAT_PATHS,
            "ALPHAS_DEG_LIST": multi.ALPHAS_DEG_LIST,
            "NITER": multi.NITER,
            "VERBOSE": multi.VERBOSE,
            "MULTI_ELEMENT_MODE": multi.MULTI_ELEMENT_MODE,
        }
        try:
            # Multi-element baseline uses S414 two-element geometry.
            multi.ELEMENT_DAT_PATHS = list(S414_ELEMENT_PATHS)
            multi.ALPHAS_DEG_LIST = [4.0, 8.0]
            multi.NITER = 6
            multi.VERBOSE = False
            multi.MULTI_ELEMENT_MODE = "phase_d_iterative"

            lines = self._run_main_capture()
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

    def test_coupled_mode_differs_from_legacy(self):
        old = {
            "ELEMENT_DAT_PATHS": multi.ELEMENT_DAT_PATHS,
            "ALPHAS_DEG_LIST": multi.ALPHAS_DEG_LIST,
            "NITER": multi.NITER,
            "VERBOSE": multi.VERBOSE,
            "MULTI_ELEMENT_MODE": multi.MULTI_ELEMENT_MODE,
        }
        try:
            multi.ELEMENT_DAT_PATHS = list(S414_ELEMENT_PATHS)
            multi.ALPHAS_DEG_LIST = [8.0]
            multi.NITER = 6
            multi.VERBOSE = False

            multi.MULTI_ELEMENT_MODE = "independent_legacy"
            legacy = self._run_main_capture()[1:]

            multi.MULTI_ELEMENT_MODE = "coupled"
            coupled = self._run_main_capture()[1:]

            # both produce element_1, element_2, total rows
            self.assertEqual(len(legacy), 3)
            self.assertEqual(len(coupled), 3)

            # parse CL of element_1; coupling should alter force vs legacy path
            leg_e1 = [r for r in legacy if ",element_1," in r][0].split(",")
            cpl_e1 = [r for r in coupled if ",element_1," in r][0].split(",")
            cl_legacy = float(leg_e1[2])
            cl_coupled = float(cpl_e1[2])
            self.assertGreater(abs(cl_coupled - cl_legacy), 1.0e-8)
        finally:
            for k, v in old.items():
                setattr(multi, k, v)

    def test_phase_d_iterative_mode_converges_and_differs_from_legacy(self):
        old = {
            "ELEMENT_DAT_PATHS": multi.ELEMENT_DAT_PATHS,
            "ALPHAS_DEG_LIST": multi.ALPHAS_DEG_LIST,
            "NITER": multi.NITER,
            "VERBOSE": multi.VERBOSE,
            "MULTI_ELEMENT_MODE": multi.MULTI_ELEMENT_MODE,
            "COUPLED_MAX_ITERS": multi.COUPLED_MAX_ITERS,
            "COUPLED_CL_TOL": multi.COUPLED_CL_TOL,
            "COUPLED_CM_TOL": multi.COUPLED_CM_TOL,
        }
        try:
            multi.ELEMENT_DAT_PATHS = list(S414_ELEMENT_PATHS)
            multi.ALPHAS_DEG_LIST = [8.0]
            multi.NITER = 6
            multi.VERBOSE = False
            multi.COUPLED_MAX_ITERS = 30
            multi.COUPLED_CL_TOL = 1.0
            multi.COUPLED_CM_TOL = 1.0

            # legacy output
            multi.MULTI_ELEMENT_MODE = "independent_legacy"
            legacy = self._run_main_capture()[1:]

            # phase-d direct helper to inspect convergence diagnostics
            multi._patch_gauss_for_python_solver()
            rows, diag = multi._run_alpha_phase_d_iterative(8.0, [pathlib.Path(p) for p in S414_ELEMENT_PATHS])
            self.assertTrue(diag["converged"])
            self.assertGreaterEqual(diag["iters"], 1)
            self.assertLessEqual(diag["iters"], multi.COUPLED_MAX_ITERS)
            hist = diag["coupling_history"]
            self.assertGreaterEqual(hist[0]["max_dCL"], hist[-1]["max_dCL"])
            self.assertGreaterEqual(hist[0]["max_dCM"], hist[-1]["max_dCM"])
            self.assertEqual(rows[-1][0], "total")

            # mode output should still have 3 rows and differ from legacy in element_1 CL
            multi.MULTI_ELEMENT_MODE = "phase_d_iterative"
            phase_d = self._run_main_capture()[1:]
            self.assertEqual(len(phase_d), 3)
            leg_e1 = [r for r in legacy if ",element_1," in r][0].split(",")
            p4_e1 = [r for r in phase_d if ",element_1," in r][0].split(",")
            self.assertGreater(abs(float(p4_e1[2]) - float(leg_e1[2])), 1.0e-8)
        finally:
            for k, v in old.items():
                setattr(multi, k, v)


if __name__ == "__main__":
    unittest.main()
