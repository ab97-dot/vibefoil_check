import contextlib
import io
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import python.run_viscous_sweep_euler_user as euler_user


class TestRunViscousSweepEulerUser(unittest.TestCase):
    def test_main_prints_csv_with_cl_cd_cm(self):
        old = {
            "NACA_CODE": euler_user.NACA_CODE,
            "AIRFOIL_DAT_PATH": euler_user.AIRFOIL_DAT_PATH,
            "ALPHAS_DEG_LIST": euler_user.ALPHAS_DEG_LIST,
            "ALPHA_START_DEG": euler_user.ALPHA_START_DEG,
            "ALPHA_END_DEG": euler_user.ALPHA_END_DEG,
            "ALPHA_STEP_DEG": euler_user.ALPHA_STEP_DEG,
            "NITER": euler_user.NITER,
            "VERBOSE": euler_user.VERBOSE,
            "INVISCID_MODEL": euler_user.INVISCID_MODEL,
        }
        try:
            euler_user.NACA_CODE = "0012"
            euler_user.AIRFOIL_DAT_PATH = ""
            euler_user.ALPHAS_DEG_LIST = [4.0, 8.0]
            euler_user.ALPHA_START_DEG = 0.0
            euler_user.ALPHA_END_DEG = 0.0
            euler_user.ALPHA_STEP_DEG = 1.0
            euler_user.NITER = 6
            euler_user.VERBOSE = False
            euler_user.INVISCID_MODEL = "euler"

            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                euler_user.main()

            lines = [ln.strip() for ln in capture.getvalue().splitlines() if ln.strip()]
            self.assertGreaterEqual(len(lines), 3)
            self.assertEqual(lines[0], "alpha_deg,CL,CD,CDp,CM")

            for ln in lines[1:3]:
                parts = ln.split(",")
                self.assertEqual(len(parts), 5)
                # parse float columns; CD should be positive for viscous run
                _alpha, cl, cd, _cdp, cm = [float(x) for x in parts]
                self.assertTrue(abs(cl) < 5.0)
                self.assertGreater(cd, 0.0)
                self.assertTrue(abs(cm) < 2.0)
        finally:
            for k, v in old.items():
                setattr(euler_user, k, v)


if __name__ == "__main__":
    unittest.main()
