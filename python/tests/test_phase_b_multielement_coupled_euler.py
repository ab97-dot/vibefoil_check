import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.euler import load_multielement_geometry, solve_multielement_forces


class TestPhaseBMultielementCoupledEuler(unittest.TestCase):
    def test_per_element_and_total_consistency(self):
        geom = load_multielement_geometry(
            [
                pathlib.Path("python/airfoils/s414_element1.dat"),
                pathlib.Path("python/airfoils/s414_element2.dat"),
            ]
        )
        res = solve_multielement_forces(geom, alpha=math.radians(4.0), minf=0.0)

        e1 = res["elements"][1]
        e2 = res["elements"][2]
        tot = res["total"]

        self.assertAlmostEqual(tot["CL"], e1["CL"] + e2["CL"], places=10)
        self.assertAlmostEqual(tot["CM"], e1["CM"] + e2["CM"], places=10)
        self.assertEqual(tot["CD"], 0.0)
        self.assertEqual(e1["CD"], 0.0)
        self.assertEqual(e2["CD"], 0.0)

        self.assertTrue(res["diagnostics"]["states_ok"])
        self.assertLess(res["diagnostics"]["residual_history"][-1], res["diagnostics"]["residual_history"][0])

    def test_coupling_effect_changes_element1_when_element2_present(self):
        single = load_multielement_geometry([pathlib.Path("python/airfoils/s414_element1.dat")])
        coupled = load_multielement_geometry(
            [
                pathlib.Path("python/airfoils/s414_element1.dat"),
                pathlib.Path("python/airfoils/s414_element2.dat"),
            ]
        )

        sres = solve_multielement_forces(single, alpha=math.radians(8.0), minf=0.0)
        cres = solve_multielement_forces(coupled, alpha=math.radians(8.0), minf=0.0)

        cl_single = sres["elements"][1]["CL"]
        cl_coupled = cres["elements"][1]["CL"]

        self.assertGreater(abs(cl_coupled - cl_single), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
