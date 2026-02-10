import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.euler.mesh import build_global_boundary_connectivity
from python.euler.multigeom import load_multielement_geometry


class TestEulerMultiElementPhaseA(unittest.TestCase):
    def test_load_multielement_geometry_preserves_ids_and_markers(self):
        paths = [
            pathlib.Path("python/airfoils/s414_element1.dat"),
            pathlib.Path("python/airfoils/s414_element2.dat"),
        ]
        geom = load_multielement_geometry(paths)

        self.assertEqual(len(geom.elements), 2)
        self.assertEqual(geom.elements[0].element_id, 1)
        self.assertEqual(geom.elements[1].element_id, 2)
        self.assertEqual(geom.elements[0].boundary_marker, "wall_e1")
        self.assertEqual(geom.elements[1].boundary_marker, "wall_e2")
        self.assertGreater(len(geom.elements[0].points), 10)
        self.assertGreater(len(geom.elements[1].points), 10)

    def test_build_global_boundary_connectivity_has_expected_markers(self):
        paths = [
            pathlib.Path("python/airfoils/s414_element1.dat"),
            pathlib.Path("python/airfoils/s414_element2.dat"),
        ]
        geom = load_multielement_geometry(paths)
        mesh = build_global_boundary_connectivity(geom)

        markers = mesh["boundary_markers"]
        self.assertIn("farfield", markers)
        self.assertIn("wall_e1", markers)
        self.assertIn("wall_e2", markers)
        self.assertEqual(markers["farfield"], 0)
        self.assertEqual(markers["wall_e1"], 1)
        self.assertEqual(markers["wall_e2"], 2)

        self.assertEqual(len(mesh["farfield_edges"]), 4)
        self.assertGreater(len(mesh["wall_edges"]), 20)


if __name__ == "__main__":
    unittest.main()
