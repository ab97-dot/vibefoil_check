import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.euler import EulerInviscidCore
from python.inviscid import PanelInviscidCore, get_inviscid_core
from python.xbl import XFoilState


class TestInviscidCoreSelection(unittest.TestCase):
    def test_default_model_is_panel(self):
        ctx = XFoilState()
        self.assertEqual(ctx.INVISCID_MODEL, "panel")
        self.assertIsInstance(get_inviscid_core(ctx), PanelInviscidCore)

    def test_euler_model_selects_euler_core(self):
        ctx = XFoilState()
        ctx.INVISCID_MODEL = "euler"
        self.assertIsInstance(get_inviscid_core(ctx), EulerInviscidCore)


if __name__ == "__main__":
    unittest.main()
