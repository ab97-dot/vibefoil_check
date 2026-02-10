# Ported from XFOIL Fortran source (Mark Drela).
# This file is a derived work and remains under the terms of the
# GNU General Public License v2 or later.
# See https://web.mit.edu/drela/Public/web/xfoil/ for the original code and license text.

from .euler import EulerInviscidCore
from .xfoil import clcalc, cpcalc


class PanelInviscidCore:
    """Adapter around the legacy panel-based inviscid routines."""

    name = "panel"

    @staticmethod
    def update_force_coefficients(ctx):
        ctx.CL, ctx.CM, ctx.CDP, ctx.CL_ALF, ctx.CL_MSQ = clcalc(
            ctx.N,
            ctx.X,
            ctx.Y,
            ctx.GAM,
            ctx.GAM_A,
            ctx.ALFA,
            ctx.MINF,
            ctx.QINF,
            ctx.XCMREF,
            ctx.YCMREF,
        )

    @staticmethod
    def update_pressure_coefficients(ctx):
        if ctx.LVISC:
            cpcalc(ctx.N + ctx.NW, ctx.QVIS, ctx.QINF, ctx.MINF, ctx.CPV)
            cpcalc(ctx.N + ctx.NW, ctx.QINV, ctx.QINF, ctx.MINF, ctx.CPI)
        elif ctx.LWAKE:
            cpcalc(ctx.N + ctx.NW, ctx.QINV, ctx.QINF, ctx.MINF, ctx.CPI)
        else:
            cpcalc(ctx.N, ctx.QINV, ctx.QINF, ctx.MINF, ctx.CPI)


_PANEL_CORE = PanelInviscidCore()


def get_inviscid_core(ctx):
    model = getattr(ctx, "INVISCID_MODEL", "panel")
    if model == "panel":
        return _PANEL_CORE
    if model == "euler":
        return EulerInviscidCore()
    raise ValueError(f"Unsupported INVISCID_MODEL: {model}")
