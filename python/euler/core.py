import math
from typing import Dict

from .multigeom import MultiElementGeometry
from .solver import solve_multielement_cp, solve_surface_cp


def _integrate_forces_from_cp(ctx):
    ca = math.cos(ctx.ALFA)
    sa = math.sin(ctx.ALFA)
    cl = 0.0
    cm = 0.0
    xref = ctx.XCMREF
    yref = ctx.YCMREF

    for i in range(1, ctx.N + 1):
        j = 1 if i == ctx.N else i + 1
        cp_i = ctx.CPI[i]
        cp_j = ctx.CPI[j]
        cp = 0.5 * (cp_i + cp_j)

        dx = ctx.X[j] - ctx.X[i]
        dy = ctx.Y[j] - ctx.Y[i]

        dfx = -cp * dy
        dfy = cp * dx

        dcl = -dfx * sa + dfy * ca
        cl += dcl

        xm = 0.5 * (ctx.X[i] + ctx.X[j]) - xref
        ym = 0.5 * (ctx.Y[i] + ctx.Y[j]) - yref
        cm += xm * dfy - ym * dfx

    return cl, cm


def _integrate_element_forces(points, cp_vals, alpha, xref, yref):
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    cl = 0.0
    cm = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        cp = 0.5 * (cp_vals[i] + cp_vals[j])
        x1, y1 = points[i]
        x2, y2 = points[j]
        dx = x2 - x1
        dy = y2 - y1

        dfx = -cp * dy
        dfy = cp * dx
        cl += -dfx * sa + dfy * ca

        xm = 0.5 * (x1 + x2) - xref
        ym = 0.5 * (y1 + y2) - yref
        cm += xm * dfy - ym * dfx
    return cl, cm


def solve_multielement_forces(geom: MultiElementGeometry, alpha: float, minf: float, xref: float = 0.25, yref: float = 0.0):
    """Phase-B coupled multi-element force extraction from one shared Euler solve."""
    sol = solve_multielement_cp(geom, alpha, minf)
    cp_by_element = sol["cp_by_element"]

    element_forces: Dict[int, Dict[str, float]] = {}
    cl_total = 0.0
    cm_total = 0.0
    for elem in geom.elements:
        cl_i, cm_i = _integrate_element_forces(elem.points, cp_by_element[elem.element_id], alpha, xref, yref)
        element_forces[elem.element_id] = {
            "name": elem.name,
            "marker": elem.boundary_marker,
            "CL": cl_i,
            "CD": 0.0,
            "CM": cm_i,
        }
        cl_total += cl_i
        cm_total += cm_i

    return {
        "elements": element_forces,
        "total": {"CL": cl_total, "CD": 0.0, "CM": cm_total},
        "diagnostics": sol["diagnostics"],
    }


class EulerInviscidCore:
    name = "euler"

    @staticmethod
    def update_pressure_coefficients(ctx):
        if getattr(ctx, "N", 0) <= 0:
            raise RuntimeError("Euler core requires single-element geometry paneling first")
        solve_surface_cp(ctx)

        disp = max(0.0, min(0.2, getattr(ctx, "EULER_BL_DISP", 0.0)))
        load_relax = 1.0 - 0.6 * disp
        for i in range(1, ctx.N + 1):
            ctx.CPI[i] = ctx.CPI[i] * load_relax

    @staticmethod
    def update_force_coefficients(ctx):
        EulerInviscidCore.update_pressure_coefficients(ctx)
        ctx.CL, ctx.CM = _integrate_forces_from_cp(ctx)
        ctx.CDP = 0.0
        ctx.CL_ALF = 2.0 * math.pi / math.sqrt(max(1.0 - ctx.MINF * ctx.MINF, 1.0e-8))
        ctx.CL_MSQ = 0.0
