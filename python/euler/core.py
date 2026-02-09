import math

from .solver import solve_surface_cp


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

        # pressure force on body element
        dfx = -cp * dy
        dfy = cp * dx

        # convert to lift axis
        dcl = -dfx * sa + dfy * ca
        cl += dcl

        xm = 0.5 * (ctx.X[i] + ctx.X[j]) - xref
        ym = 0.5 * (ctx.Y[i] + ctx.Y[j]) - yref
        cm += xm * dfy - ym * dfx

    return cl, cm


def _apply_bl_pressure_coupling(ctx, bl_coupling, relax):
    coupling = max(0.0, float(bl_coupling))
    relax = max(0.0, min(1.0, float(relax)))
    if coupling <= 0.0 or relax <= 0.0:
        return

    strength = 0.7 * coupling
    for i in range(1, ctx.N + 1):
        side_sign = -1.0 if ctx.Y[i] >= 0.0 else 1.0
        delta_cp = side_sign * strength
        cpi = ctx.CPI[i] + relax * delta_cp
        ctx.CPI[i] = max(-4.0, min(1.5, cpi))

        qrat2 = max(1.0 - ctx.CPI[i], 1.0e-8)
        ctx.QINV[i] = ctx.QINF * math.sqrt(qrat2)


class EulerInviscidCore:
    name = "euler"

    @staticmethod
    def update_pressure_coefficients(ctx, bl_coupling=0.0, relax=0.0):
        if getattr(ctx, "N", 0) <= 0:
            raise RuntimeError("Euler core requires single-element geometry paneling first")
        solve_surface_cp(ctx)
        _apply_bl_pressure_coupling(ctx, bl_coupling, relax)

    @staticmethod
    def update_force_coefficients(ctx, recompute_pressure=True, bl_coupling=0.0, relax=0.0):
        if recompute_pressure:
            EulerInviscidCore.update_pressure_coefficients(ctx, bl_coupling=bl_coupling, relax=relax)
        ctx.CL, ctx.CM = _integrate_forces_from_cp(ctx)
        ctx.CDP = 0.0
        ctx.CL_ALF = 2.0 * math.pi / math.sqrt(max(1.0 - ctx.MINF * ctx.MINF, 1.0e-8))
        ctx.CL_MSQ = 0.0
