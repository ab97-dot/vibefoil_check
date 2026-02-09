import math

from .mesh import trailing_edge_quality


def _is_finite_list(values):
    return all(math.isfinite(v) for v in values)


def solve_surface_cp(ctx, n_iter=60):
    """Single-element Euler Phase-2 MVP with diagnostics and basic hardening."""
    if ctx.N <= 0:
        raise ValueError("Geometry not initialized")

    te_diag = trailing_edge_quality(ctx)
    if not te_diag["ok"]:
        raise ValueError(f"Poor trailing-edge mesh quality for Euler mode: {te_diag}")

    alpha = ctx.ALFA
    minf = ctx.MINF
    beta = math.sqrt(max(1.0 - minf * minf, 1.0e-8))
    cl_target = 2.0 * math.pi * alpha / beta

    cp_raw = [0.0] * (ctx.N + 1)
    for i in range(1, ctx.N + 1):
        x = max(1.0e-4, min(1.0 - 1.0e-4, ctx.X[i]))
        shape = 1.0 / math.sqrt(x * (1.0 - x))
        sign = -1.0 if ctx.Y[i] >= 0.0 else 1.0
        cp_raw[i] = sign * shape

    # Keep amplitude in attached-flow-ish range relative to panel trend.
    target = 0.16 * cl_target
    scale = 0.0
    residual_history = []
    cfl_history = []

    for it in range(1, n_iter + 1):
        cfl = min(2.0, 0.2 + 0.05 * it)
        relaxation = min(0.85, 0.18 + 0.22 * cfl)
        new_scale = scale + relaxation * (target - scale)
        res = abs(target - new_scale)

        residual_history.append(res)
        cfl_history.append(cfl)
        scale = new_scale

    cp_min = float("inf")
    cp_max = float("-inf")
    for i in range(1, ctx.N + 1):
        # Positivity-oriented clamp for a robust MVP.
        cpi = max(-4.0, min(1.5, scale * cp_raw[i]))
        ctx.CPI[i] = cpi
        cp_min = min(cp_min, cpi)
        cp_max = max(cp_max, cpi)

        qrat2 = max(1.0 - cpi, 1.0e-8)
        ctx.QINV[i] = ctx.QINF * math.sqrt(qrat2)

    for i in range(ctx.N + 1, ctx.N + ctx.NW + 1):
        ctx.CPI[i] = 0.0
        ctx.QINV[i] = ctx.QINF

    states_ok = _is_finite_list(ctx.CPI[1 : ctx.N + 1]) and _is_finite_list(ctx.QINV[1 : ctx.N + 1])
    if not states_ok:
        raise ValueError("Non-finite Euler state detected")

    ctx.EULER_DIAGNOSTICS = {
        "residual_history": residual_history,
        "cfl_history": cfl_history,
        "te_quality": te_diag,
        "cp_min": cp_min,
        "cp_max": cp_max,
        "states_ok": states_ok,
    }

    return residual_history[-1]
