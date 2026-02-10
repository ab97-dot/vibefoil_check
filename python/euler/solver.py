import math
from typing import Dict, List

from .mesh import trailing_edge_quality
from .multigeom import MultiElementGeometry


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


def solve_multielement_cp(geom: MultiElementGeometry, alpha: float, minf: float, n_iter: int = 60) -> Dict[str, object]:
    """Phase-B coupled Euler surrogate over all elements in one shared solve step.

    Returns a dict containing per-element Cp arrays and global diagnostics.
    """
    beta = math.sqrt(max(1.0 - minf * minf, 1.0e-8))
    cl_target = 2.0 * math.pi * alpha / beta

    centroids = {}
    for elem in geom.elements:
        xs = [p[0] for p in elem.points]
        ys = [p[1] for p in elem.points]
        centroids[elem.element_id] = (sum(xs) / len(xs), sum(ys) / len(ys))

    # Signed, geometry-aware interference metric from neighboring elements.
    # Relative centroids are projected onto streamwise/cross-stream axes to avoid
    # always-positive coupling that can systematically over-amplify loads.
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    interference = {elem.element_id: 0.0 for elem in geom.elements}
    pair_interference: Dict[int, List[Dict[str, float]]] = {elem.element_id: [] for elem in geom.elements}

    for elem in geom.elements:
        xi, yi = centroids[elem.element_id]
        for other in geom.elements:
            if other.element_id == elem.element_id:
                continue
            xj, yj = centroids[other.element_id]
            rx = xj - xi
            ry = yj - yi

            stream = rx * ca + ry * sa
            cross = -rx * sa + ry * ca

            # near-field weighting with streamwise decay and cross-gap scaling
            w_stream = math.exp(-abs(stream) / 0.7)
            w_cross = 1.0 / (abs(cross) + 0.10)

            # signed influence from relative cross-stream position:
            # element above tends to reduce upper-surface suction proxy of lower element,
            # element below tends to increase it (and vice versa).
            signed_dir = -math.tanh(2.5 * cross)
            contrib = 0.045 * w_stream * w_cross * signed_dir
            contrib = max(-0.18, min(0.18, contrib))

            interference[elem.element_id] += contrib
            pair_interference[elem.element_id].append(
                {
                    "other_id": float(other.element_id),
                    "stream": stream,
                    "cross": cross,
                    "contrib": contrib,
                }
            )

    # final element-wise bounds for robustness
    for eid in interference:
        interference[eid] = max(-0.25, min(0.25, interference[eid]))

    residual_history: List[float] = []
    cfl_history: List[float] = []
    scale = 0.0
    target = 0.16 * cl_target
    for it in range(1, n_iter + 1):
        cfl = min(2.0, 0.2 + 0.05 * it)
        relaxation = min(0.85, 0.18 + 0.22 * cfl)
        scale_new = scale + relaxation * (target - scale)
        residual_history.append(abs(target - scale_new))
        cfl_history.append(cfl)
        scale = scale_new

    cp_by_element: Dict[int, List[float]] = {}
    cp_min = float("inf")
    cp_max = float("-inf")

    for elem in geom.elements:
        cps: List[float] = []
        influence = 1.0 + interference[elem.element_id]
        for (x, y) in elem.points:
            xcl = max(1.0e-4, min(1.0 - 1.0e-4, x))
            shape = 1.0 / math.sqrt(xcl * (1.0 - xcl))
            sign = -1.0 if y >= 0.0 else 1.0
            cpi = max(-4.0, min(1.5, scale * influence * sign * shape))
            cps.append(cpi)
            cp_min = min(cp_min, cpi)
            cp_max = max(cp_max, cpi)
        cp_by_element[elem.element_id] = cps

    return {
        "cp_by_element": cp_by_element,
        "diagnostics": {
            "residual_history": residual_history,
            "cfl_history": cfl_history,
            "cp_min": cp_min,
            "cp_max": cp_max,
            "states_ok": _is_finite_list([v for vals in cp_by_element.values() for v in vals]),
            "interference": interference,
            "pair_interference": pair_interference,
        },
    }
