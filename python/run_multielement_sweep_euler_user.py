#!/usr/bin/env python3
import contextlib
import io
import math
import pathlib
import sys
from typing import Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.euler import load_multielement_geometry, solve_multielement_forces
from python.run_viscous_sweep_euler_user import (  # reuse user-level utilities
    _patch_gauss_for_python_solver,
    build_viscal_context,
)
from python.xbl import XBlState, blpini
from python.xoper import viscal

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# Edit these values directly, then run:
#   python python/run_multielement_sweep_euler_user.py
# -----------------------------------------------------------------------------
ELEMENT_DAT_PATHS: List[str] = [
    "python/airfoils/s414_element1.dat",
    "python/airfoils/s414_element2.dat",
]
REYNOLDS = 1.0e6
MACH = 0.1
WAKLEN = 1.0
NITER = 100
VERBOSE = False

# Phase-C rollout switch:
# - "coupled": single relaxed blend toward shared multi-element Euler forces
# - "phase_d_iterative": iterative relaxed blend with convergence controls and diagnostics
# - "independent_legacy": previous behavior (solve each element independently and sum)
MULTI_ELEMENT_MODE = "phase_d_iterative"

# relaxed blending from BL forces to coupled Euler forces
COUPLED_FORCE_RELAX = 0.12
# Cap total Phase-D blend strength so repeated iterations do not over-drive toward Euler surrogate
PHASE_D_EFFECTIVE_RELAX_CAP = 0.25

# Phase-D iterative coupling controls
COUPLED_MAX_ITERS = 12
COUPLED_CL_TOL = 5.0e-5
COUPLED_CM_TOL = 5.0e-5

# Option A: explicit list of alphas (deg)
ALPHAS_DEG_LIST: List[float] = [4.0, 8.0]

# Option B: generated range (used when ALPHAS_DEG_LIST is empty)
ALPHA_START_DEG = 0.0
ALPHA_END_DEG = 10.0
ALPHA_STEP_DEG = 1.0


def alphas_from_config() -> List[float]:
    if ALPHAS_DEG_LIST:
        return [float(a) for a in ALPHAS_DEG_LIST]

    vals = []
    a = ALPHA_START_DEG
    while a <= ALPHA_END_DEG + 1.0e-12:
        vals.append(round(a, 10))
        a += ALPHA_STEP_DEG
    return vals


def _run_element(alpha_deg: float, dat_path: pathlib.Path):
    alfa = alpha_deg * math.pi / 180.0
    ctx = build_viscal_context(
        ides=None,
        minf=MACH,
        reinf=REYNOLDS,
        alfa_rad=alfa,
        waklen=WAKLEN,
        quiet=not VERBOSE,
        airfoil_dat_path=dat_path,
    )

    cosa = math.cos(ctx.ALFA)
    sina = math.sin(ctx.ALFA)
    for i in range(1, ctx.N + 1):
        ctx.GAM[i] = cosa * ctx.GAMU[i][1] + sina * ctx.GAMU[i][2]

    bl = XBlState()
    blpini(bl)
    if VERBOSE:
        viscal(ctx, bl, NITER)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            viscal(ctx, bl, NITER)

    return ctx


def _run_alpha_legacy(alpha_deg: float, element_paths: List[pathlib.Path]):
    rows = []
    total_cl = 0.0
    total_cd = 0.0
    total_cdp = 0.0
    total_cm = 0.0

    for idx, dat_path in enumerate(element_paths, start=1):
        ctx = _run_element(alpha_deg, dat_path)
        cdp = ctx.CD - ctx.CDF
        rows.append((f"element_{idx}", ctx.CL, ctx.CD, cdp, ctx.CM))
        total_cl += ctx.CL
        total_cd += ctx.CD
        total_cdp += cdp
        total_cm += ctx.CM

    rows.append(("total", total_cl, total_cd, total_cdp, total_cm))
    return rows, {"mode": "independent_legacy"}


def _run_alpha_coupled(alpha_deg: float, element_paths: List[pathlib.Path]):
    # 1) Run BL stacks independently to obtain viscous drag per element (authoritative CD/CDF/CDP)
    viscous_by_element: Dict[int, Dict[str, float]] = {}
    for idx, dat_path in enumerate(element_paths, start=1):
        ctx = _run_element(alpha_deg, dat_path)
        viscous_by_element[idx] = {
            "cl": ctx.CL,
            "cm": ctx.CM,
            "cd": ctx.CD,
            "cdf": ctx.CDF,
            "cdp": ctx.CD - ctx.CDF,
        }

    # 2) Coupled shared Euler solve for force interaction
    geom = load_multielement_geometry(element_paths)
    coupled = solve_multielement_forces(geom, alpha=math.radians(alpha_deg), minf=MACH)

    # 3) Relax Euler forces into BL force baseline; keep BL drag as authoritative
    rows = []
    total_cl = 0.0
    total_cd = 0.0
    total_cdp = 0.0
    total_cm = 0.0
    history = []

    for elem in geom.elements:
        eid = elem.element_id
        v = viscous_by_element[eid]
        e = coupled["elements"][eid]

        cl = (1.0 - COUPLED_FORCE_RELAX) * v["cl"] + COUPLED_FORCE_RELAX * e["CL"]
        cm = (1.0 - COUPLED_FORCE_RELAX) * v["cm"] + COUPLED_FORCE_RELAX * e["CM"]
        cd = v["cd"]
        cdp = v["cdp"]

        history.append(
            {
                "element_id": eid,
                "dCL": abs(cl - v["cl"]),
                "dCM": abs(cm - v["cm"]),
                "dCD": abs(cd - v["cd"]),
            }
        )

        rows.append((f"element_{eid}", cl, cd, cdp, cm))
        total_cl += cl
        total_cd += cd
        total_cdp += cdp
        total_cm += cm

    rows.append(("total", total_cl, total_cd, total_cdp, total_cm))

    diagnostics = {
        "mode": "coupled",
        "euler_diagnostics": coupled["diagnostics"] if coupled is not None else {},
        "coupling_history": history,
    }
    return rows, diagnostics


def _phase_d_effective_iteration_limit() -> int:
    if COUPLED_MAX_ITERS <= 0:
        return 0
    if COUPLED_FORCE_RELAX <= 0.0:
        return 0
    cap = max(0.0, min(PHASE_D_EFFECTIVE_RELAX_CAP, 0.999999))
    if cap <= 0.0:
        return 0
    # effective blend after k steps: 1 - (1-r)^k <= cap
    k_cap = math.floor(math.log(1.0 - cap) / math.log(1.0 - COUPLED_FORCE_RELAX))
    return max(1, min(COUPLED_MAX_ITERS, k_cap))


def _run_alpha_phase_d_iterative(alpha_deg: float, element_paths: List[pathlib.Path]):
    # Start from authoritative viscous quantities.
    viscous_by_element: Dict[int, Dict[str, float]] = {}
    for idx, dat_path in enumerate(element_paths, start=1):
        ctx = _run_element(alpha_deg, dat_path)
        viscous_by_element[idx] = {
            "cl": ctx.CL,
            "cm": ctx.CM,
            "cd": ctx.CD,
            "cdf": ctx.CDF,
            "cdp": ctx.CD - ctx.CDF,
        }

    # Shared geometry for coupled target loads.
    geom = load_multielement_geometry(element_paths)

    cl_curr = {eid: viscous_by_element[eid]["cl"] for eid in viscous_by_element}
    cm_curr = {eid: viscous_by_element[eid]["cm"] for eid in viscous_by_element}

    history = []
    converged = False
    coupled = None
    alpha_rad = math.radians(alpha_deg)
    for it in range(1, _phase_d_effective_iteration_limit() + 1):
        # Recompute coupled Euler target every iteration (true fixed-point style outer loop).
        coupled = solve_multielement_forces(geom, alpha=alpha_rad, minf=MACH)

        max_dcl = 0.0
        max_dcm = 0.0
        max_target_dcl = 0.0
        max_target_dcm = 0.0
        euler_weight = min(1.0, PHASE_D_EFFECTIVE_RELAX_CAP)

        for elem in geom.elements:
            eid = elem.element_id
            target_raw = coupled["elements"][eid]
            # Blend Euler target with current iterate so the target can evolve with the outer loop.
            target_cl = (1.0 - euler_weight) * cl_curr[eid] + euler_weight * target_raw["CL"]
            target_cm = (1.0 - euler_weight) * cm_curr[eid] + euler_weight * target_raw["CM"]

            cl_new = (1.0 - COUPLED_FORCE_RELAX) * cl_curr[eid] + COUPLED_FORCE_RELAX * target_cl
            cm_new = (1.0 - COUPLED_FORCE_RELAX) * cm_curr[eid] + COUPLED_FORCE_RELAX * target_cm

            max_dcl = max(max_dcl, abs(cl_new - cl_curr[eid]))
            max_dcm = max(max_dcm, abs(cm_new - cm_curr[eid]))
            max_target_dcl = max(max_target_dcl, abs(target_cl - cl_curr[eid]))
            max_target_dcm = max(max_target_dcm, abs(target_cm - cm_curr[eid]))

            cl_curr[eid] = cl_new
            cm_curr[eid] = cm_new

        history.append(
            {
                "iter": it,
                "max_dCL": max_dcl,
                "max_dCM": max_dcm,
                "max_target_dCL": max_target_dcl,
                "max_target_dCM": max_target_dcm,
            }
        )
        if max_dcl <= COUPLED_CL_TOL and max_dcm <= COUPLED_CM_TOL:
            converged = True
            break

    rows = []
    total_cl = 0.0
    total_cd = 0.0
    total_cdp = 0.0
    total_cm = 0.0
    for elem in geom.elements:
        eid = elem.element_id
        v = viscous_by_element[eid]
        cl = cl_curr[eid]
        cm = cm_curr[eid]
        cd = v["cd"]
        cdp = v["cdp"]
        rows.append((f"element_{eid}", cl, cd, cdp, cm))
        total_cl += cl
        total_cd += cd
        total_cdp += cdp
        total_cm += cm

    rows.append(("total", total_cl, total_cd, total_cdp, total_cm))
    diagnostics = {
        "mode": "phase_d_iterative",
        "converged": converged,
        "iters": len(history),
        "tolerances": {"dCL": COUPLED_CL_TOL, "dCM": COUPLED_CM_TOL},
        "effective_relax_cap": PHASE_D_EFFECTIVE_RELAX_CAP,
        "effective_iter_limit": _phase_d_effective_iteration_limit(),
        "euler_diagnostics": coupled["diagnostics"] if coupled is not None else {},
        "coupling_history": history,
    }
    return rows, diagnostics


def main():
    _patch_gauss_for_python_solver()

    element_paths = [pathlib.Path(p).expanduser() for p in ELEMENT_DAT_PATHS]
    alphas_deg = alphas_from_config()

    print("alpha_deg,element,CL,CD,CDp,CM")
    for alpha_deg in alphas_deg:
        if MULTI_ELEMENT_MODE == "independent_legacy":
            rows, _diag = _run_alpha_legacy(alpha_deg, element_paths)
        elif MULTI_ELEMENT_MODE == "coupled":
            rows, _diag = _run_alpha_coupled(alpha_deg, element_paths)
        elif MULTI_ELEMENT_MODE == "phase_d_iterative":
            rows, _diag = _run_alpha_phase_d_iterative(alpha_deg, element_paths)
        else:
            raise ValueError(f"Unknown MULTI_ELEMENT_MODE: {MULTI_ELEMENT_MODE}")

        for element_name, cl, cd, cdp, cm in rows:
            print(f"{alpha_deg:.6g},{element_name},{cl:.6f},{cd:.6f},{cdp:.6f},{cm:.6f}")


if __name__ == "__main__":
    main()
