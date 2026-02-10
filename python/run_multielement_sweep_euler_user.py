#!/usr/bin/env python3
import contextlib
import io
import math
import pathlib
import sys
from typing import List

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
MACH = 0.0
WAKLEN = 1.0
NITER = 10
VERBOSE = False

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


def main():
    _patch_gauss_for_python_solver()

    element_paths = [pathlib.Path(p).expanduser() for p in ELEMENT_DAT_PATHS]
    alphas_deg = alphas_from_config()

    print("alpha_deg,element,CL,CD,CDp,CM")
    for alpha_deg in alphas_deg:
        total_cl = 0.0
        total_cd = 0.0
        total_cdp = 0.0
        total_cm = 0.0

        for idx, dat_path in enumerate(element_paths, start=1):
            ctx = _run_element(alpha_deg, dat_path)
            cdp = ctx.CD - ctx.CDF
            print(f"{alpha_deg:.6g},element_{idx},{ctx.CL:.6f},{ctx.CD:.6f},{cdp:.6f},{ctx.CM:.6f}")

            total_cl += ctx.CL
            total_cd += ctx.CD
            total_cdp += cdp
            total_cm += ctx.CM

        print(f"{alpha_deg:.6g},total,{total_cl:.6f},{total_cd:.6f},{total_cdp:.6f},{total_cm:.6f}")


if __name__ == "__main__":
    main()
