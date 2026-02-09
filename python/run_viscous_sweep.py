#!/usr/bin/env python3
import contextlib
import io
import math
import pathlib
import sys
from typing import List

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import python.xbl as xbl_mod
from python.xbl import XFoilState, XBlState, blpini
from python.xfoil import comset, naca
from python.xpanel import ggcalc
from python.xoper import viscal
from python.xsolve import gauss as gauss_base

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# Edit these values directly, then run:
#   python python/run_viscous_sweep.py
# -----------------------------------------------------------------------------
NACA_CODE = "0012"          # 4- or 5-digit NACA code string, e.g. "0012" or "23012"
REYNOLDS = 1.0e6            # Reynolds number
MACH = 0.0                  # Mach number (MINF)
WAKLEN = 1.0                # Wake length parameter
NITER = 10                  # VISCAL iterations per alpha
VERBOSE = False             # True to print full solver logs

# Option A: explicit list of alphas (deg)
ALPHAS_DEG_LIST: List[float] = []  # e.g. [0, 2, 4, 6, 8, 10]

# Option B: generated range (used when ALPHAS_DEG_LIST is empty)
ALPHA_START_DEG = 0.0
ALPHA_END_DEG = 10.0
ALPHA_STEP_DEG = 1.0



def _patch_gauss_for_python_solver():
    def gauss1(nsiz, nn, z, r, nrhs):
        rmat = [[0.0] * (nrhs + 1) for _ in range(nn + 1)]
        for i in range(1, nn + 1):
            rmat[i][1] = r[i]
        gauss_base(nsiz, nn, z, rmat, nrhs)
        for i in range(1, nn + 1):
            r[i] = rmat[i][1]

    xbl_mod.gauss = gauss1


def build_viscal_context(ides: int, minf: float, reinf: float, alfa_rad: float, waklen: float = 1.0, quiet: bool = True) -> XFoilState:
    ctx = XFoilState()
    ctx.NPAN = 160
    ctx.CVPAR = 1.0
    ctx.CTERAT = 0.15
    ctx.CTRRAT = 0.2
    ctx.XSREF1 = 1.0
    ctx.XSREF2 = 1.0
    ctx.XPREF1 = 1.0
    ctx.XPREF2 = 1.0

    ctx.WAKLEN = waklen
    ctx.ALFA = alfa_rad
    ctx.ADEG = alfa_rad / ctx.DTOR
    ctx.QINF = 1.0
    ctx.MINF = minf
    ctx.MINF1 = minf
    ctx.REINF = reinf
    ctx.REINF1 = reinf
    ctx.LALFA = True
    ctx.LVISC = True
    ctx.VACCEL = 0.01
    ctx.XCMREF = 0.25
    ctx.YCMREF = 0.0

    ctx.ACRIT[1] = 9.0
    ctx.ACRIT[2] = 9.0
    ctx.XSTRIP[1] = 1.0
    ctx.XSTRIP[2] = 1.0

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            naca(ctx, ides)
            comset(ctx)
            ggcalc(ctx)
    else:
        naca(ctx, ides)
        comset(ctx)
        ggcalc(ctx)

    for i in range(1, ctx.N + 1):
        ctx.GAM[i] = 1.0 if i <= ctx.N // 2 else -1.0

    return ctx


def parse_naca(code: str) -> int:
    if not code.isdigit() or len(code) not in (4, 5):
        raise ValueError("NACA_CODE must be a 4- or 5-digit code, e.g. '0012' or '23012'")
    return int(code)


def alphas_from_config() -> List[float]:
    if ALPHAS_DEG_LIST:
        return [float(a) for a in ALPHAS_DEG_LIST]

    vals = []
    a = ALPHA_START_DEG
    while a <= ALPHA_END_DEG + 1.0e-12:
        vals.append(round(a, 10))
        a += ALPHA_STEP_DEG
    return vals


def main():
    _patch_gauss_for_python_solver()

    naca_code = parse_naca(NACA_CODE)
    alphas_deg = alphas_from_config()

    print("alpha_deg,CL,CD,CDp,Cm")
    for alpha_deg in alphas_deg:
        alfa = alpha_deg * math.pi / 180.0
        ctx = build_viscal_context(naca_code, MACH, REYNOLDS, alfa, WAKLEN, quiet=not VERBOSE)

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

        cdp = ctx.CD - ctx.CDF
        print(f"{alpha_deg:.6g},{ctx.CL:.6f},{ctx.CD:.6f},{cdp:.6f},{ctx.CM:.6f}")


if __name__ == "__main__":
    main()
