#!/usr/bin/env python3
import contextlib
import io
import math
import pathlib
import re
import sys
from typing import List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.xbl import XFoilState
from python.inviscid import get_inviscid_core
from python.xfoil import comset, naca, pangen
from python.xpanel import ggcalc

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# Edit these values directly, then run:
#   python python/run_inviscid_sweep.py
# -----------------------------------------------------------------------------
NACA_CODE = "0012"          # 4- or 5-digit NACA code string, e.g. "0012" or "23012"
AIRFOIL_DAT_PATH = ""       # Optional path to a Selig-format .dat file. If set, NACA_CODE is ignored.
TE_THICKNESS_FRAC = 0.002   # Trailing-edge thickness fraction of chord for DAT airfoils (blunt TE target).
MACH = 0.0                  # Mach number (MINF)
VERBOSE = False             # True to print panel setup logs
INVISCID_MODEL = "panel"     # "panel" or "euler"

# Option A: explicit list of alphas (deg)
ALPHAS_DEG_LIST: List[float] = []  # e.g. [0, 2, 4, 6, 8, 10]

# Option B: generated range (used when ALPHAS_DEG_LIST is empty)
ALPHA_START_DEG = 0.0
ALPHA_END_DEG = 10.0
ALPHA_STEP_DEG = 1.0


def parse_selig_dat(dat_path: pathlib.Path) -> Tuple[str, List[Tuple[float, float]]]:
    lines = dat_path.read_text(encoding="utf-8").splitlines()

    name = dat_path.stem
    coords: List[Tuple[float, float]] = []
    seen_coords = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        fields = [part for part in re.split(r"[\s,]+", line) if part]
        if len(fields) < 2:
            if not seen_coords:
                name = line
            continue

        try:
            x = float(fields[0])
            y = float(fields[1])
        except ValueError:
            if not seen_coords:
                name = line
            continue

        coords.append((x, y))
        seen_coords = True

    if len(coords) < 3:
        raise ValueError(f"Selig .dat file must contain at least 3 coordinate pairs: {dat_path}")

    return name, coords


def enforce_blunt_te(coords: List[Tuple[float, float]], te_thickness_frac: float) -> List[Tuple[float, float]]:
    if te_thickness_frac < 0.0:
        raise ValueError("TE_THICKNESS_FRAC must be >= 0")

    xs = [xy[0] for xy in coords]
    chord = max(xs) - min(xs)
    if chord <= 0.0:
        raise ValueError("Invalid DAT geometry: chord length must be positive")

    target_gap = te_thickness_frac * chord

    x_u, y_u = coords[0]
    x_l, y_l = coords[-1]
    y_mid = 0.5 * (y_u + y_l)

    sign = 1.0 if (y_u - y_l) >= 0.0 else -1.0
    half_gap = 0.5 * target_gap

    new_coords = list(coords)
    new_coords[0] = (x_u, y_mid + sign * half_gap)
    new_coords[-1] = (x_l, y_mid - sign * half_gap)
    return new_coords


def load_airfoil_dat(ctx: XFoilState, dat_path: pathlib.Path, te_thickness_frac: float):
    name, coords = parse_selig_dat(dat_path)
    coords = enforce_blunt_te(coords, te_thickness_frac)

    if len(coords) >= len(ctx.XB):
        raise ValueError(
            f"Too many points in {dat_path} ({len(coords)}); maximum supported is {len(ctx.XB) - 1}."
        )

    ctx.NB = len(coords)
    ctx.NAME = name
    ctx.NNAME = len(name)

    for idx, (x, y) in enumerate(coords, start=1):
        ctx.XB[idx] = x
        ctx.YB[idx] = y

    ctx.LCLOCK = False
    ctx.XBF = 0.0
    ctx.YBF = 0.0
    ctx.LBFLAP = False

    pangen(ctx, True)


def build_inviscid_context(
    ides: Optional[int],
    minf: float,
    alfa_rad: float,
    quiet: bool = True,
    airfoil_dat_path: Optional[pathlib.Path] = None,
) -> XFoilState:
    ctx = XFoilState()
    ctx.NPAN = 160
    ctx.CVPAR = 1.0
    ctx.CTERAT = 0.15
    ctx.CTRRAT = 0.2
    ctx.XSREF1 = 1.0
    ctx.XSREF2 = 1.0
    ctx.XPREF1 = 1.0
    ctx.XPREF2 = 1.0

    ctx.ALFA = alfa_rad
    ctx.ADEG = alfa_rad / ctx.DTOR
    ctx.QINF = 1.0
    ctx.MINF = minf
    ctx.MINF1 = minf
    ctx.LALFA = True
    ctx.LVISC = False
    ctx.XCMREF = 0.25
    ctx.YCMREF = 0.0
    ctx.INVISCID_MODEL = INVISCID_MODEL

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            if airfoil_dat_path is not None:
                load_airfoil_dat(ctx, airfoil_dat_path, TE_THICKNESS_FRAC)
            else:
                if ides is None:
                    raise ValueError("ides must be provided when AIRFOIL_DAT_PATH is not set")
                naca(ctx, ides)
            comset(ctx)
            ggcalc(ctx)
    else:
        if airfoil_dat_path is not None:
            load_airfoil_dat(ctx, airfoil_dat_path, TE_THICKNESS_FRAC)
        else:
            if ides is None:
                raise ValueError("ides must be provided when AIRFOIL_DAT_PATH is not set")
            naca(ctx, ides)
        comset(ctx)
        ggcalc(ctx)

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
    airfoil_dat_path = pathlib.Path(AIRFOIL_DAT_PATH).expanduser() if AIRFOIL_DAT_PATH else None
    naca_code = None if airfoil_dat_path is not None else parse_naca(NACA_CODE)
    alphas_deg = alphas_from_config()

    print("alpha_deg,CL,CD,CDp,CM")
    for alpha_deg in alphas_deg:
        alfa = alpha_deg * math.pi / 180.0
        ctx = build_inviscid_context(
            naca_code,
            MACH,
            alfa,
            quiet=not VERBOSE,
            airfoil_dat_path=airfoil_dat_path,
        )

        cosa = math.cos(ctx.ALFA)
        sina = math.sin(ctx.ALFA)
        for i in range(1, ctx.N + 1):
            ctx.GAM[i] = cosa * ctx.GAMU[i][1] + sina * ctx.GAMU[i][2]

        get_inviscid_core(ctx).update_force_coefficients(ctx)
        cdp = ctx.CDP
        ctx.CD = 0.0
        ctx.CDF = 0.0

        print(f"{alpha_deg:.6g},{ctx.CL:.6f},{ctx.CD:.6f},{cdp:.6f},{ctx.CM:.6f}")


if __name__ == "__main__":
    main()
