import math
from typing import Dict, List, Tuple

from .multigeom import MultiElementGeometry


Point = Tuple[float, float]


def build_surface_connectivity(ctx):
    edges = []
    for i in range(1, ctx.N + 1):
        j = 1 if i == ctx.N else i + 1
        edges.append((i, j))
    return edges



def trailing_edge_quality(ctx):
    if ctx.N < 2:
        return {"ok": False, "reason": "insufficient_points", "te_gap": float("inf")}

    te_gap = math.hypot(ctx.X[1] - ctx.X[ctx.N], ctx.Y[1] - ctx.Y[ctx.N])
    chord = max(ctx.X[1:ctx.N + 1]) - min(ctx.X[1:ctx.N + 1]) if ctx.N > 0 else 0.0
    rel_gap = te_gap / max(chord, 1.0e-12)
    ok = rel_gap < 5.0e-2
    return {"ok": ok, "te_gap": te_gap, "rel_te_gap": rel_gap}



def _element_edges(points: List[Point], marker: str) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        edges.append({"marker": marker, "p1": p1, "p2": p2})
    return edges



def build_global_boundary_connectivity(geom: MultiElementGeometry, farfield_padding: float = 0.5) -> Dict[str, object]:
    """Phase-A foundation: shared boundary representation + marker IDs."""
    all_pts = [pt for elem in geom.elements for pt in elem.points]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = xmax - xmin
    dy = ymax - ymin
    pad = farfield_padding * max(dx, dy, 1.0)

    farfield = [
        (xmin - pad, ymin - pad),
        (xmax + pad, ymin - pad),
        (xmax + pad, ymax + pad),
        (xmin - pad, ymax + pad),
    ]

    boundary_markers: Dict[str, int] = {"farfield": 0}
    wall_edges: List[Dict[str, object]] = []

    for elem in geom.elements:
        marker = elem.boundary_marker
        boundary_markers[marker] = elem.element_id
        wall_edges.extend(_element_edges(elem.points, marker))

    farfield_edges = _element_edges(farfield, "farfield")

    return {
        "boundary_markers": boundary_markers,
        "wall_edges": wall_edges,
        "farfield_edges": farfield_edges,
        "bbox": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "pad": pad},
    }
