import math


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
