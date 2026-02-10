def farfield_state(ctx):
    return {
        "rho": 1.0,
        "u": ctx.QINF,
        "v": 0.0,
        "p": 1.0,
        "mach": ctx.MINF,
    }


def wall_normal_velocity(u, v, nx, ny):
    return u * nx + v * ny
