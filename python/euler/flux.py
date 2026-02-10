from .transforms import conservative_to_primitive


def hll_flux(q_l, q_r, nx, ny, gamma=1.4):
    rl, ul, vl, pl = conservative_to_primitive(*q_l, gamma=gamma)
    rr, ur, vr, pr = conservative_to_primitive(*q_r, gamma=gamma)

    unl = ul * nx + vl * ny
    unr = ur * nx + vr * ny

    al = (gamma * pl / rl) ** 0.5
    ar = (gamma * pr / rr) ** 0.5

    sl = min(unl - al, unr - ar)
    sr = max(unl + al, unr + ar)

    fl = (
        q_l[1] * nx + q_l[2] * ny,
        (q_l[1] * ul + pl) * nx + (q_l[1] * vl) * ny,
        (q_l[2] * ul) * nx + (q_l[2] * vl + pl) * ny,
        (q_l[3] + pl) * unl,
    )
    fr = (
        q_r[1] * nx + q_r[2] * ny,
        (q_r[1] * ur + pr) * nx + (q_r[1] * vr) * ny,
        (q_r[2] * ur) * nx + (q_r[2] * vr + pr) * ny,
        (q_r[3] + pr) * unr,
    )

    if sl >= 0.0:
        return fl
    if sr <= 0.0:
        return fr

    return tuple((sr * fl[i] - sl * fr[i] + sl * sr * (q_r[i] - q_l[i])) / (sr - sl) for i in range(4))
