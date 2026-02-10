import math


def primitive_to_conservative(rho, u, v, p, gamma=1.4):
    e = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return rho, rho * u, rho * v, e


def conservative_to_primitive(rho, rhou, rhov, e, gamma=1.4):
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    u = rhou / rho
    v = rhov / rho
    kinetic = 0.5 * rho * (u * u + v * v)
    p = (gamma - 1.0) * max(e - kinetic, 1.0e-12)
    return rho, u, v, p


def mach_from_state(rho, u, v, p, gamma=1.4):
    a = math.sqrt(gamma * p / rho)
    return math.sqrt(u * u + v * v) / a
