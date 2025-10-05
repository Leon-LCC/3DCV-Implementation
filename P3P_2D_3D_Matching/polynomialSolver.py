# Python reimplementation of the polynomial solver in OpenCV
# (Orignial Code URL) https://github.com/cdemel/OpenCV/blob/master/modules/calib3d/src/polynom_solver.cpp

from math import sqrt, acos, cos, pi


def solve_deg2(a, b, c):
    delta = b * b - 4 * a * c
    if delta < 0:
        return None, 0

    inv_2a = 0.5 / a

    if delta == 0:
        x1 = -b * inv_2a
        return [x1, x1], 2

    sqrt_delta = sqrt(delta)
    x1 = (-b + sqrt_delta) * inv_2a
    x2 = (-b - sqrt_delta) * inv_2a
    return [x1, x2], 2


# Reference : Eric W. Weisstein. "Cubic Equation." From MathWorld--A Wolfram Web Resource.
# http://mathworld.wolfram.com/CubicEquation.html
def solve_deg3(a, b, c, d):
    if a == 0: # Solve second order sytem
        if b == 0: # Solve first order system
            if c == 0:
                return None, 0

            x0 = -d / c
            return [x0, None, None], 1

        return  solve_deg2(b, c, d)
    
    # Calculate the normalized form x^3 + a2 * x^2 + a1 * x + a0 = 0
    inv_a = 1. / a
    b_a = inv_a * b
    b_a2 = b_a * b_a
    c_a = inv_a * c
    d_a = inv_a * d

    # Solve the cubic equation
    Q = (3 * c_a - b_a2) / 9
    R = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / 54
    Q3 = Q * Q * Q
    D = Q3 + R * R
    b_a_3 = (1. / 3.) * b_a

    if Q == 0:
        if R == 0:
            return [-b_a_3, -b_a_3, -b_a_3], 3
        else:
            x0 = pow(2 * R, 1 / 3.0) - b_a_3
            return [x0, None, None], 1

    # Three real roots
    if D <= 0: 
        theta = acos(R / sqrt(-Q3))
        sqrt_Q = sqrt(-Q)
        x0 = 2 * sqrt_Q * cos(theta             / 3.0) - b_a_3;
        x1 = 2 * sqrt_Q * cos((theta + 2 * pi)/ 3.0) - b_a_3;
        x2 = 2 * sqrt_Q * cos((theta + 4 * pi)/ 3.0) - b_a_3;

        return [x0, x1, x2], 3


    # D > 0, only one real root
    sign_R = 1 if R > 0 else -1 if R < 0 else 0
    AD = pow(abs(R) + sqrt(D), 1.0 / 3.0) * sign_R
    BD = 0 if AD == 0 else -Q / AD
    # Calculate the only real root
    x0 = AD + BD - b_a_3

    return [x0, None, None], 1

# Reference : Eric W. Weisstein. "Quartic Equation." From MathWorld--A Wolfram Web Resource.
# http://mathworld.wolfram.com/QuarticEquation.html
def solve_deg4(a, b, c, d, e):
    if a == 0:
        return solve_deg3(b, c, d, e) 

    # Normalize coefficients
    inv_a = 1. / a
    b *= inv_a
    c *= inv_a
    d *= inv_a
    e *= inv_a
    b2 = b * b
    bc = b * c
    b3 = b2 * b

    # Solve resultant cubic
    res, n_3 = solve_deg3(1, -c, d * b - 4 * e, 4 * c * e - d * d - b2 * e)
    if res is None:
        return None, 0

    # Calculate R^2
    R2 = 0.25 * b2 - c + res[0]
    if R2 < 0:
        return None, 0

    R = sqrt(R2)
    inv_R = 1. / R
    nb_real_roots = 0
    x0, x1, x2, x3 = None, None, None, None

    # Calculate D^2 and E^2
    if R < 10E-12:
        temp = res[0] * res[0] - 4 * e
        if temp < 0:
            D2 = E2 = -1
        else:
            sqrt_temp = sqrt(temp)
            D2 = 0.75 * b2 - 2 * c + 2 * sqrt_temp
            E2 = D2 - 4 * sqrt_temp
    else:
        u = 0.75 * b2 - 2 * c - R2
        v = 0.25 * inv_R * (4 * bc - 8 * d - b3)
        D2 = u + v
        E2 = u - v

    b_4 = 0.25 * b
    R_2 = 0.5 * R
    if D2 >= 0:
        D = sqrt(D2)
        nb_real_roots = 2
        D_2 = 0.5 * D
        x0 = R_2 + D_2 - b_4
        x1 = x0 - D


    # Calculate E^2
    if E2 >= 0:
        E = sqrt(E2)
        E_2 = 0.5 * E
        if nb_real_roots == 0:
            x0 = - R_2 + E_2 - b_4
            x1 = x0 - E
            nb_real_roots = 2
        else:
            x2 = - R_2 + E_2 - b_4
            x3 = x2 - E
            nb_real_roots = 4

    return [x0, x1, x2, x3], nb_real_roots