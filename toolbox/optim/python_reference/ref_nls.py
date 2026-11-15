"""SciPy references for test_nls.m and test_fminunc.m (least_squares, fsolve, BFGS)."""
import numpy as np
from scipy.optimize import least_squares, fsolve, minimize, rosen
r = least_squares(lambda x: [10 * (x[1] - x[0] ** 2), 1 - x[0]], [-1.2, 1], bounds=([-np.inf, -np.inf], [0.5, np.inf]), xtol=1e-15, ftol=1e-15, gtol=1e-15); print("SCIPY N2", 2 * r.cost, r.x)
xd = np.linspace(0, 4, 25); yn = 3 / (1 + ((xd - 2) / 0.5) ** 2) + 0.05 * np.sin(37 * xd)
r = least_squares(lambda p: p[0] / (1 + ((xd - p[1]) / p[2]) ** 2) - yn, [1, 1.5, 1], bounds=([0, 0, 0.01], [10, 4, 5]), xtol=1e-15, ftol=1e-15, gtol=1e-15); print("SCIPY N4", 2 * r.cost, r.x)
i = np.arange(1, 21) / 5
bd = lambda x: (x[0] + x[1] * i - np.exp(i)) ** 2 + (x[2] + x[3] * np.sin(i) - np.cos(i)) ** 2
r = least_squares(bd, [25, 5, -5, -1], xtol=1e-15, ftol=1e-15, gtol=1e-15, method="lm"); print("SCIPY N5", 2 * r.cost, r.x, "nfev", r.nfev)
print("SCIPY S1", fsolve(lambda x: [2 * x[0] - x[1] - np.exp(-x[0]), -x[0] + 2 * x[1] - np.exp(-x[1])], [-5, -5], xtol=1e-14))
print("SCIPY S2", fsolve(lambda x: [1e4 * x[0] * x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - 1.0001], [0, 1], xtol=1e-14))
def br(x):
    xm = np.r_[0, x[:-1]]; xp = np.r_[x[1:], 0]; return (3 - 2 * x) * x - xm - 2 * xp + 1
print("SCIPY S5", fsolve(br, -np.ones(10), xtol=1e-14)[:4])
for n, x0 in (("U1", [-1.2, 1]), ("U2", -np.ones(10))):
    r = minimize(rosen, x0, method="BFGS", options=dict(gtol=1e-6)); print("SCIPY", n, r.fun, "nit", r.nit, "nfev", r.nfev, "err", abs(r.x - 1).max())
