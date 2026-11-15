"""SciPy (HiGHS) references for test_milp.m."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
def run(name, c, integ, A=None, bu=None, Aeq=None, beq=None, lb=None, ub=None, show=None):
    cons = []
    if A is not None: cons.append(LinearConstraint(np.atleast_2d(A), -np.inf, bu))
    if Aeq is not None: cons.append(LinearConstraint(np.atleast_2d(Aeq), beq, beq))
    n = len(c); I = np.zeros(n); I[np.array(integ) - 1] = 1
    r = milp(c, constraints=cons, integrality=I, bounds=Bounds(lb if lb is not None else 0, ub if ub is not None else np.inf))
    x = r.x
    if x is not None and show == "idx": x = np.flatnonzero(np.round(x)) + 1
    print(f"SCIPY {name} status={r.status} f={r.fun} x={x}")
w = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82, 37, 52]); v = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72, 41, 55])
run("M1", -v, range(1, 13), [w], [165], lb=0, ub=1, show="idx")
run("M2", [-3, -2, -1], [3], [[1, 1, 1]], [7], [[4, 2, 1]], [12], lb=0, ub=[np.inf, np.inf, 1])
i = np.arange(1, 6)[:, None]; j = np.arange(1, 6)[None, :]
C = np.mod(i * j * 7 + i, 11) + 1
Aeq = np.vstack([np.kron(np.eye(5), np.ones((1, 5))), np.kron(np.ones((1, 5)), np.eye(5))])
run("M3", C.flatten(order="F"), range(1, 26), Aeq=Aeq, beq=np.ones(10), lb=0, ub=1, show="idx")
i = np.arange(1, 16)[:, None]; j = np.arange(1, 21)[None, :]
S = (np.mod(i * j + i ** 2, 4) == 0) | (np.eye(15, 20) == 1)
cost = np.mod(np.arange(1, 21), 5) + 1
run("M4", cost, range(1, 21), -S.astype(float), -np.ones(15), lb=0, ub=1, show="idx")
run("M5", [1, -2, 3, -1], [1, 2, 4], [[1, 1, 1, 1], [-1, 2, -1, 1], [3, -1, 0, 2]], [10, 4, 12], lb=[-5, -5, 0, -5], ub=[5, 5, 2.5, 5])
run("M6", [1, 1], [1, 2], Aeq=[[2, 2]], beq=[3], lb=0, ub=5)
run("M7", [-1, 0], [1], [[1, -1]], [1], lb=0)
run("M9", [-1, -1], [1, 2], [[2, 1], [1, 2]], [4.5, 4.5], lb=0)
w = np.mod(np.arange(1, 31) * 37, 61) + 20; v = w + 10; cap = np.floor(w.sum() / 2)
import time; t = time.time(); run("M10", -v, range(1, 31), [w], [cap], lb=0, ub=1, show="idx"); print("  time", time.time() - t)
i = np.arange(1, 11)[:, None]; j = np.arange(1, 16)[None, :]
Am = np.mod(i * j * 13 + i, 17) - 5; bm = 20 + np.mod(np.arange(1, 11) * 7, 23); cm = -(np.mod(np.arange(1, 16) * 11, 9) + 1)
t = time.time(); run("M11", cm, range(1, 16), Am, bm, lb=0, ub=10); print("  time", time.time() - t)
i = np.arange(1, 7)[:, None]; j = np.arange(1, 9)[None, :]
Am = np.mod(i * j * 5 + i, 11) - 3; bm = 10 + np.mod(np.arange(1, 7) * 7, 13); cm = -(np.mod(np.arange(1, 9) * 7, 5) + 1)
run("M12", cm, range(1, 9), Am, bm, lb=0, ub=6)
