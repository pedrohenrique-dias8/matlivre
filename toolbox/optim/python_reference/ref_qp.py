import numpy as np
from scipy.optimize import linprog, minimize, lsq_linear

def p(name, x, f, extra=""):
    print(f"SCIPY {name} f={f:.10g} x={np.array2string(np.asarray(x), precision=9, max_line_width=200)} {extra}")

r = linprog([-1, 4], A_ub=[[-3, 1], [1, 2]], b_ub=[6, 4], bounds=[(None, None), (-3, None)], method="highs"); p("LP1", r.x, r.fun)
r = linprog([-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], method="highs"); p("LP2", r.x, r.fun, f"duals={-r.ineqlin.marginals}")
r = linprog([1, 1, 1], A_eq=[[1, 1, 1], [1, -1, 0]], b_eq=[6, 1], bounds=[(0, 10), (0, 10), (1, 1)], method="highs"); p("LP3", r.x, r.fun)
r = linprog([1, 1], A_ub=[[1, 1], [-1, -1]], b_ub=[1, -3], method="highs"); print("SCIPY LP4 status", r.status, "(2 = infeasible)")
r = linprog([-1, 0], A_ub=[[1, -1]], b_ub=[1], method="highs"); print("SCIPY LP5 status", r.status, "(3 = unbounded)")
m, n = 30, 20
I = np.arange(1, m + 1)[:, None]; J = np.arange(1, n + 1)[None, :]
Am = np.mod(I * J * 7, 11) / 11 - 0.4
x_ok = np.mod(np.arange(1, n + 1), 3) / 3
bm = Am @ x_ok + 0.5
fm = -(np.mod(np.arange(1, n + 1), 5) + 1) / 5
r = linprog(fm, A_ub=Am, b_ub=bm, bounds=[(0, 1)] * n, method="highs"); p("LP6", r.x[:5], r.fun)
xm = np.loadtxt("out_lp6.csv", delimiter=",")
print(f"CHECK LP6 matlivre f={fm @ xm:.10g} maxviol={max(0, (Am @ xm - bm).max()):.2e}  |f-f*|={abs(fm @ xm - r.fun):.2e}")

H = np.array([[1, -1], [-1, 2.]]); f = np.array([-2, -6.]); A = np.array([[1, 1], [-1, 2], [2, 1.]]); b = np.array([2, 2, 3.])
r = minimize(lambda x: 0.5 * x @ H @ x + f @ x, [0, 0], jac=lambda x: H @ x + f, method="SLSQP",
             constraints=[dict(type="ineq", fun=lambda x: b - A @ x, jac=lambda x: -A)], bounds=[(0, None)] * 2, options=dict(ftol=1e-14))
p("QP1", r.x, r.fun)
n = 10
I = np.arange(1, n + 1)[:, None]; J = np.arange(1, n + 1)[None, :]
M = np.mod(I * J * 3, 7) / 7 - 0.3
Hq = M.T @ M + 0.1 * np.eye(n)
fq = np.mod(np.arange(1, n + 1), 4) - 1.5
aeq = np.arange(1, n + 1) / n
r = minimize(lambda x: 0.5 * x @ Hq @ x + fq @ x, np.zeros(n), jac=lambda x: Hq @ x + fq, method="SLSQP",
             constraints=[dict(type="ineq", fun=lambda x: 2 - x.sum()), dict(type="eq", fun=lambda x: aeq @ x - 0.5)],
             bounds=[(-1, 1)] * n, options=dict(ftol=1e-15, maxiter=1000))
p("QP2", r.x[:5], r.fun)
xq = np.loadtxt("out_qp2.csv", delimiter=",")
print(f"CHECK QP2 max|x-x*|={abs(xq - r.x).max():.2e}")
r = minimize(lambda x: 0.5 * x @ x - np.array([1, 2, 3.]) @ x, np.zeros(3), method="SLSQP",
             constraints=[dict(type="ineq", fun=lambda x: 3 - x.sum())], options=dict(ftol=1e-15))
p("QP4", r.x, r.fun)
I = np.arange(1, 9)[:, None]; J = np.arange(1, 5)[None, :]
C = np.mod(I * J * 5, 9) / 9; d = np.arange(1, 9) / 4
r = lsq_linear(C, d, bounds=(0, 0.5), tol=1e-14); p("LSQLIN1", r.x, 2 * r.cost)
r = minimize(lambda x: np.sum((C @ x - d) ** 2), np.zeros(4), jac=lambda x: 2 * C.T @ (C @ x - d), method="SLSQP",
             constraints=[dict(type="ineq", fun=lambda x: 1 - x.sum()), dict(type="eq", fun=lambda x: x[0] - x[1])], options=dict(ftol=1e-15))
p("LSQLIN2", r.x, r.fun)
