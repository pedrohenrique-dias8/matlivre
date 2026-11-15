import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
def run(name, f, x0, cons, bounds=None):
    r = minimize(f, x0, method="SLSQP", constraints=cons, bounds=bounds, options=dict(ftol=1e-12, maxiter=500))
    print(f"SCIPY {name} f={r.fun:.10g} x={np.array2string(r.x, precision=9)} nit={r.nit} nfev={r.nfev} ok={r.success}")
rosen=lambda x:100*(x[1]-x[0]**2)**2+(1-x[0])**2
run("P1", rosen, [0,0], [dict(type="ineq", fun=lambda x:1-x[0]**2-x[1]**2)])
hs=lambda x:x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]
run("P2", hs, [1,5,5,1], [dict(type="ineq", fun=lambda x:np.prod(x)-25), dict(type="eq", fun=lambda x:np.sum(x**2)-40)], [(1,5)]*4)
q=lambda x:(x[0]-1)**2+(x[1]-2.5)**2
A=np.array([[-1,2],[1,2],[1,-2]]); b=np.array([2,6,2])
run("P4", q, [2,0], [dict(type="ineq", fun=lambda x:b-A@x)], [(0,None),(0,None)])
run("P5", rosen, [-1.2,1], [], [(-2,0.8),(-2,2)])
run("P6", lambda x:x[0]+x[1], [1,0.5], [dict(type="eq", fun=lambda x:x[0]**2+x[1]**2-2)])
f7=lambda x:np.exp(x[0])*(4*x[0]**2+2*x[1]**2+4*x[0]*x[1]+2*x[1]+1)
run("P7", f7, [-1,1], [dict(type="eq", fun=lambda x:x[0]+x[1]), dict(type="ineq", fun=lambda x:-np.array([1.5+x[0]*x[1]-x[0]-x[1], -x[0]*x[1]-10]))])
T=np.array([1,3,2,4.])  # column-major [1 2;3 4]
run("P8", lambda X:np.sum((X-T)**2), np.zeros(4), [dict(type="ineq", fun=lambda X:6-X.sum()), dict(type="ineq", fun=lambda X:4-X[0]**2-X[3]**2)])
