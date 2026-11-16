% linprog on small LPs with unique optima: x, fval and exitflag (values rounded to 1e-8)
o = optimoptions('linprog', 'Display', 'off');
r = @(v) round(v*1e8)/1e8 + 0;
% inequalities, a free variable and a lower bound
[x, f, e] = linprog([-1; 4], [-3 1; 1 2], [6; 4], [], [], [-inf; -3], [], o);
x1 = r(x); f1 = r(f); e1 = e;
% classic two-variable LP
[x, f, e] = linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0], [], o);
x2 = r(x); f2 = r(f); e2 = e;
% equalities and a fixed variable
[x, f, e] = linprog([1; 1; 1], [], [], [1 1 1; 1 -1 0], [6; 1], [0; 0; 1], [10; 10; 1], o);
x3 = r(x); f3 = r(f); e3 = e;
% only bounds
[x, f, e] = linprog([1; -2; 0.5], [], [], [], [], [-1; -2; 3], [4; 5; 6], o);
x4 = r(x); f4 = r(f); e4 = e;
% row vector f and row-vector bounds
[x, f, e] = linprog([2 1], [-1 -1], -3, [], [], [0 0], [5 5], o);
x5 = r(x); f5 = r(f); e5 = e; sz5 = size(x);
% dense 30x20 LP: the optimum value only (x is not unique)
m = 30; n = 20;
Am = mod((1:m)' * (1:n) * 7, 11) / 11 - 0.4;
bm = Am * (mod((1:n)', 3) / 3) + 0.5;
fm = -(mod((1:n)', 5) + 1) / 5;
[x, f, e] = linprog(fm, Am, bm, [], [], zeros(n, 1), ones(n, 1), o);
f6 = round(f*1e6)/1e6 + 0; e6 = e; feasible6 = max(Am*x - bm) <= 1e-9;
clear o r x f e m n Am bm fm
