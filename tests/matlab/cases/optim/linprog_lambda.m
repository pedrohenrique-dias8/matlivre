% linprog Lagrange multipliers (MATLAB's signs) and output fields on nondegenerate LPs
o = optimoptions('linprog', 'Display', 'off');
r = @(v) round(v*1e8)/1e8 + 0;
[x, f, e, out, lam] = linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0], [], o);
ineqlin = r(lam.ineqlin); lower2 = r(lam.lower); upper2 = r(lam.upper); eqlin2 = lam.eqlin;
viol_small = out.constrviolation <= 1e-8; fo_small = out.firstorderopt <= 1e-8;
% an equality and an active lower bound: eqlin = 2, lower = [1; 0]
[x, f, e, out, lam] = linprog([-1; -2], [1 0], 3, [1 1], 4, [0; 0], [], o);
eqlin = r(lam.eqlin); lower = r(lam.lower); upper = r(lam.upper); ineq3 = r(lam.ineqlin);
% an active upper bound
[x, f, e, out, lam] = linprog([-1; 1], [], [], [], [], [0; 0], [2; 3], o);
upper4 = r(lam.upper); lower4 = r(lam.lower);
lam_fields = fieldnames(lam)';
out_fields = fieldnames(out)';
clear o r x f e out lam
