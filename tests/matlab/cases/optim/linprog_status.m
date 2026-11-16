% linprog exit flags and empty outputs: infeasible, unbounded, inconsistent bounds,
% iteration limit
o = optimoptions('linprog', 'Display', 'off');
[x, f, e] = linprog([1; 1], [1 1; -1 -1], [1; -3], [], [], [], [], o);
e_infeasible = e; xempty_infeasible = isempty(x); fempty_infeasible = isempty(f);
[x, f, e] = linprog([-1; 0], [1 -1], 1, [], [], [0; 0], [], o);
e_unbounded = e; xempty_unbounded = isempty(x); fempty_unbounded = isempty(f);
[x, f, e] = linprog([1; 1], [], [], [], [], [0; 2], [1; 1], o);
e_bounds = e; xempty_bounds = isempty(x);
[x, f, e] = linprog([1; 1], [], [], [1 1], 1, [0; 0], [0.2; 0.3], o);
e_eq_infeasible = e;
[x, f, e, out] = linprog([-1; -1], [1 1], 4, [], [], [0; 0], [], optimoptions(o, 'MaxIterations', 0));
e_maxiter = e; it_maxiter = out.iterations;
clear o x f e out
