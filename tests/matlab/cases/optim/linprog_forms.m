% linprog call forms: problem structure, optimset options, fewer arguments
r = @(v) round(v*1e8)/1e8 + 0;
p = struct('f', [-3; -5], 'Aineq', [1 0; 0 2; 3 2], 'bineq', [4; 12; 18], 'Aeq', [], ...
    'beq', [], 'lb', [0; 0], 'ub', [], 'solver', 'linprog', ...
    'options', optimoptions('linprog', 'Display', 'off'));
[x, f, e] = linprog(p);
xp = r(x); fp = r(f); ep = e;
x = linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0], [], optimset('Display', 'off'));
xs = r(x);
[x, f] = linprog([1; 2], [-1 -1; 1 -1], [-2; 1]);
x3 = r(x); f3 = r(f);
clear r p x f e
