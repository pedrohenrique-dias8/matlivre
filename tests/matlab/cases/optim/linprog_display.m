% linprog's default display ('final') and Display = 'off', 'notify' on a solved and an
% infeasible problem
x = linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0]);
x = linprog([1; 1], [1 1; -1 -1], [1; -3]);
x = linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0], [], optimoptions('linprog', 'Display', 'notify'));
x = linprog([1; 1], [1 1; -1 -1], [1; -3], [], [], [], [], optimoptions('linprog', 'Display', 'notify'));
x = linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0], [], optimoptions('linprog', 'Display', 'off'));
clear x
