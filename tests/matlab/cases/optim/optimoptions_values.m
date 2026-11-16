% optimoptions for linprog: class, values set and copied, and the solver name
o = optimoptions('linprog', 'Display', 'off', 'MaxIterations', 50);
cls = class(o); d = o.Display; mi = o.MaxIterations;
o2 = optimoptions(o, 'Display', 'iter');
d2 = o2.Display; mi2 = o2.MaxIterations;
o3 = optimoptions(@linprog, 'Display', 'final');
d3 = o3.Display;
try, optimoptions('linprog', 'Display'); catch ME, id1 = ME.identifier; msg1 = ME.message; end
try, optimoptions('linprog', 'NotAnOption', 1); catch ME, id2 = ME.identifier; msg2 = ME.message; end
clear o o2 o3 ME
