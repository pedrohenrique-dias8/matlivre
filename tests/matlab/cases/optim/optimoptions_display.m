% How an optimoptions object for linprog displays (default options and one option set)
o = optimoptions('linprog')
o2 = optimoptions('linprog', 'Display', 'off', 'MaxIterations', 50)
