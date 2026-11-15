function stop = psplotmeshsize(optimvalues, flag)
%PSPLOTMESHSIZE  patternsearch plot function: mesh size at each iteration.
stop = optim_plotscalar(optimvalues.iteration, optimvalues.meshsize, flag, 'Current Mesh Size: %g', 'Mesh size', 'kd');
end
