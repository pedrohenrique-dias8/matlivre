function stop = psplotbestf(optimvalues, flag)
%PSPLOTBESTF  patternsearch plot function: best objective value at each iteration.
stop = optim_plotscalar(optimvalues.iteration, optimvalues.fval, flag, 'Best Function Value: %g', 'Function value', 'kd');
end
