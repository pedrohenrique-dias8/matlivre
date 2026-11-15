function stop = optimplotstepsize(x, optimValues, state)
%OPTIMPLOTSTEPSIZE  Plot function: norm of the step taken at each iteration.
v = []; if isfield(optimValues, 'stepsize'), v = optimValues.stepsize; end
stop = optim_plotscalar(optimValues.iteration, v, state, 'Step Size: %g', 'Step size');
end
