function stop = pswplotbestf(optimValues, state)
%PSWPLOTBESTF  particleswarm plot function: best objective value at each iteration.
stop = optim_plotscalar(optimValues.iteration, optimValues.bestfval, state, 'Best Function Value: %g', 'Function value', 'b.');
end
