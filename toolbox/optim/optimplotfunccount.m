function stop = optimplotfunccount(x, optimValues, state)
%OPTIMPLOTFUNCCOUNT  Plot function: number of function evaluations at each iteration.
stop = optim_plotscalar(optimValues.iteration, optimValues.funccount, state, 'Total Function Evaluations: %d', 'Evaluations');
end
