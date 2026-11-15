function state = gaplotbestf(options, state, flag)
%GAPLOTBESTF  ga plot function: best and mean fitness at each generation.
if strcmp(flag, 'done') || strcmp(flag, 'interrupt'), return, end
ax = gca;
sc = state.Score(isfinite(state.Score));
best = min(state.Score); mn = mean(sc);
if strcmp(flag, 'init')
    cla(ax); hold(ax, 'on');
    xlabel(ax, 'Generation'); ylabel(ax, 'Fitness value');
end
plot(ax, state.Generation, best, 'k.');
plot(ax, state.Generation, mn, 'b.');
title(ax, sprintf('Best: %g Mean: %g', best, mn));
end
