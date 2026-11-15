if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Tests for OutputFcn and PlotFcn. Prints OUTFCN lines.
global TOF_LOG
rosen = @(x) 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;
disk = @(x) deal(x(1)^2 + x(2)^2 - 1, []);
% 1. history recording in fmincon
TOF_LOG = [];
o = optimoptions('fmincon', 'Display', 'off', 'OutputFcn', @tof_record);
[x, f, ef, out] = fmincon(rosen, [0 0], [], [], [], [], [], [], disk, o);
fprintf('OUTFCN fmincon_record ef=%d iterations=%d calls=%d states=%s last_fval_matches=%d\n', ef, out.iterations, ...
    size(TOF_LOG, 1), mat2str(unique(TOF_LOG(:, 1)).'), abs(TOF_LOG(end, 3) - f) < 1e-12);
% 2. stop after 3 iterations -> exitflag -1, for every local solver
stop3 = @(x, ov, st) ov.iteration >= 3;
[~, ~, ef, out] = fmincon(rosen, [0 0], [], [], [], [], [], [], disk, optimoptions('fmincon', 'Display', 'off', 'OutputFcn', stop3));
fprintf('OUTFCN fmincon_stop ef=%d it=%d\n', ef, out.iterations);
[~, ~, ef, out] = fminunc(rosen, [-1.2 1], optimoptions('fminunc', 'Display', 'off', 'OutputFcn', stop3));
fprintf('OUTFCN fminunc_stop ef=%d it=%d\n', ef, out.iterations);
[~, ~, ~, ef, out] = lsqnonlin(@(x) [10 * (x(2) - x(1)^2); 1 - x(1)], [-1.2 1], [], [], optimoptions('lsqnonlin', 'Display', 'off', 'OutputFcn', stop3));
fprintf('OUTFCN lsqnonlin_stop ef=%d it=%d\n', ef, out.iterations);
[~, ~, ef, out] = fsolve(@(x) [2*x(1) - x(2) - exp(-x(1)); -x(1) + 2*x(2) - exp(-x(2))], [-5 -5], optimoptions('fsolve', 'Display', 'off', 'OutputFcn', stop3));
fprintf('OUTFCN fsolve_stop ef=%d it=%d\n', ef, out.iterations);
% 3. plot functions (run without error, figure created, result unchanged)
nf0 = numel(findobj('Type', 'figure'));
o = optimoptions('fmincon', 'Display', 'off', 'PlotFcn', {@optimplotfval, @optimplotx, @optimplotfunccount, ...
    @optimplotconstrviolation, @optimplotstepsize, @optimplotfirstorderopt});
[x2, f2, ef2] = fmincon(rosen, [0 0], [], [], [], [], [], [], disk, o);
fprintf('OUTFCN fmincon_plots ef=%d same_result=%d new_figures=%d\n', ef2, abs(f2 - f) < 1e-12, numel(findobj('Type', 'figure')) - nf0);
[~, ~, ~, ef] = lsqcurvefit(@(p, t) p(1) * exp(-p(2) * t), [1 1], 0:0.5:3, 2 * exp(-0.7 * (0:0.5:3)), [], [], ...
    optimoptions('lsqcurvefit', 'Display', 'off', 'PlotFcn', {'optimplotresnorm', 'optimplotfval'}));
fprintf('OUTFCN lsqcurvefit_plots ef=%d\n', ef);
% 4. global solvers
ras = @(x) 10 * numel(x) + sum(x.^2 - 10 * cos(2 * pi * x));
rng(1); [~, ~, ef, out] = particleswarm(ras, 2, [-5 -5], [5 5], optimoptions('particleswarm', 'Display', 'off', ...
    'OutputFcn', @(ov, st) ov.iteration >= 5, 'PlotFcn', @pswplotbestf));
fprintf('OUTFCN pso_stop ef=%d it=%d\n', ef, out.iterations);
rng(1); [~, ~, ef, out] = ga(ras, 2, [], [], [], [], [-5 -5], [5 5], [], optimoptions('ga', 'Display', 'off', ...
    'OutputFcn', @tof_ga_stop, 'PlotFcn', @gaplotbestf));
fprintf('OUTFCN ga_stop ef=%d generations=%d\n', ef, out.generations);
[~, ~, ef, out] = patternsearch(rosen, [-1.2 1], [], [], [], [], [], [], [], optimoptions('patternsearch', 'Display', 'off', ...
    'OutputFcn', @tof_ps_stop, 'PlotFcn', {@psplotbestf, @psplotmeshsize}));
fprintf('OUTFCN ps_stop ef=%d it=%d\n', ef, out.iterations);

function stop = tof_record(x, ov, state)
global TOF_LOG
codes = struct('init', 1, 'iter', 2, 'done', 3);
TOF_LOG(end + 1, :) = [codes.(state), ov.iteration, ov.fval];
stop = false;
end
function [state, options, changed] = tof_ga_stop(options, state, flag)
changed = false;
if state.Generation >= 4, state.StopFlag = 'y'; end
end
function [stop, options, changed] = tof_ps_stop(ov, options, flag)
changed = false; stop = ov.iteration >= 10;
end
