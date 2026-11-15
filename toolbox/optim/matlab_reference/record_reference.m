function record_reference()
%RECORD_REFERENCE  Run the toolbox test scripts with MATLAB's own solvers and save the output.
%
% How to use (in MATLAB R2023b, with the Optimization Toolbox and, for test_global, the Global
% Optimization Toolbox):
%   1. Make sure the toolbox folder (the parent of this one, with fmincon.m, ga.m, ...) is NOT
%      on the MATLAB path: otherwise MATLAB would run our files instead of its own. The script
%      checks this before starting. It adds only ../tests (the test scripts) to the path.
%   2. cd to this folder and run:   record_reference
%   3. Send back the file  reference_matlab.txt  (and the out_*.csv files it creates).
%
% Every test prints RESULT/LAMBDA/HS/GLOBAL lines in the same format as in MatLivre, so the two
% outputs can be compared line by line. Stochastic tests (test_global) report success rates over
% 20 seeds; MATLAB's and MatLivre's random streams differ, so only the rates are comparable.

here = fileparts(mfilename('fullpath'));
cd(here);
solvers = {'fmincon', 'fminunc', 'quadprog', 'linprog', 'lsqlin', 'lsqnonlin', 'lsqcurvefit', ...
    'fsolve', 'ga', 'particleswarm', 'patternsearch', 'MultiStart', 'GlobalSearch', 'createOptimProblem', ...
    'checkGradients', 'intlinprog', 'optimplotfval', 'optimplotx', 'gaplotbestf', 'pswplotbestf', 'psplotbestf'};
bad = {};
for i = 1:numel(solvers)
    w = which(solvers{i});
    if ~isempty(w) && ~contains(w, matlabroot)
        bad{end + 1} = sprintf('%s -> %s', solvers{i}, w); %#ok<SAGROW>
    end
end
if ~isempty(bad)
    error('These names resolve to files outside MATLAB (remove them from the path first):\n  %s', strjoin(bad, '\n  '));
end

addpath(fullfile(here, '..', 'tests'));   % the test scripts only, never the toolbox folder
out = fullfile(here, 'reference_matlab.txt');
if exist(out, 'file'), delete(out); end
diary(out);
cleanup = onCleanup(@() diary('off'));
fprintf('MATLAB %s   %s\n', version, char(datetime('now')));
v = ver; fprintf('Toolboxes: %s\n', strjoin({v.Name}, '; '));
tests = {'test_fmincon', 'test_hs', 'test_qp', 'test_fminunc', 'test_nls', 'test_global', 'test_checkgradients', 'test_outputfcn', 'test_milp', 'test_simplex'};
needs = {'optim', 'optim', 'optim', 'optim', 'optim', 'globaloptim', 'optim', 'globaloptim', 'optim', 'optim'};
for i = 1:numel(tests)
    fprintf('\n=================== %s ===================\n', tests{i});
    if ~license('test', strrep(strrep(needs{i}, 'globaloptim', 'GADS_Toolbox'), 'optim', 'Optimization_Toolbox'))
        fprintf('SKIPPED %s: toolbox not licensed\n', tests{i});
        continue
    end
    try
        t = tic;
        rr_run(fullfile(here, [tests{i} '.m']));
        fprintf('ELAPSED %s %.3f s\n', tests{i}, toc(t));
    catch e
        fprintf('FAILED %s: %s\n', tests{i}, e.message);
    end
end
diary('off');
fprintf('\nDone. Output written to %s\n', out);
end

function rr_run(file)
% run a test script in its own workspace, so its variables cannot overwrite the runner's
run(file);
end
