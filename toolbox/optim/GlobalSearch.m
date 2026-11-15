classdef GlobalSearch
    %GLOBALSEARCH  Find a global minimum by running fmincon from selected trial points.
    %   gs = GlobalSearch
    %   gs = GlobalSearch('Name', value, ...)
    %   [x, fval, exitflag, output, solutions] = run(gs, problem)
    %
    %   problem comes from createOptimProblem('fmincon', ...). Outline (after the public
    %   description of MATLAB's GlobalSearch): run fmincon from x0; generate NumTrialPoints random
    %   points in the bounds and score them (objective plus penalty for constraint violation);
    %   run fmincon from the best of the first NumStageOnePoints; then go through the remaining
    %   trial points and run fmincon only from points whose score is below a threshold and that
    %   lie outside the basins of the solutions found so far (spheres of radius
    %   DistanceThresholdFactor times the distance from a start point to its solution). The
    %   threshold rises after MaxWaitCycle points without a run, and basin radii shrink by
    %   BasinRadiusFactor after MaxWaitCycle consecutive points fall inside a basin.
    %   Trial points are uniform random here (MATLAB uses a scatter-search generator).
    %
    %   Properties: Display ('final' | 'iter' | 'off'), NumTrialPoints (1000),
    %   NumStageOnePoints (200), MaxWaitCycle (20), BasinRadiusFactor (0.2),
    %   DistanceThresholdFactor (0.75), PenaltyThresholdFactor (0.2), FunctionTolerance (1e-6),
    %   XTolerance (1e-6), MaxTime (Inf), StartPointsToRun ('all'), ArtificialBound (1000).
    %   exitflag: 1 at least one local solution, all runs converged; 2 some runs did not
    %   converge; -2 no feasible local solution; -5 MaxTime.
    %   Part of the MatLivre optimization toolbox draft (clean-room).
    properties
        Display = 'final'
        NumTrialPoints = 1000
        NumStageOnePoints = 200
        MaxWaitCycle = 20
        BasinRadiusFactor = 0.2
        DistanceThresholdFactor = 0.75
        PenaltyThresholdFactor = 0.2
        FunctionTolerance = 1e-6
        XTolerance = 1e-6
        MaxTime = Inf
        StartPointsToRun = 'all'
        ArtificialBound = 1000
    end
    methods
        function obj = GlobalSearch(varargin)
            for i = 1:2:numel(varargin)
                obj.(char(varargin{i})) = varargin{i + 1};
            end
        end
        function [x, fval, exitflag, output, solutions] = run(obj, problem)
            if ~strcmpi(problem.solver, 'fmincon')
                error('MatLivre:GlobalSearch:solver', 'GlobalSearch requires a problem for fmincon.');
            end
            t0 = tic;
            x0 = problem.x0; n = numel(x0);
            lb = problem.lb; ub = problem.ub;
            if isempty(lb), lb = -inf(1, n); end
            if isempty(ub), ub = inf(1, n); end
            lb = reshape(lb, 1, []); ub = reshape(ub, 1, []);
            lo = lb; hi = ub; xr = reshape(x0, 1, []);
            lo(~isfinite(lo)) = xr(~isfinite(lo)) - obj.ArtificialBound;
            hi(~isfinite(hi)) = xr(~isfinite(hi)) + obj.ArtificialBound;
            T = lo + rand(obj.NumTrialPoints, n) .* (hi - lo);
            sc = zeros(size(T, 1), 1);
            for i = 1:size(T, 1), sc(i) = GlobalSearch.gs_score(problem, T(i, :)); end
            nev = size(T, 1);
            sols = struct('X', {}, 'Fval', {}, 'Exitflag', {}, 'Output', {}, 'X0', {});
            basins = zeros(0, n + 1);            % centre, radius
            nrun = 0; nconv = 0;
            % stage 0: x0
            [sols, basins, nrun, nconv, nev] = GlobalSearch.gs_runfrom(problem, xr, sols, basins, nrun, nconv, nev, obj);
            % stage 1: best of the first NumStageOnePoints
            k1 = min(obj.NumStageOnePoints, size(T, 1));
            [s1, i1] = min(sc(1:k1));
            [sols, basins, nrun, nconv, nev] = GlobalSearch.gs_runfrom(problem, T(i1, :), sols, basins, nrun, nconv, nev, obj);
            thr = s1; waitlow = 0; waitbasin = zeros(size(basins, 1), 1);
            % stage 2
            for i = k1 + 1:size(T, 1)
                if toc(t0) > obj.MaxTime, break, end
                d = sqrt(sum((basins(:, 1:n) - T(i, :)).^2, 2));
                inb = d <= basins(:, n + 1);
                if any(inb)
                    waitbasin(inb) = waitbasin(inb) + 1;
                    hit = waitbasin >= obj.MaxWaitCycle;
                    basins(hit, n + 1) = basins(hit, n + 1) * (1 - obj.BasinRadiusFactor);
                    waitbasin(hit) = 0;
                    continue
                end
                if sc(i) < thr
                    [sols, basins, nrun, nconv, nev] = GlobalSearch.gs_runfrom(problem, T(i, :), sols, basins, nrun, nconv, nev, obj);
                    waitbasin(end + 1:size(basins, 1), 1) = 0;
                    thr = sc(i); waitlow = 0;
                else
                    waitlow = waitlow + 1;
                    if waitlow >= obj.MaxWaitCycle
                        thr = thr + obj.PenaltyThresholdFactor * (1 + abs(thr));
                        waitlow = 0;
                    end
                end
            end
            if isempty(sols)
                x = []; fval = []; exitflag = -2;
                msg = 'GlobalSearch: no local solver run converged to a feasible local solution.';
            else
                [~, ord] = sort([sols.Fval]); sols = sols(ord);
                x = sols(1).X; fval = sols(1).Fval;
                if nconv == nrun, exitflag = 1; else, exitflag = 2; end
                if toc(t0) > obj.MaxTime, exitflag = -5; end
                msg = sprintf('GlobalSearch stopped: %d local solver runs, %d converged with a positive local exit flag.', nrun, nconv);
            end
            output = struct('funcCount', nev, 'localSolverTotal', nrun, 'localSolverSuccess', nconv, ...
                'localSolverIncomplete', nrun - nconv, 'localSolverNoSolution', nrun - nconv, 'message', msg);
            solutions = sols;
            if strcmpi(obj.Display, 'final') || strcmpi(obj.Display, 'iter')
                fprintf('\n%s\n\n', msg);
            end
        end
    end
    methods (Static, Access = private)
    % helpers (static methods: MatLivre does not yet resolve local functions of a classdef file)
        function s = gs_score(p, x)
        X = reshape(x, size(p.x0));
        s = p.objective(X);
        v = 0;
        if ~isempty(p.Aineq), v = v + sum(max(0, p.Aineq * x(:) - p.bineq(:))); end
        if ~isempty(p.Aeq), v = v + sum(abs(p.Aeq * x(:) - p.beq(:))); end
        if ~isempty(p.nonlcon)
            [c, ceq] = p.nonlcon(X); v = v + sum(max(0, c(:))) + sum(abs(ceq(:)));
        end
        s = s + 1000 * v;
        end

        function [sols, basins, nrun, nconv, nev] = gs_runfrom(problem, xs, sols, basins, nrun, nconv, nev, obj)
        p = problem; p.x0 = reshape(xs, size(problem.x0));
        [x, f, ef, out] = fmincon(p);
        nrun = nrun + 1; nev = nev + out.funcCount;
        if strcmpi(obj.Display, 'iter')
            fprintf(' run %3d  exitflag %2d  f = %.6g\n', nrun, ef, f);
        end
        r = obj.DistanceThresholdFactor * norm(x(:).' - xs);
        basins(end + 1, :) = [x(:).', max(r, 0)];
        if ef > 0
            nconv = nconv + 1;
            for j = 1:numel(sols)
                if norm(x(:) - sols(j).X(:)) <= obj.XTolerance * max(1, norm(x(:))) && abs(f - sols(j).Fval) <= obj.FunctionTolerance * max(1, abs(f))
                    sols(j).X0{end + 1} = p.x0;
                    return
                end
            end
            sols(end + 1) = struct('X', x, 'Fval', f, 'Exitflag', ef, 'Output', out, 'X0', {{p.x0}});
        end
        end
    end
end
