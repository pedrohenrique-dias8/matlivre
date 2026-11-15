classdef MultiStart
    %MULTISTART  Find multiple local minima by running a local solver from many start points.
    %   ms = MultiStart
    %   ms = MultiStart('Name', value, ...)
    %   [x, fval, exitflag, output, solutions] = run(ms, problem, k)
    %   [x, fval, exitflag, output, solutions] = run(ms, problem, startpoints)
    %
    %   problem comes from createOptimProblem. k random start points are drawn uniformly in the
    %   bounds (problem.x0 is always the first one; where a bound is infinite, ArtificialBound
    %   around x0 is used). startpoints may instead be a matrix with one start point per row.
    %   Local solutions closer than XTolerance (relative) with values within FunctionTolerance
    %   are merged; solutions are returned sorted by fval.
    %
    %   Properties: Display ('final' | 'iter' | 'off'), FunctionTolerance (1e-6),
    %   XTolerance (1e-6), MaxTime (Inf), StartPointsToRun ('all' | 'bounds' | 'bounds-ineqs'),
    %   ArtificialBound (1000), UseParallel (false; runs serially).
    %   exitflag: 1 at least one local solution, all runs converged; 2 at least one local solution,
    %   some runs did not converge; 0 no run converged; -2 no feasible local solution; -5 MaxTime.
    %   Part of the MatLivre optimization toolbox draft (clean-room).
    properties
        Display = 'final'
        FunctionTolerance = 1e-6
        XTolerance = 1e-6
        MaxTime = Inf
        StartPointsToRun = 'all'
        ArtificialBound = 1000
        UseParallel = false
    end
    methods
        function obj = MultiStart(varargin)
            if nargin == 1 && isa(varargin{1}, 'GlobalSearch')
                g = varargin{1};
                obj.Display = g.Display; obj.FunctionTolerance = g.FunctionTolerance;
                obj.XTolerance = g.XTolerance; obj.MaxTime = g.MaxTime;
                return
            end
            for i = 1:2:numel(varargin)
                obj.(char(varargin{i})) = varargin{i + 1};
            end
        end
        function [x, fval, exitflag, output, solutions] = run(obj, problem, k)
            x0 = problem.x0;
            n = numel(x0);
            if isscalar(k)
                X = MultiStart.ms_startpoints(problem, k, obj.ArtificialBound);
            else
                X = [reshape(x0, 1, []); k];
            end
            X = MultiStart.ms_filter(X, problem, obj.StartPointsToRun);
            t0 = tic;
            sols = struct('X', {}, 'Fval', {}, 'Exitflag', {}, 'Output', {}, 'X0', {});
            nconv = 0; nrun = 0; feval_total = 0;
            if strcmpi(obj.Display, 'iter')
                fprintf('\n Run       Local       Local      Local    Local   First-order\n');
                fprintf('Index     exitflag      f(x)     # iter   F-count   optimality\n');
            end
            for i = 1:size(X, 1)
                if toc(t0) > obj.MaxTime, break, end
                p = problem;
                p.x0 = reshape(X(i, :), size(x0));
                [xi, fi, ef, out] = MultiStart.ms_local(p);
                nrun = nrun + 1;
                if isfield(out, 'funcCount'), feval_total = feval_total + out.funcCount; end
                if strcmpi(obj.Display, 'iter')
                    fo = NaN; if isfield(out, 'firstorderopt'), fo = out.firstorderopt; end
                    fprintf('%4d %10d %12.4g %9d %9d %12.4g\n', i, ef, fi, out.iterations, out.funcCount, fo);
                end
                if ef > 0
                    nconv = nconv + 1;
                    sols = MultiStart.ms_merge(sols, xi, fi, ef, out, p.x0, obj.XTolerance, obj.FunctionTolerance);
                end
            end
            if isempty(sols)
                x = []; fval = [];
                if nrun > 0, exitflag = -2; else, exitflag = -5; end
                msg = 'MultiStart: no local solver run converged to a feasible local solution.';
            else
                [~, ord] = sort([sols.Fval]);
                sols = sols(ord);
                x = sols(1).X; fval = sols(1).Fval;
                if nconv == nrun, exitflag = 1; else, exitflag = 2; end
                if toc(t0) > obj.MaxTime && nrun < size(X, 1), exitflag = -5; end
                msg = sprintf('MultiStart completed %d of %d runs from %d start points; %d runs converged with a positive local exit flag.', ...
                    nrun, size(X, 1), size(X, 1), nconv);
            end
            output = struct('funcCount', feval_total, 'localSolverTotal', nrun, ...
                'localSolverSuccess', nconv, 'localSolverIncomplete', nrun - nconv, ...
                'localSolverNoSolution', nrun - nconv, 'message', msg);
            solutions = sols;
            if strcmpi(obj.Display, 'final') || strcmpi(obj.Display, 'iter')
                fprintf('\n%s\n\n', msg);
            end
        end
    end
    methods (Static, Access = private)
    % helpers (static methods: MatLivre does not yet resolve local functions of a classdef file)
        function X = ms_startpoints(problem, k, abound)
        x0 = reshape(problem.x0, 1, []); n = numel(x0);
        lb = problem.lb; ub = problem.ub;
        if isempty(lb), lb = -inf(1, n); end
        if isempty(ub), ub = inf(1, n); end
        lb = reshape(lb, 1, []); ub = reshape(ub, 1, []);
        lo = lb; hi = ub;
        lo(~isfinite(lo)) = x0(~isfinite(lo)) - abound;
        hi(~isfinite(hi)) = x0(~isfinite(hi)) + abound;
        X = [x0; lo + rand(max(k - 1, 0), n) .* (hi - lo)];
        end

        function X = ms_filter(X, problem, mode)
        if strcmpi(mode, 'all'), return, end
        n = size(X, 2);
        keep = true(size(X, 1), 1);
        lb = problem.lb; ub = problem.ub;
        if ~isempty(lb), keep = keep & all(X >= reshape(lb, 1, []), 2); end
        if ~isempty(ub), keep = keep & all(X <= reshape(ub, 1, []), 2); end
        if strcmpi(mode, 'bounds-ineqs') && ~isempty(problem.Aineq)
            keep = keep & all(X * problem.Aineq.' <= reshape(problem.bineq, 1, []) + 1e-12, 2);
        end
        X = X(keep, :);
        if isempty(X), X = zeros(0, n); end
        end

        function [x, f, ef, out] = ms_local(p)
        switch p.solver
            case 'fmincon'
                [x, f, ef, out] = fmincon(p);
            case 'fminunc'
                [x, f, ef, out] = fminunc(p);
            case 'lsqnonlin'
                [x, f, ~, ef, out] = lsqnonlin(p);
            case 'lsqcurvefit'
                [x, f, ~, ef, out] = lsqcurvefit(p);
        end
        if ~isfield(out, 'funcCount') && isfield(out, 'funccount'), out.funcCount = out.funccount; end
        end

        function sols = ms_merge(sols, x, f, ef, out, x0, xtol, ftol)
        for j = 1:numel(sols)
            dx = norm(x(:) - sols(j).X(:)) <= xtol * max(1, norm(x(:)));
            df = abs(f - sols(j).Fval) <= ftol * max(1, abs(f));
            if dx && df
                sols(j).X0{end + 1} = x0;
                if f < sols(j).Fval, sols(j).X = x; sols(j).Fval = f; sols(j).Exitflag = ef; sols(j).Output = out; end
                return
            end
        end
        sols(end + 1) = struct('X', x, 'Fval', f, 'Exitflag', ef, 'Output', out, 'X0', {{x0}});
        end
    end
end
