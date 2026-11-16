//! Optimization Toolbox solvers (`linprog`, `optimoptions`).
//!
//! Expected values come from MATLAB R2026b (the only release available to the author; linprog's
//! default became HiGHS's dual simplex in R2024a, so only solutions, objective values and exit
//! flags are compared, not iteration counts) and from problems with analytic optima.
//! `data/optim_lp60.m` solves 60 random LPs (35 optimal, 5 infeasible, 20 unbounded);
//! `data/optim_lp60_matlab.txt` is MATLAB's output for the same calls.

use livre_lang::{Interpreter, OutputKind, Value};

fn run(src: &str) -> (Interpreter, Vec<String>) {
    let mut it = Interpreter::new();
    let _ = it.run_source(src);
    let out = it
        .take_outputs()
        .into_iter()
        .map(|o| match o.kind {
            OutputKind::Text(t) => t,
            OutputKind::Value { text, .. } => text,
            OutputKind::Error(e) => format!("Error: {e}"),
            OutputKind::Warning(w) => format!("Warning: {w}"),
            OutputKind::Figure(_) => String::new(),
        })
        .collect();
    (it, out)
}

fn num(it: &Interpreter, name: &str) -> Vec<f64> {
    match it.get_var(name) {
        Some(Value::Num(m)) => m.data.clone(),
        other => panic!("{name} = {other:?}"),
    }
}

fn close(it: &Interpreter, name: &str, want: &[f64], tol: f64) {
    let got = num(it, name);
    assert_eq!(got.len(), want.len(), "{name}: {got:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= tol * w.abs().max(1.0), "{name}: got {got:?}, want {want:?}");
    }
}

fn no_errors(out: &[String]) {
    assert!(!out.iter().any(|l| l.starts_with("Error")), "{out:?}");
}

#[test]
fn linprog_small_problems() {
    let (it, out) = run(
        "o = optimoptions('linprog', 'Display', 'off');
         [x1, f1, e1] = linprog([-1; 4], [-3 1; 1 2], [6; 4], [], [], [-inf; -3], [], o);
         [x2, f2, e2, out2, lam2] = linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0], [], o);
         [x3, f3, e3] = linprog([1; 1; 1], [], [], [1 1 1; 1 -1 0], [6; 1], [0; 0; 1], [10; 10; 1], o);
         [x4, f4, e4] = linprog([1; 1], [1 1; -1 -1], [1; -3], [], [], [], [], o);
         [x5, f5, e5] = linprog([-1; 0], [1 -1], 1, [], [], [0; 0], [], o);
         l2 = lam2.ineqlin; c2 = out2.constrviolation; g2 = out2.firstorderopt;",
    );
    no_errors(&out);
    // MATLAB R2026b: LP1 x = [10 -3], f = -22; LP2 x = [2 6], f = -36, ineqlin = [0 1.5 1];
    // LP3 x = [3 2 1], f = 6; LP4 exitflag -2; LP5 exitflag -3.
    close(&it, "x1", &[10.0, -3.0], 1e-9);
    close(&it, "f1", &[-22.0], 1e-9);
    close(&it, "e1", &[1.0], 0.0);
    close(&it, "x2", &[2.0, 6.0], 1e-9);
    close(&it, "f2", &[-36.0], 1e-9);
    close(&it, "l2", &[0.0, 1.5, 1.0], 1e-9);
    close(&it, "c2", &[0.0], 1e-12);
    close(&it, "g2", &[0.0], 1e-9);
    close(&it, "x3", &[3.0, 2.0, 1.0], 1e-9);
    close(&it, "f3", &[6.0], 1e-9);
    close(&it, "e4", &[-2.0], 0.0);
    close(&it, "e5", &[-3.0], 0.0);
    // Infeasible and unbounded problems return [] for x and fval.
    for v in ["x4", "f4", "x5", "f5"] {
        assert!(num(&it, v).is_empty(), "{v}");
    }
}

#[test]
fn linprog_dense_30x20() {
    // A 30x20 LP built from a feasible point; MATLAB R2026b: f = -9.533333333. The optimum is
    // not unique (MATLAB returns x(1:5) = [0 0.833333333 0.666666667 1 0]), so only the value
    // and feasibility are checked.
    let (it, out) = run(
        "m = 30; n = 20;
         Am = mod((1:m)' * (1:n) * 7, 11) / 11 - 0.4;
         bm = Am * (mod((1:n)', 3) / 3) + 0.5;
         fm = -(mod((1:n)', 5) + 1) / 5;
         [x, f, e] = linprog(fm, Am, bm, [], [], zeros(n, 1), ones(n, 1), optimoptions('linprog', 'Display', 'off'));
         viol = max([Am * x - bm; -x; x - 1]);",
    );
    no_errors(&out);
    close(&it, "f", &[-9.533333333333333], 1e-9);
    close(&it, "e", &[1.0], 0.0);
    assert!(num(&it, "viol")[0] <= 1e-9);
}

#[test]
fn linprog_sixty_random_lps_match_matlab() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let src = std::fs::read_to_string(dir.join("optim_lp60.m")).unwrap();
    let want = std::fs::read_to_string(dir.join("optim_lp60_matlab.txt")).unwrap();
    let (it, out) = run(&src);
    no_errors(&out);
    let got = num(&it, "out"); // 60x3, column-major: exitflag, fval, iterations
    let mut lines = 0;
    for line in want.lines() {
        // SIMPLEX  k ef= e f=v it=i
        let f: Vec<&str> = line.split_whitespace().collect();
        let k: usize = f[1].parse().unwrap();
        let ef: f64 = line.split("ef=").nth(1).unwrap().split_whitespace().next().unwrap().parse().unwrap();
        let fv: f64 = line.split(" f=").nth(1).unwrap().split_whitespace().next().unwrap().parse().unwrap();
        assert_eq!(got[k], ef, "LP {k}: exitflag");
        if fv.is_nan() {
            assert!(got[60 + k].is_nan(), "LP {k}: fval {}", got[60 + k]);
        } else {
            assert!((got[60 + k] - fv).abs() <= 1e-7 * fv.abs().max(1.0), "LP {k}: fval {} vs {fv}", got[60 + k]);
        }
        lines += 1;
    }
    assert_eq!(lines, 60);
}

#[test]
fn linprog_problem_struct_options_and_display() {
    let (it, out) = run(
        "p = struct('f', [-3; -5], 'Aineq', [1 0; 0 2; 3 2], 'bineq', [4; 12; 18], 'lb', [0; 0], 'solver', 'linprog');
         p.options = optimoptions('linprog', 'Display', 'off');
         [x, f, e, out] = linprog(p);
         alg = out.algorithm;
         x2 = linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0], [], optimset('Display', 'off'));
         [~, ~, e3, out3] = linprog([-1; -1], [1 1], 4, [], [], [0; 0], [], optimoptions('linprog', 'Display', 'off', 'MaxIterations', 0));
         it3 = out3.iterations;
         linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0]);",
    );
    no_errors(&out);
    close(&it, "x", &[2.0, 6.0], 1e-9);
    close(&it, "e", &[1.0], 0.0);
    close(&it, "x2", &[2.0, 6.0], 1e-9);
    close(&it, "e3", &[0.0], 0.0);
    close(&it, "it3", &[0.0], 0.0);
    assert!(matches!(it.get_var("alg"), Some(v) if v.text().unwrap() == "dual-simplex"));
    // The default display level ('final') prints the exit message.
    assert!(out.iter().any(|l| l.contains("Optimal solution found.")), "{out:?}");
}

#[test]
fn linprog_errors_and_bounds() {
    let (_, out) = run("linprog([1; 1], [1 1 1], 1)");
    assert!(out.iter().any(|l| l.contains("The number of columns in A must be the same as the number of elements of f.")), "{out:?}");
    let (it, out) = run(
        "[x, f, e] = linprog([1; 1], [], [], [], [], [0; 2], [1; 1], optimoptions('linprog', 'Display', 'off'));
         ef_bounds = e;",
    );
    no_errors(&out);
    close(&it, "ef_bounds", &[-2.0], 0.0);
}

#[test]
fn optimoptions_forms() {
    let (it, out) = run(
        "o = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 50);
         o2 = optimoptions(o, 'Display', 'off');
         o3 = optimoptions(@linprog, o2);
         s1 = o.SolverName; d2 = o2.Display; s3 = o3.SolverName; m3 = o3.MaxIterations;",
    );
    no_errors(&out);
    assert_eq!(it.get_var("s1").unwrap().text().unwrap(), "fmincon");
    assert_eq!(it.get_var("d2").unwrap().text().unwrap(), "off");
    assert_eq!(it.get_var("s3").unwrap().text().unwrap(), "linprog");
    close(&it, "m3", &[50.0], 0.0);
    let (_, out) = run("optimoptions('linprog', 'Display')");
    assert!(out.iter().any(|l| l.contains("Option 'Display' does not have a value.")), "{out:?}");
}
