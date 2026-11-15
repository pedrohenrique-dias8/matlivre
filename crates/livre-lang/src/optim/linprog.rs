//! `linprog`: linear programming with the bounded dual simplex of `simplex.rs`.
//!
//! x = linprog(f, A, b, Aeq, beq, lb, ub, options) or linprog(problem), up to
//! [x, fval, exitflag, output, lambda]. exitflag: 1 solved, 0 iteration limit, -2 infeasible,
//! -3 unbounded, -4 numerical breakdown. lambda follows MATLAB's sign convention: the Lagrangian
//! is f'x + ineqlin'(Ax - b) + eqlin'(Aeq x - beq) - lower'(x - lb) + upper'(x - ub), with
//! ineqlin, lower and upper >= 0.
//!
//! The algorithm is the same for every `Algorithm` option ('dual-simplex' in MATLAB R2023b); an
//! interior-point path will come with `quadprog`.

use super::simplex::{self, Rows, Status};
use super::{bounds, column, display_level, getopt, real_arg, Display};
use crate::builtins::Ret;
use crate::error::rt;
use crate::interp::{Interpreter, OutputKind};
use crate::value::*;

pub fn linprog(it: &mut Interpreter, a: &[Value], nargout: usize) -> Ret {
    // Positional arguments or a problem structure.
    let args: Vec<Option<Value>> = match a {
        [Value::Struct(p)] => {
            let f = |k: &str| p.get(k).cloned();
            vec![f("f"), f("Aineq"), f("bineq"), f("Aeq"), f("beq"), f("lb"), f("ub"), f("options")]
        }
        _ => {
            if a.is_empty() {
                return rt("Not enough input arguments.");
            }
            if a.len() > 8 {
                return rt("Too many input arguments.");
            }
            (0..8).map(|i| a.get(i).cloned()).collect()
        }
    };
    let f = real_arg(args[0].as_ref(), "LINPROG", "f")?;
    let n = f.numel();
    if n == 0 {
        return rt("LINPROG requires a nonempty objective vector f.");
    }
    let mat = |i: usize, name: &str| real_arg(args[i].as_ref(), "LINPROG", name);
    let (am, bm, aeq, beq, lbm, ubm) = (mat(1, "A")?, mat(2, "b")?, mat(3, "Aeq")?, mat(4, "beq")?, mat(5, "lb")?, mat(6, "ub")?);
    let o = match &args[7] {
        Some(v) => super::options_arg(v)?,
        None => None,
    };

    let ma = if am.is_empty() { 0 } else { am.rows() };
    let me = if aeq.is_empty() { 0 } else { aeq.rows() };
    if ma > 0 && (am.cols() != n || bm.numel() != ma) || ma == 0 && bm.numel() != 0 {
        return rt(format!("A must have {n} columns and as many rows as b has elements."));
    }
    if me > 0 && (aeq.cols() != n || beq.numel() != me) || me == 0 && beq.numel() != 0 {
        return rt(format!("Aeq must have {n} columns and as many rows as beq has elements."));
    }
    let lb = bounds(&lbm, n, f64::NEG_INFINITY);
    let ub = bounds(&ubm, n, f64::INFINITY);
    let level = display_level(o, Display::Final)?;
    let maxit = match getopt(o, &["MaxIterations", "MaxIter"]) {
        Some(v) => v.scalar()?.max(0.0) as usize,
        None => 10 * (n + ma + me) + 1000,
    };

    // [A; Aeq] in column-major order, and [b; beq].
    let m = ma + me;
    let mut rows = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..ma {
            rows[j * m + i] = am.get(i, j);
        }
        for i in 0..me {
            rows[j * m + ma + i] = aeq.get(i, j);
        }
    }
    let rhs: Vec<f64> = bm.data.iter().chain(&beq.data).copied().collect();
    let c = f.data.clone();

    let mut x = None;
    let (exitflag, message, iterations, mut lambda) = if lb.iter().zip(&ub).any(|(l, u)| l > u) {
        (-2.0, "No feasible point: lb(i) > ub(i) for some i.".to_string(), 0, None)
    } else {
        let rowsref = Rows { data: &rows, rows: m, ineq: ma };
        let s = simplex::solve(&c, &rowsref, &rhs, &lb, &ub, simplex::Options { maxit, tol: 1e-9, basis: None });
        let (flag, msg) = match s.status {
            Status::Optimal => (1.0, "Optimal solution found.".to_string()),
            Status::Infeasible => (-2.0, "No feasible point found: the constraints cannot be satisfied.".to_string()),
            Status::Unbounded => (-3.0, "Problem is unbounded: the objective decreases without limit along a feasible direction.".to_string()),
            Status::IterationLimit => (0.0, format!("Stopped: the number of iterations reached MaxIterations = {maxit}.")),
            Status::Singular => (-4.0, "Stopped: the basis matrix became singular.".to_string()),
        };
        let mut lam = None;
        if matches!(s.status, Status::Optimal | Status::IterationLimit) {
            let ineqlin: Vec<f64> = s.y[..ma].iter().map(|y| (-y).max(0.0)).collect();
            let eqlin: Vec<f64> = s.y[ma..].iter().map(|y| -y).collect();
            let lower: Vec<f64> = (0..n).map(|j| if s.basis.stat[j] == -1 { s.d[j].max(0.0) } else { 0.0 }).collect();
            let upper: Vec<f64> = (0..n).map(|j| if s.basis.stat[j] == 1 { (-s.d[j]).max(0.0) } else { 0.0 }).collect();
            lam = Some((ineqlin, eqlin, lower, upper));
            x = Some(s.x.clone());
        }
        (flag, msg, s.iterations, lam)
    };

    let mut output = StructArray::scalar();
    output.set("iterations", Value::num(iterations as f64));
    let (mut viol, mut firstorder) = (f64::NAN, f64::NAN);
    let mut fval = Value::empty();
    if let (Some(x), Some((ineqlin, eqlin, lower, upper))) = (&x, &lambda) {
        fval = Value::num(c.iter().zip(x).map(|(a, b)| a * b).sum());
        let mut v = 0.0f64;
        let mut grad = c.clone();
        for i in 0..m {
            let ax: f64 = (0..n).map(|j| rows[j * m + i] * x[j]).sum();
            let r = ax - rhs[i];
            v = v.max(if i < ma { r } else { r.abs() });
            let li = if i < ma { ineqlin[i] } else { eqlin[i - ma] };
            for j in 0..n {
                grad[j] += rows[j * m + i] * li;
            }
        }
        for j in 0..n {
            v = v.max(lb[j] - x[j]).max(x[j] - ub[j]);
            grad[j] += upper[j] - lower[j];
        }
        viol = v;
        firstorder = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    }
    output.set("constrviolation", Value::num(viol));
    output.set("message", Value::chars(&message));
    output.set("algorithm", Value::chars("dual-simplex"));
    output.set("firstorderopt", Value::num(firstorder));

    let show = match level {
        Display::Off => false,
        Display::Notify => exitflag <= 0.0,
        _ => true,
    };
    if show {
        it.emit(OutputKind::Text(format!("\n{message}\n\n")));
    }

    let (ineqlin, eqlin, lower, upper) = lambda.take().unwrap_or_else(|| (vec![0.0; ma], vec![0.0; me], vec![0.0; n], vec![0.0; n]));
    let mut lam = StructArray::scalar();
    lam.set("ineqlin", column(ineqlin));
    lam.set("eqlin", column(eqlin));
    lam.set("lower", column(lower));
    lam.set("upper", column(upper));

    let xv = match x {
        Some(x) => column(x),
        None => Value::empty(),
    };
    Ok(Some(vec![xv, fval, Value::num(exitflag), Value::Struct(output), Value::Struct(lam)].into_iter().take(nargout.max(1)).collect()))
}
