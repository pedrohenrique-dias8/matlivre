//! `intlinprog`: mixed-integer linear programming by LP-based branch and bound.
//!
//! Relaxations are solved by the bounded dual simplex of `simplex.rs`, each node warm-started
//! from its parent's optimal basis. At the root, rounds of Gomory mixed-integer cuts are added
//! (CutGeneration 'basic' = 5 rounds, 'intermediate' = 10, 'advanced' = 20, 'none'; at most 20
//! cuts per round). Search: depth-first dives until an integer-feasible point is found, then best
//! bound (NodeSelection 'minobj'), or depth-first throughout ('mininfeas'). Branching variable:
//! BranchRule 'maxfun' (default here), 'reliability' (pseudocost product rule with strong
//! branching on up to 8 unreliable candidates), 'maxpscost' or 'mostfractional'. When the
//! objective can only take integer values, nodes whose bound rounds up to the incumbent are
//! pruned. A rounding heuristic tries the rounded LP point at every node; x0, if integer
//! feasible, is the first incumbent. No presolve, and cuts only at the root.
//!
//! exitflag: 1 optimal (gap within tolerance), 2 stopped with an integer feasible point, 0
//! stopped without one, -1 stopped by an output function before a feasible point, -2
//! infeasible, -3 LP relaxation unbounded.
//!
//! References: A. H. Land and A. G. Doig (1960), "An automatic method of solving discrete
//! programming problems", Econometrica 28; L. A. Wolsey, Integer Programming (1998), Ch. 7-8;
//! T. Achterberg, T. Koch and A. Martin (2005), "Branching rules revisited", Oper. Res. Lett. 33.
//! Clean room: no MathWorks code.

use super::simplex::{self, Basis, Rows, Solution, Status};
use super::{bounds, column, display_level, getopt, real_arg, Display};
use crate::builtins::Ret;
use crate::builtins_funfun::OptimHooks;
use crate::error::{rt, LangError};
use crate::interp::{Interpreter, OutputKind};
use crate::value::*;
use std::time::Instant;

type R<T> = Result<T, LangError>;

/// Constraint data with the inequality rows kept separately so cuts can be appended.
struct Lp {
    n: usize,
    c: Vec<f64>,
    /// Inequality rows (row-major), then equality rows.
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    aeq: Vec<Vec<f64>>,
    beq: Vec<f64>,
    /// [A; Aeq] column-major and [b; beq], rebuilt when rows change.
    cm: Vec<f64>,
    rhs: Vec<f64>,
}

impl Lp {
    fn rebuild(&mut self) {
        let m = self.a.len() + self.aeq.len();
        let mut cm = vec![0.0; m * self.n];
        for (i, row) in self.a.iter().chain(&self.aeq).enumerate() {
            for (j, v) in row.iter().enumerate() {
                cm[j * m + i] = *v;
            }
        }
        self.cm = cm;
        self.rhs = self.b.iter().chain(&self.beq).copied().collect();
    }

    fn rows(&self) -> Rows<'_> {
        Rows { data: &self.cm, rows: self.a.len() + self.aeq.len(), ineq: self.a.len() }
    }

    fn solve(&self, lb: &[f64], ub: &[f64], basis: Option<Basis>) -> Solution {
        simplex::solve(&self.c, &self.rows(), &self.rhs, lb, ub, simplex::Options { maxit: 50 * (self.rhs.len() + self.n) + 1000, tol: 1e-9, basis })
    }

    fn obj(&self, x: &[f64]) -> f64 {
        self.c.iter().zip(x).map(|(a, b)| a * b).sum()
    }
}

fn dot(a: &[f64], x: &[f64]) -> f64 {
    a.iter().zip(x).map(|(a, b)| a * b).sum()
}

/// Integer feasibility of x for the original constraints.
#[allow(clippy::too_many_arguments)]
fn feasible(x: &[f64], a: &[Vec<f64>], b: &[f64], aeq: &[Vec<f64>], beq: &[f64], lb: &[f64], ub: &[f64], intcon: &[usize], itol: f64, ctol: f64) -> bool {
    intcon.iter().all(|&j| (x[j] - x[j].round()).abs() <= itol)
        && x.iter().zip(lb).all(|(x, l)| *x >= l - ctol)
        && x.iter().zip(ub).all(|(x, u)| *x <= u + ctol)
        && a.iter().zip(b).all(|(r, b)| dot(r, x) - b <= ctol * b.abs().max(1.0))
        && aeq.iter().zip(beq).all(|(r, b)| (dot(r, x) - b).abs() <= ctol * b.abs().max(1.0))
}

/// LP relaxation of a node, warm-started from `basis`: the clamped point and the optimal basis,
/// or None when infeasible (or on numerical failure).
fn node_lp(lp: &Lp, lb: &[f64], ub: &[f64], basis: &Basis) -> Option<(Vec<f64>, Basis)> {
    if lb.iter().zip(ub).any(|(l, u)| *l > u + 1e-9) {
        return None;
    }
    let s = lp.solve(lb, ub, Some(basis.clone()));
    if s.status != Status::Optimal {
        return None;
    }
    let x = s.x.iter().zip(lb.iter().zip(ub)).map(|(x, (l, u))| x.max(*l).min(*u)).collect();
    Some((x, s.basis))
}

/// Gomory mixed-integer cuts from the rows of the optimal tableau whose basic variable is integer
/// and fractional. Nonbasic variables are shifted to their active bounds; slacks of inequality
/// rows with integer data on integer variables count as integer. A cut that uses a column at an
/// artificial bound is skipped. Returned as rows a'x <= b.
fn gmi_cuts(lp: &Lp, s: &Solution, isint_var: &[bool], maxcuts: usize) -> Vec<(Vec<f64>, f64)> {
    let n = lp.n;
    let ma = lp.a.len();
    let m = ma + lp.aeq.len();
    let (basic, stat) = (&s.basis.b, &s.basis.stat);
    let mut isint = isint_var.to_vec();
    isint.resize(n + m, false);
    for i in 0..ma {
        let row = &lp.a[i];
        isint[n + i] = row.iter().enumerate().all(|(j, v)| *v == 0.0 || (isint_var[j] && (v - v.round()).abs() < 1e-12)) && (lp.b[i] - lp.b[i].round()).abs() < 1e-12;
    }
    let mut cand: Vec<(usize, f64)> = (0..m)
        .filter(|&r| isint[basic[r]])
        .map(|r| (r, s.xz[basic[r]] - s.xz[basic[r]].floor()))
        .filter(|&(_, fr)| fr > 0.01 && fr < 0.99)
        .collect();
    cand.sort_by(|a, b| (a.1 - 0.5).abs().partial_cmp(&(b.1 - 0.5).abs()).unwrap_or(std::cmp::Ordering::Equal));
    let which: Vec<usize> = cand.iter().map(|c| c.0).collect();
    let Some(tab) = simplex::tableau_rows(&lp.rows(), n, basic, &which) else { return Vec::new() };
    let mut cuts = Vec::new();
    for (k, &(_, f0)) in cand.iter().enumerate() {
        if cuts.len() >= maxcuts {
            break;
        }
        let alpha = &tab[k];
        let nb: Vec<usize> = (0..n + m).filter(|&j| stat[j] != 0 && s.la[j] != s.ua[j] && alpha[j].abs() > 1e-11).collect();
        if nb.iter().any(|&j| s.atart[j]) {
            continue;
        }
        let mut pi = vec![0.0; n];
        let mut pi0 = 0.0;
        for &j in &nb {
            let a = if stat[j] == 1 { -alpha[j] } else { alpha[j] };
            let g = if isint[j] {
                let fj = a - a.floor();
                if fj > f0 {
                    (1.0 - fj) / (1.0 - f0)
                } else {
                    fj / f0
                }
            } else if a < 0.0 {
                -a / (1.0 - f0)
            } else {
                a / f0
            };
            if g == 0.0 {
                continue;
            }
            // sum g_j x'_j >= 1 with x' = x - l (at lower), u - x (at upper), s = b - A x (slack)
            if j < n {
                if stat[j] == -1 {
                    pi[j] += g;
                    pi0 -= g * s.la[j];
                } else {
                    pi[j] -= g;
                    pi0 += g * s.ua[j];
                }
            } else {
                let row = j - n; // only inequality slacks are not fixed
                for (p, v) in pi.iter_mut().zip(&lp.a[row]) {
                    *p -= g * v;
                }
                pi0 += g * lp.b[row];
            }
        }
        // pi'x + pi0 >= 1  ->  -pi'x <= pi0 - 1
        let nz: Vec<f64> = pi.iter().filter(|v| **v != 0.0).map(|v| v.abs()).collect();
        if nz.is_empty() {
            continue;
        }
        let (lo, hi) = nz.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        if hi / lo > 1e6 {
            continue;
        }
        cuts.push((pi.iter().map(|v| -v / hi).collect(), (pi0 - 1.0) / hi));
    }
    cuts
}

/// The node cannot improve the incumbent u by more than the gap tolerances.
fn prune(bound: f64, u: f64, agap: f64, rgap: f64, objint: bool) -> bool {
    if !u.is_finite() {
        return false;
    }
    u - bound <= agap.max(rgap * (u.abs() + 1.0)) || objint && (bound - 1e-6).ceil() >= u
}

fn gap(u: f64, l: f64) -> f64 {
    if u.is_finite() {
        100.0 * (u - l).max(0.0) / (u.abs() + 1.0)
    } else {
        f64::INFINITY
    }
}

fn opt_values(u: f64, nodes: usize, nfeas: usize, gap: f64, phase: &str) -> Value {
    let mut s = StructArray::scalar();
    s.set("phase", Value::chars(phase));
    s.set("fval", if u.is_finite() { Value::num(u) } else { Value::empty() });
    s.set("numnodes", Value::num(nodes as f64));
    s.set("numfeaspoints", Value::num(nfeas as f64));
    s.set("relativegap", Value::num(gap));
    s.set("iteration", Value::num(nodes as f64));
    Value::Struct(s)
}

struct Node {
    lb: Vec<f64>,
    ub: Vec<f64>,
    bound: f64,
    depth: usize,
    /// Branching that created the node: variable (or usize::MAX at the root), direction
    /// (0 down, 1 up) and distance moved.
    var: usize,
    dir: usize,
    dist: f64,
    basis: Basis,
}

pub fn intlinprog(it: &mut Interpreter, a: &[Value], nargout: usize) -> Ret {
    const S: &str = "INTLINPROG";
    let args: Vec<Option<Value>> = match a {
        [Value::Struct(p)] => {
            let f = |k: &str| p.get(k).cloned();
            vec![f("f"), f("intcon"), f("Aineq"), f("bineq"), f("Aeq"), f("beq"), f("lb"), f("ub"), f("x0"), f("options")]
        }
        _ => {
            if a.len() < 2 {
                return rt("Not enough input arguments.");
            }
            if a.len() > 10 {
                return rt("Too many input arguments.");
            }
            (0..10).map(|i| a.get(i).cloned()).collect()
        }
    };
    let mat = |i: usize, name: &str| real_arg(args[i].as_ref(), S, name);
    let f = mat(0, "f")?;
    let n = f.numel();
    if n == 0 {
        return rt("INTLINPROG requires a nonempty objective vector f.");
    }
    let (icm, am, bm, aeqm, beqm, lbm, ubm, x0m) = (mat(1, "intcon")?, mat(2, "A")?, mat(3, "b")?, mat(4, "Aeq")?, mat(5, "beq")?, mat(6, "lb")?, mat(7, "ub")?, mat(8, "x0")?);
    let o = match &args[9] {
        Some(v) => super::options_arg(v)?,
        None => None,
    };
    let mut intcon: Vec<usize> = Vec::new();
    for &v in &icm.data {
        if v != v.round() || v < 1.0 || v > n as f64 {
            return rt(format!("INTCON must contain integers between 1 and {n}."));
        }
        intcon.push(v as usize - 1);
    }
    intcon.sort_unstable();
    intcon.dedup();
    let mut isint = vec![false; n];
    for &j in &intcon {
        isint[j] = true;
    }
    let ma = if am.is_empty() { 0 } else { am.rows() };
    let me = if aeqm.is_empty() { 0 } else { aeqm.rows() };
    if ma > 0 && (am.cols() != n || bm.numel() != ma) || ma == 0 && bm.numel() != 0 {
        return rt(format!("A must have {n} columns and as many rows as b has elements."));
    }
    if me > 0 && (aeqm.cols() != n || beqm.numel() != me) || me == 0 && beqm.numel() != 0 {
        return rt(format!("Aeq must have {n} columns and as many rows as beq has elements."));
    }
    let rowsof = |m: &Matrix, k: usize| -> Vec<Vec<f64>> { (0..k).map(|i| (0..n).map(|j| m.get(i, j)).collect()).collect() };
    let mut lb = bounds(&lbm, n, f64::NEG_INFINITY);
    let mut ub = bounds(&ubm, n, f64::INFINITY);
    for &j in &intcon {
        lb[j] = (lb[j] - 1e-9).ceil();
        ub[j] = (ub[j] + 1e-9).floor();
    }
    let mut lp = Lp { n, c: f.data.clone(), a: rowsof(&am, ma), b: bm.data.clone(), aeq: rowsof(&aeqm, me), beq: beqm.data.clone(), cm: Vec::new(), rhs: Vec::new() };
    lp.rebuild();
    let (a0, b0) = (lp.a.clone(), lp.b.clone()); // original rows, for feasibility checks

    let num = |names: &[&str], d: f64| -> R<f64> { getopt(o, names).map(|v| v.scalar()).unwrap_or(Ok(d)) };
    let text = |names: &[&str], d: &str| -> R<String> { getopt(o, names).map(|v| v.text().map(|t| t.to_lowercase())).unwrap_or(Ok(d.to_string())) };
    let maxnodes = num(&["MaxNodes"], 1e7)?;
    let maxtime = num(&["MaxTime"], 7200.0)?;
    let rgap = num(&["RelativeGapTolerance", "TolGapRel"], 1e-4)?;
    let agap = num(&["AbsoluteGapTolerance", "TolGapAbs"], 0.0)?;
    let itol = num(&["IntegerTolerance", "TolInteger"], 1e-5)?;
    let ctol = num(&["ConstraintTolerance", "TolCon"], 1e-4)?;
    let maxfeas = num(&["MaxFeasiblePoints"], f64::INFINITY)?;
    let brule = text(&["BranchRule", "BranchingRule"], "maxfun")?;
    let nsel = text(&["NodeSelection"], "minobj")?;
    let cutmode = text(&["CutGeneration"], "basic")?;
    let level = display_level(o, Display::Iter)?;
    let mut hooks = OptimHooks::new(o);
    let c = lp.c.clone();
    let objint = (0..n).all(|j| if isint[j] { (c[j] - c[j].round()).abs() < 1e-12 } else { c[j] == 0.0 });
    let t0 = Instant::now();

    let mut output = StructArray::scalar();
    let finish = |it: &mut Interpreter, output: &mut StructArray, x: Option<Vec<f64>>, flag: f64, msg: String| -> Vec<Value> {
        output.set("message", Value::chars(&msg));
        let show = match level {
            Display::Off => false,
            Display::Notify => flag <= 0.0,
            _ => true,
        };
        if show {
            it.emit(OutputKind::Text(format!("\n{msg}\n\n")));
        }
        let fval = x.as_ref().map(|x| Value::num(dot(&c, x))).unwrap_or_else(Value::empty);
        let xv = x.map(column).unwrap_or_else(Value::empty);
        vec![xv, fval, Value::num(flag), Value::Struct(output.clone())]
    };
    for k in ["relativegap", "absolutegap"] {
        output.set(k, Value::empty());
    }
    output.set("numfeaspoints", Value::num(0.0));
    output.set("numnodes", Value::num(0.0));
    output.set("constrviolation", Value::empty());
    output.set("message", Value::chars(""));
    output.set("numcuts", Value::num(0.0));
    let ret = |v: Vec<Value>| -> Ret { Ok(Some(v.into_iter().take(nargout.max(1)).collect())) };

    let mut u = f64::INFINITY;
    let mut xbest: Option<Vec<f64>> = None;
    let mut nfeas = 0usize;
    if x0m.numel() == n && feasible(&x0m.data, &a0, &b0, &lp.aeq, &lp.beq, &lb, &ub, &intcon, itol, ctol) {
        let mut xs = x0m.data.clone();
        for &j in &intcon {
            xs[j] = xs[j].round();
        }
        u = dot(&c, &xs);
        xbest = Some(xs);
        nfeas = 1;
    }
    if lb.iter().zip(&ub).any(|(l, u)| l > u) {
        let v = finish(it, &mut output, None, -2.0, "No feasible solution found: lb > ub after rounding the integer bounds.".into());
        return ret(v);
    }
    let mut root = lp.solve(&lb, &ub, None);
    match root.status {
        Status::Infeasible => {
            let v = finish(it, &mut output, None, -2.0, "No feasible solution found: the LP relaxation is infeasible.".into());
            return ret(v);
        }
        Status::Unbounded => {
            let v = finish(it, &mut output, None, -3.0, "Problem is unbounded: the LP relaxation is unbounded.".into());
            return ret(v);
        }
        _ => {}
    }
    let rootlp = lp.obj(&root.x);

    // ---- Gomory mixed-integer cuts at the root
    let rounds = match cutmode.as_str() {
        "none" => 0,
        "basic" => 5,
        "intermediate" => 10,
        _ => 20,
    };
    let mut ncuts = 0usize;
    let mut stall = 0;
    for _ in 0..rounds {
        let cuts = gmi_cuts(&lp, &root, &isint, 20);
        if cuts.is_empty() {
            break;
        }
        let k = cuts.len();
        let ma_old = lp.a.len();
        // warm start: the new slacks are basic, the equality slacks move by k positions
        let mut bs: Vec<usize> = root.basis.b.iter().map(|&j| if j >= n + ma_old { j + k } else { j }).collect();
        bs.extend(n + ma_old..n + ma_old + k);
        let mut st = root.basis.stat[..n + ma_old].to_vec();
        st.extend(std::iter::repeat(0i8).take(k));
        st.extend_from_slice(&root.basis.stat[n + ma_old..]);
        for (row, rhs) in cuts {
            lp.a.push(row);
            lp.b.push(rhs);
        }
        lp.rebuild();
        let fold = lp.obj(&root.x);
        let s2 = lp.solve(&lb, &ub, Some(Basis { b: bs, stat: st }));
        if s2.status != Status::Optimal {
            lp.a.truncate(ma_old); // numerical trouble: drop the round
            lp.b.truncate(ma_old);
            lp.rebuild();
            break;
        }
        ncuts += k;
        root = s2;
        if lp.obj(&root.x) - fold <= 1e-6 * fold.abs().max(1.0) {
            stall += 1; // degenerate LPs can keep the value for a while
            if stall >= 3 {
                break;
            }
        } else {
            stall = 0;
        }
    }
    if level == Display::Iter {
        it.emit(OutputKind::Text(format!(
            "\nLP relaxation: f = {}; after {ncuts} Gomory cuts: f = {}\n\n    Nodes      Total time (s)   Num int solution   Integer fval     Relative gap (%)\n",
            fmt_g(rootlp),
            fmt_g(lp.obj(&root.x))
        )));
    }

    let rootx: Vec<f64> = root.x.clone();
    let mut nodes = vec![Node { lb: lb.clone(), ub: ub.clone(), bound: lp.obj(&rootx), depth: 0, var: usize::MAX, dir: 0, dist: 0.0, basis: root.basis.clone() }];
    let mut pc = vec![[0.0f64; 2]; n];
    let mut pcn = vec![[0usize; 2]; n];
    let mut nnodes = 0usize;
    let mut stopreason = "";
    let mut userstop = false;
    if hooks.active() {
        userstop = hooks.call(it, &Value::empty(), &opt_values(u, 0, nfeas, f64::INFINITY, "init"), "init")?;
        if userstop {
            stopreason = "output function";
        }
    }
    let dfs_always = nsel == "mininfeas" || nsel == "depthfirst";
    let avg_f = intcon.iter().map(|&j| c[j].abs()).sum::<f64>() / (intcon.len().max(1) as f64) + f64::EPSILON;
    while !nodes.is_empty() && !userstop {
        // depth-first until the first incumbent, best bound afterwards (ties to the deeper node)
        let k = if !u.is_finite() || dfs_always {
            argbest(nodes.iter().map(|nd| nd.depth as f64 - 1e-9 * nd.bound), true)
        } else {
            argbest(nodes.iter().map(|nd| nd.bound - 1e-9 * nd.depth as f64), false)
        };
        let node = nodes.swap_remove(k);
        if !prune(node.bound, u, agap, rgap, objint) {
            let solved = if nnodes == 0 { Some((rootx.clone(), root.basis.clone())) } else { node_lp(&lp, &node.lb, &node.ub, &node.basis) };
            nnodes += 1;
            if let Some((xn, nbasis)) = solved {
                let fn_ = lp.obj(&xn);
                if node.var != usize::MAX && node.dist > 0.0 {
                    pc[node.var][node.dir] += (fn_ - node.bound).max(0.0) / node.dist;
                    pcn[node.var][node.dir] += 1;
                }
                if !prune(fn_, u, agap, rgap, objint) {
                    // rounding heuristic
                    let mut xh = xn.clone();
                    for &j in &intcon {
                        xh[j] = xh[j].round();
                    }
                    if dot(&c, &xh) < u && feasible(&xh, &a0, &b0, &lp.aeq, &lp.beq, &lb, &ub, &intcon, itol, ctol) {
                        u = dot(&c, &xh);
                        xbest = Some(xh.clone());
                        nfeas += 1;
                    }
                    let frac: Vec<usize> = intcon.iter().copied().filter(|&j| (xn[j] - xn[j].round()).abs() > itol).collect();
                    if frac.is_empty() {
                        if dot(&c, &xh) < u {
                            u = dot(&c, &xh);
                            xbest = Some(xh);
                            nfeas += 1;
                        }
                    } else {
                        let fl: Vec<f64> = frac.iter().map(|&j| xn[j] - xn[j].floor()).collect();
                        let fr: Vec<f64> = frac.iter().map(|&j| (xn[j] - xn[j].round()).abs()).collect();
                        let pick = match brule.as_str() {
                            "mostfractional" => argbest(fl.iter().map(|f| f * (1.0 - f)), true),
                            "maxfun" => argbest(frac.iter().zip(&fr).map(|(&j, f)| c[j].abs() * (1.0 + f)), true),
                            _ => {
                                if brule == "reliability" {
                                    let mut unrel: Vec<usize> = (0..frac.len()).filter(|&q| pcn[frac[q]][0].min(pcn[frac[q]][1]) < 2).collect();
                                    if unrel.len() > 8 {
                                        unrel.sort_by(|&p, &q| (fl[p] - 0.5).abs().partial_cmp(&(fl[q] - 0.5).abs()).unwrap_or(std::cmp::Ordering::Equal));
                                        unrel.truncate(8);
                                    }
                                    for q in unrel {
                                        let vq = frac[q];
                                        let mut u1 = node.ub.clone();
                                        u1[vq] = xn[vq].floor();
                                        let mut l2 = node.lb.clone();
                                        l2[vq] = xn[vq].ceil();
                                        let pen = 1e6 * (1.0 + fn_.abs());
                                        let gd = node_lp(&lp, &node.lb, &u1, &nbasis).map(|(x, _)| (lp.obj(&x) - fn_).max(0.0)).unwrap_or(pen);
                                        let gu = node_lp(&lp, &l2, &node.ub, &nbasis).map(|(x, _)| (lp.obj(&x) - fn_).max(0.0)).unwrap_or(pen);
                                        pc[vq][0] += gd / fl[q].max(1e-9);
                                        pcn[vq][0] += 1;
                                        pc[vq][1] += gu / (1.0 - fl[q]).max(1e-9);
                                        pcn[vq][1] += 1;
                                    }
                                }
                                let score = |q: usize| {
                                    let j = frac[q];
                                    let qd = if pcn[j][0] == 0 { c[j].abs() + avg_f } else { pc[j][0] / pcn[j][0] as f64 };
                                    let qu = if pcn[j][1] == 0 { c[j].abs() + avg_f } else { pc[j][1] / pcn[j][1] as f64 };
                                    (qd * fl[q]).max(1e-6) * (qu * (1.0 - fl[q])).max(1e-6)
                                };
                                argbest((0..frac.len()).map(score), true)
                            }
                        };
                        let v = frac[pick];
                        let xv = xn[v];
                        // down branch, then up branch; the side closer to the LP value is pushed
                        // last, so depth-first search takes it first
                        let order = if xv - xv.floor() >= 0.5 { [1, 0] } else { [0, 1] };
                        for dir in order {
                            let (mut nl, mut nu) = (node.lb.clone(), node.ub.clone());
                            let dist = if dir == 0 {
                                nu[v] = xv.floor();
                                xv - xv.floor()
                            } else {
                                nl[v] = xv.ceil();
                                xv.ceil() - xv
                            };
                            nodes.push(Node { lb: nl, ub: nu, bound: fn_, depth: node.depth + 1, var: v, dir, dist, basis: nbasis.clone() });
                        }
                    }
                }
            }
        }
        let l = if nodes.is_empty() { u } else { nodes.iter().map(|nd| nd.bound).fold(f64::INFINITY, f64::min) };
        if level == Display::Iter && nnodes > 0 && (nnodes % 500 == 0 || nnodes <= 10) {
            it.emit(OutputKind::Text(format!("{nnodes:9} {:16.2} {nfeas:18} {:>16} {:>18}\n", t0.elapsed().as_secs_f64(), fmt_g(u), fmt_g3(gap(u, l)))));
        }
        if u.is_finite() && (u - l <= agap || u - l <= rgap * (u.abs() + 1.0) || objint && (l - 1e-6).ceil() >= u) {
            nodes.clear();
            break;
        }
        if hooks.active() {
            let xv = xbest.clone().map(column).unwrap_or_else(Value::empty);
            if hooks.call(it, &xv, &opt_values(u, nnodes, nfeas, gap(u, l), "branching"), "iter")? {
                stopreason = "output function";
                break;
            }
        }
        if nfeas as f64 >= maxfeas {
            stopreason = "MaxFeasiblePoints";
            break;
        }
        if nnodes as f64 >= maxnodes {
            stopreason = "MaxNodes";
            break;
        }
        if t0.elapsed().as_secs_f64() > maxtime {
            stopreason = "MaxTime";
            break;
        }
    }
    let l = nodes.iter().map(|nd| nd.bound).fold(u, f64::min);
    output.set("numnodes", Value::num(nnodes as f64));
    output.set("numfeaspoints", Value::num(nfeas as f64));
    output.set("numcuts", Value::num(ncuts as f64));
    let (x, flag, msg) = match xbest {
        None => match stopreason {
            "output function" => (None, -1.0, "Stopped by an output function before an integer feasible point was found.".to_string()),
            "" => (None, -2.0, "No integer feasible point exists.".to_string()),
            r => (None, 0.0, format!("Stopped ({r}) before finding an integer feasible point.")),
        },
        Some(xb) => {
            let ag = (u - l).max(0.0);
            let rg = 100.0 * ag / (u.abs() + 1.0);
            output.set("absolutegap", Value::num(ag));
            output.set("relativegap", Value::num(rg));
            let mut v = 0.0f64;
            for (r, b) in a0.iter().zip(&b0) {
                v = v.max(dot(r, &xb) - b);
            }
            for (r, b) in lp.aeq.iter().zip(&lp.beq) {
                v = v.max((dot(r, &xb) - b).abs());
            }
            for j in 0..n {
                v = v.max(lb[j] - xb[j]).max(xb[j] - ub[j]);
            }
            output.set("constrviolation", Value::num(v));
            if stopreason.is_empty() {
                (Some(xb), 1.0, format!("Optimal solution found: the relative gap is {}% (RelativeGapTolerance {}).", fmt_g(rg), fmt_g(rgap)))
            } else {
                (Some(xb), 2.0, format!("Stopped ({stopreason}) with an integer feasible point; relative gap {}%.", fmt_g(rg)))
            }
        }
    };
    let rg = gap(u, l);
    let fval = x.as_ref().map(|x| dot(&c, x)).unwrap_or(f64::NAN);
    let xv = x.clone().map(column).unwrap_or_else(Value::empty);
    let v = finish(it, &mut output, x, flag, msg);
    if hooks.active() {
        hooks.call(it, &xv, &opt_values(fval, nnodes, nfeas, rg, "done"), "done")?;
    }
    ret(v)
}

/// Index of the largest (or smallest) value, the first one on ties, as MATLAB's max/min.
fn argbest(v: impl Iterator<Item = f64>, largest: bool) -> usize {
    let mut best = (0, if largest { f64::NEG_INFINITY } else { f64::INFINITY });
    for (i, x) in v.enumerate() {
        if largest && x > best.1 || !largest && x < best.1 {
            best = (i, x);
        }
    }
    best.0
}

/// printf's %g with 6 significant digits.
fn fmt_g(v: f64) -> String {
    fmt_sig(v, 6)
}

fn fmt_g3(v: f64) -> String {
    fmt_sig(v, 3)
}

fn fmt_sig(v: f64, p: usize) -> String {
    if !v.is_finite() {
        return if v.is_nan() { "NaN".into() } else if v > 0.0 { "Inf".into() } else { "-Inf".into() };
    }
    if v == 0.0 {
        return "0".into();
    }
    let e = v.abs().log10().floor() as i32;
    if e < -4 || e >= p as i32 {
        let s = format!("{:.*e}", p - 1, v);
        let (m, ex) = s.split_once('e').unwrap();
        let m = if m.contains('.') { m.trim_end_matches('0').trim_end_matches('.') } else { m };
        let ex: i32 = ex.parse().unwrap();
        format!("{m}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs())
    } else {
        let d = (p as i32 - 1 - e).max(0) as usize;
        let s = format!("{:.*}", d, v);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}
