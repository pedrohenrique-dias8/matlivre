//! Cutting planes for `intlinprog`, separated at an LP point x*: complemented mixed-integer
//! rounding (c-MIR) cuts from single rows, and the selection of the cuts to add in a round. Every
//! cut is returned as a row a'x <= b. (Extended knapsack cover cuts were tried and dropped: on the
//! benchmark set they increased the node count of multidimensional knapsacks.)
//!
//! References: H. Marchand and L. A. Wolsey (2001),
//! "Aggregation and mixed integer rounding to solve MIPs", Oper. Res. 49 (c-MIR); T. Achterberg
//! (2007), "Constraint Integer Programming", PhD thesis, TU Berlin, Ch. 8 (cut selection).
//! Clean room: no MathWorks code.

pub(super) type Cut = (Vec<f64>, f64);

fn dot(a: &[f64], x: &[f64]) -> f64 {
    a.iter().zip(x).map(|(a, b)| a * b).sum()
}

/// Violation of a'x <= b at x divided by ||a||: the distance of x from the cut.
pub(super) fn efficacy(cut: &Cut, x: &[f64]) -> f64 {
    let norm = cut.0.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        return 0.0;
    }
    (dot(&cut.0, x) - cut.1) / norm
}

/// The rows as a'x <= b: inequalities, then both directions of each equality.
pub(super) fn le_rows<'a>(a: &'a [Vec<f64>], b: &'a [f64], aeq: &'a [Vec<f64>], beq: &'a [f64]) -> Vec<(Vec<f64>, f64)> {
    let mut out: Vec<(Vec<f64>, f64)> = a.iter().cloned().zip(b.iter().copied()).collect();
    for (r, v) in aeq.iter().zip(beq) {
        out.push((r.clone(), *v));
        out.push((r.iter().map(|x| -x).collect(), -v));
    }
    out
}

/// c-MIR cuts from single rows a'x <= b. Each variable is moved to its closest finite bound
/// (y = x - l or y = u - x, y >= 0); rows with a free variable are skipped. The MIR formula is
/// tried with the divisors |g_j| of the integer variables strictly inside their bounds (and 1),
/// then with halves of the best one; the most efficacious violated cut of the row is kept.
pub(super) fn mir_cuts(rows: &[(Vec<f64>, f64)], isint: &[bool], lb: &[f64], ub: &[f64], x: &[f64]) -> Vec<Cut> {
    let n = x.len();
    let mut cuts = Vec::new();
    'rows: for (row, b) in rows {
        // transformed row: sum g_j y_j <= beta, with (j, g, y*, at_upper)
        let mut beta = *b;
        let mut terms: Vec<(usize, f64, f64, bool)> = Vec::new();
        let mut has_int = false;
        for j in 0..n {
            let a = row[j];
            if a == 0.0 {
                continue;
            }
            if lb[j] == ub[j] {
                beta -= a * lb[j];
                continue;
            }
            let upper = match (lb[j].is_finite(), ub[j].is_finite()) {
                (false, false) => continue 'rows,
                (true, false) => false,
                (false, true) => true,
                (true, true) => x[j] - lb[j] > ub[j] - x[j],
            };
            if upper {
                beta -= a * ub[j];
                terms.push((j, -a, ub[j] - x[j], true));
            } else {
                beta -= a * lb[j];
                terms.push((j, a, x[j] - lb[j], false));
            }
            has_int |= isint[j];
        }
        if !has_int {
            continue;
        }
        let mut deltas: Vec<f64> = terms.iter().filter(|t| isint[t.0] && t.2 > 1e-6 && t.2 < ub[t.0] - lb[t.0] - 1e-6).map(|t| t.1.abs()).filter(|d| *d > 1e-6).collect();
        deltas.push(1.0);
        deltas.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
        deltas.dedup_by(|p, q| (*p - *q).abs() <= 1e-9 * q.abs().max(1.0));
        deltas.truncate(8);
        let build = |delta: f64| -> Option<(Cut, f64)> {
            let bd = beta / delta;
            let f0 = bd - bd.floor();
            if !(0.01..=0.99).contains(&f0) {
                return None;
            }
            let mut a = vec![0.0; n];
            let mut rhs = bd.floor();
            for &(j, g, _, upper) in &terms {
                let coef = if isint[j] {
                    let gd = g / delta;
                    let fj = gd - gd.floor();
                    gd.floor() + (fj - f0).max(0.0) / (1.0 - f0)
                } else {
                    g.min(0.0) / (delta * (1.0 - f0))
                };
                if coef == 0.0 {
                    continue;
                }
                // back to x: y = x - l, or y = u - x
                if upper {
                    a[j] -= coef;
                    rhs -= coef * ub[j];
                } else {
                    a[j] += coef;
                    rhs += coef * lb[j];
                }
            }
            let cut = (a, rhs);
            let e = efficacy(&cut, x);
            (e > 1e-6).then_some((cut, e))
        };
        let mut best: Option<(Cut, f64, f64)> = None;
        for &d in &deltas {
            if let Some((c, e)) = build(d) {
                if best.as_ref().is_none_or(|b| e > b.1) {
                    best = Some((c, e, d));
                }
            }
        }
        if let Some((_, _, d0)) = best {
            for k in [2.0, 4.0, 8.0] {
                if let Some((c, e)) = build(d0 / k) {
                    if best.as_ref().is_none_or(|b| e > b.1) {
                        best = Some((c, e, d0 / k));
                    }
                }
            }
        }
        if let Some((cut, _, _)) = best {
            // numerically safe: bounded dynamism
            let nz: Vec<f64> = cut.0.iter().filter(|v| **v != 0.0).map(|v| v.abs()).collect();
            let (lo, hi) = nz.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            if !nz.is_empty() && hi / lo <= 1e6 {
                cuts.push(cut);
            }
        }
    }
    cuts
}

/// The cuts to add in a round: violated by at least 1e-4 (efficacy), sorted by efficacy, skipping
/// any cut nearly parallel (cosine > 0.98) to one already chosen or already in `existing`.
pub(super) fn select(cands: Vec<Cut>, x: &[f64], existing: &[Vec<f64>], max: usize) -> Vec<Cut> {
    let mut scored: Vec<(f64, Cut)> = cands.into_iter().map(|c| (efficacy(&c, x), c)).filter(|(e, _)| *e > 1e-4).collect();
    scored.sort_by(|p, q| q.0.partial_cmp(&p.0).unwrap_or(std::cmp::Ordering::Equal));
    let norm = |a: &[f64]| a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let parallel = |p: &[f64], q: &[f64]| {
        let (np, nq) = (norm(p), norm(q));
        np > 0.0 && nq > 0.0 && (dot(p, q) / (np * nq)) > 0.98
    };
    let mut out: Vec<Cut> = Vec::new();
    for (_, c) in scored {
        if out.len() >= max {
            break;
        }
        if out.iter().any(|o| parallel(&o.0, &c.0)) || existing.iter().any(|e| parallel(e, &c.0)) {
            continue;
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mir_cut_is_valid_and_violated() {
        // x1 + x2 <= 1.5 with integers in [0, 5]: MIR gives x1 + x2 <= 1
        let rows = vec![(vec![1.0, 1.0], 1.5)];
        let cuts = mir_cuts(&rows, &[true, true], &[0.0; 2], &[5.0; 2], &[0.75, 0.75]);
        assert_eq!(cuts.len(), 1);
        let (a, b) = &cuts[0];
        assert!((a[0] - 1.0).abs() < 1e-12 && (a[1] - 1.0).abs() < 1e-12 && (b - 1.0).abs() < 1e-12, "{a:?} {b}");
    }
}
