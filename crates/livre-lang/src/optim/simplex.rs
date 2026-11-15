//! Dense bounded dual simplex with warm start.
//!
//! minimize c'x subject to A x <= b, Aeq x = beq, lb <= x <= ub.
//!
//! The rows are written as [A; Aeq] x + s = [b; beq] with slacks s in [0, Inf) for the
//! inequalities and s = 0 for the equalities. Infinite bounds are replaced by an artificial box
//! (+-BIG). With every variable boxed, any basis is made dual feasible by moving each nonbasic
//! variable to the bound that matches the sign of its reduced cost, so the dual simplex needs no
//! phase 1. Leaving row: largest primal infeasibility. Entering column: dual ratio test, ties
//! broken by the largest pivot (and by the smallest index after a run of degenerate pivots).
//!
//! References: A. Koberstein (2005), "The dual simplex method, techniques for a fast and stable
//! implementation", PhD thesis, Paderborn; J. Nocedal and S. J. Wright, Numerical Optimization,
//! 2nd ed., Ch. 13. Clean room: no MathWorks code.

/// Basis of the slack form: `b` holds the basic columns (indices into [x; s]), `stat` is -1 at
/// the lower bound, +1 at the upper bound and 0 for basic columns.
#[derive(Clone, Debug)]
pub struct Basis {
    pub b: Vec<usize>,
    pub stat: Vec<i8>,
}

pub struct Options {
    pub maxit: usize,
    /// Primal feasibility tolerance, scaled by max(1, ||rhs||inf).
    pub tol: f64,
    pub basis: Option<Basis>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    IterationLimit,
    Infeasible,
    Unbounded,
    /// The basis matrix became numerically singular.
    Singular,
}

pub struct Solution {
    pub x: Vec<f64>,
    pub status: Status,
    pub basis: Basis,
    /// Row duals: inv(B)' c_B.
    pub y: Vec<f64>,
    /// Reduced costs of [x; s].
    pub d: Vec<f64>,
    pub iterations: usize,
    /// Values of [x; s], and the bounds used for them (infinite bounds replaced by +-BIG).
    pub xz: Vec<f64>,
    pub la: Vec<f64>,
    pub ua: Vec<f64>,
    /// Columns sitting at an artificial (+-BIG) bound.
    pub atart: Vec<bool>,
}

/// The constraint rows [A; Aeq] in column-major order, `rows` x `n`.
pub struct Rows<'a> {
    pub data: &'a [f64],
    pub rows: usize,
    /// Number of inequality rows (the first ones).
    pub ineq: usize,
}

/// Dense LU with partial pivoting (P A = L U), column-major, in place.
struct Lu {
    n: usize,
    a: Vec<f64>,
    piv: Vec<usize>,
}

impl Lu {
    fn factor(n: usize, mut a: Vec<f64>) -> Option<Lu> {
        let mut piv = vec![0; n];
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
        for k in 0..n {
            let mut p = k;
            let mut best = a[k * n + k].abs();
            for i in k + 1..n {
                let v = a[k * n + i].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if best <= scale * 1e-14 {
                return None;
            }
            piv[k] = p;
            if p != k {
                for j in 0..n {
                    a.swap(j * n + k, j * n + p);
                }
            }
            let akk = a[k * n + k];
            for i in k + 1..n {
                a[k * n + i] /= akk;
            }
            for j in k + 1..n {
                let akj = a[j * n + k];
                if akj != 0.0 {
                    for i in k + 1..n {
                        a[j * n + i] -= a[k * n + i] * akj;
                    }
                }
            }
        }
        Some(Lu { n, a, piv })
    }

    /// x = inv(A) b
    fn solve(&self, b: &mut [f64]) {
        let (n, a) = (self.n, &self.a);
        for k in 0..n {
            b.swap(k, self.piv[k]);
        }
        for k in 0..n {
            let bk = b[k];
            if bk != 0.0 {
                for i in k + 1..n {
                    b[i] -= a[k * n + i] * bk;
                }
            }
        }
        for k in (0..n).rev() {
            b[k] /= a[k * n + k];
            let bk = b[k];
            if bk != 0.0 {
                for i in 0..k {
                    b[i] -= a[k * n + i] * bk;
                }
            }
        }
    }

    /// y = inv(A)' c
    fn solve_t(&self, c: &mut [f64]) {
        let (n, a) = (self.n, &self.a);
        for k in 0..n {
            let mut s = c[k];
            for i in 0..k {
                s -= a[k * n + i] * c[i];
            }
            c[k] = s / a[k * n + k];
        }
        for k in (0..n).rev() {
            let mut s = c[k];
            for i in k + 1..n {
                s -= a[k * n + i] * c[i];
            }
            c[k] = s;
        }
        for k in (0..n).rev() {
            c.swap(k, self.piv[k]);
        }
    }
}

/// Solve the LP. `lb` and `ub` have one entry per variable and may hold infinities; `rhs` is
/// [b; beq].
pub fn solve(c: &[f64], rows: &Rows, rhs: &[f64], lb: &[f64], ub: &[f64], opts: Options) -> Solution {
    let n = c.len();
    let m = rows.rows;
    let nn = n + m;
    // Column j of [M, I] dotted with v, and v += a * column j.
    let dot = |j: usize, v: &[f64]| -> f64 {
        if j < n {
            let col = &rows.data[j * m..(j + 1) * m];
            col.iter().zip(v).map(|(a, b)| a * b).sum()
        } else {
            v[j - n]
        }
    };
    let axpy = |j: usize, a: f64, out: &mut [f64]| {
        if j < n {
            for (o, v) in out.iter_mut().zip(&rows.data[j * m..(j + 1) * m]) {
                *o += a * v;
            }
        } else {
            out[j - n] += a;
        }
    };

    let mut cz = c.to_vec();
    cz.resize(nn, 0.0);
    let mut lz = lb.to_vec();
    lz.resize(nn, 0.0);
    let mut uz = ub.to_vec();
    for i in 0..m {
        uz.push(if i < rows.ineq { f64::INFINITY } else { 0.0 });
    }
    let mut big = 1.0f64;
    for v in rows.data.iter().chain(rhs).chain(c) {
        big = big.max(v.abs());
    }
    for v in lz.iter().chain(&uz) {
        if v.is_finite() {
            big = big.max(v.abs());
        }
    }
    let big = 1e7 * big;
    let la: Vec<f64> = lz.iter().map(|&v| if v.is_finite() { v } else { -big }).collect();
    let ua: Vec<f64> = uz.iter().map(|&v| if v.is_finite() { v } else { big }).collect();
    let norm_inf = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let tolp = opts.tol * norm_inf(rhs).max(1.0);
    let told = 1e-9 * norm_inf(c).max(1.0);
    let tolpiv = 1e-9;

    let (mut basic, mut stat) = match opts.basis {
        Some(ws) if ws.stat.len() == nn && ws.b.len() == m => (ws.b, ws.stat),
        _ => {
            let mut stat = vec![-1i8; nn];
            for s in stat.iter_mut().skip(n) {
                *s = 0;
            }
            ((n..nn).collect(), stat)
        }
    };
    let mut isb = vec![false; nn];
    for &j in &basic {
        isb[j] = true;
    }

    let factor = |basic: &[usize]| -> Option<Lu> {
        let mut mb = vec![0.0; m * m];
        for (k, &j) in basic.iter().enumerate() {
            if j < n {
                mb[k * m..(k + 1) * m].copy_from_slice(&rows.data[j * m..(j + 1) * m]);
            } else {
                mb[k * m + (j - n)] = 1.0;
            }
        }
        Lu::factor(m, mb)
    };
    // Duals, reduced costs and the primal point for the current basis and bound statuses.
    let evaluate = |lu: &Lu, basic: &[usize], stat: &[i8], isb: &[bool], flip: bool| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<i8>) {
        let mut y: Vec<f64> = basic.iter().map(|&j| cz[j]).collect();
        lu.solve_t(&mut y);
        let mut d: Vec<f64> = (0..nn).map(|j| if isb[j] { 0.0 } else { cz[j] - dot(j, &y) }).collect();
        let mut stat = stat.to_vec();
        if flip {
            for j in 0..nn {
                if isb[j] {
                    continue;
                }
                if stat[j] == 1 && d[j] > told {
                    stat[j] = -1;
                } else if stat[j] == -1 && d[j] < -told {
                    stat[j] = 1;
                }
                if la[j] == ua[j] && stat[j] == 0 {
                    stat[j] = -1;
                }
            }
        }
        let mut xz = vec![0.0; nn];
        let mut r = rhs.to_vec();
        for j in 0..nn {
            if isb[j] {
                continue;
            }
            xz[j] = match stat[j] {
                -1 => la[j],
                1 => ua[j],
                _ => 0.0,
            };
            if xz[j] != 0.0 {
                axpy(j, -xz[j], &mut r);
            }
        }
        lu.solve(&mut r);
        for (k, &j) in basic.iter().enumerate() {
            xz[j] = r[k];
        }
        for j in basic {
            d[*j] = 0.0;
        }
        (y, d, xz, stat)
    };

    let mut it = 0usize;
    let mut status = Status::IterationLimit;
    let mut degen = 0usize;
    let mut lastobj = f64::NEG_INFINITY;
    while it < opts.maxit {
        it += 1;
        let Some(lu) = factor(&basic) else {
            status = Status::Singular;
            break;
        };
        let (_, d, xz, st) = evaluate(&lu, &basic, &stat, &isb, true);
        stat = st;
        // Leaving row: largest primal infeasibility (first index on ties).
        let mut r = usize::MAX;
        let mut vmax = f64::NEG_INFINITY;
        let mut below = false;
        for (k, &j) in basic.iter().enumerate() {
            let lo = la[j] - xz[j];
            let hi = xz[j] - ua[j];
            let v = lo.max(hi);
            if v > vmax {
                vmax = v;
                r = k;
                below = lo > hi;
            }
        }
        if r == usize::MAX || vmax <= tolp {
            status = Status::Optimal;
            break;
        }
        let mut rho = vec![0.0; m];
        rho[r] = 1.0;
        lu.solve_t(&mut rho);
        // Entering column: dual ratio test over the eligible nonbasic columns.
        let mut cand: Vec<(usize, f64, f64)> = Vec::new(); // (j, ratio, |alpha|)
        for j in 0..nn {
            if isb[j] || la[j] == ua[j] {
                continue;
            }
            let alpha = dot(j, &rho);
            let ok = if below {
                (stat[j] == -1 && alpha < -tolpiv) || (stat[j] == 1 && alpha > tolpiv)
            } else {
                (stat[j] == -1 && alpha > tolpiv) || (stat[j] == 1 && alpha < -tolpiv)
            };
            if ok {
                cand.push((j, d[j].abs() / alpha.abs(), alpha.abs()));
            }
        }
        if cand.is_empty() {
            status = Status::Infeasible; // dual unbounded
            break;
        }
        let rmin = cand.iter().fold(f64::INFINITY, |m, c| m.min(c.1));
        let lim = rmin + 1e-12 * rmin.max(1.0);
        let ties = cand.iter().filter(|c| c.1 <= lim);
        let q = if degen > 50 {
            ties.map(|c| c.0).min().unwrap()
        } else {
            let mut best = (usize::MAX, f64::NEG_INFINITY);
            for c in ties {
                if c.2 > best.1 {
                    best = (c.0, c.2);
                }
            }
            best.0
        };
        let obj: f64 = cz.iter().zip(&xz).map(|(a, b)| a * b).sum();
        if (obj - lastobj).abs() <= 1e-14 * obj.abs().max(1.0) {
            degen += 1;
        } else {
            degen = 0;
        }
        lastobj = obj;
        let leave = basic[r];
        stat[leave] = if below { -1 } else { 1 };
        isb[leave] = false;
        isb[q] = true;
        basic[r] = q;
        stat[q] = 0;
    }

    // Final values with the last basis.
    let (y, d, xz) = match factor(&basic) {
        Some(lu) => {
            let (y, d, xz, _) = evaluate(&lu, &basic, &stat, &isb, false);
            (y, d, xz)
        }
        None => {
            status = Status::Singular;
            (vec![f64::NAN; m], vec![f64::NAN; nn], vec![f64::NAN; nn])
        }
    };
    let atart: Vec<bool> =
        (0..nn).map(|j| (stat[j] == -1 && !lz[j].is_finite()) || (stat[j] == 1 && !uz[j].is_finite()) || (isb[j] && xz[j].abs() >= 0.5 * big)).collect();
    if status == Status::Optimal && (0..nn).any(|j| atart[j] && (d[j].abs() > told || isb[j])) {
        status = Status::Unbounded; // optimum of the boxed LP lies on the artificial box
    }
    Solution { x: xz[..n].to_vec(), status, basis: Basis { b: basic, stat }, y, d, iterations: it, xz, la, ua, atart }
}

/// Rows `which` (positions in the basis) of inv(B) [M, I], the simplex tableau: one entry per
/// column of [x; s]. None when the basis is singular.
pub fn tableau_rows(rows: &Rows, n: usize, basic: &[usize], which: &[usize]) -> Option<Vec<Vec<f64>>> {
    let m = rows.rows;
    let mut mb = vec![0.0; m * m];
    for (k, &j) in basic.iter().enumerate() {
        if j < n {
            mb[k * m..(k + 1) * m].copy_from_slice(&rows.data[j * m..(j + 1) * m]);
        } else {
            mb[k * m + (j - n)] = 1.0;
        }
    }
    let lu = Lu::factor(m, mb)?;
    Some(
        which
            .iter()
            .map(|&r| {
                let mut rho = vec![0.0; m];
                rho[r] = 1.0;
                lu.solve_t(&mut rho);
                (0..n + m)
                    .map(|j| if j < n { rows.data[j * m..(j + 1) * m].iter().zip(&rho).map(|(a, b)| a * b).sum() } else { rho[j - n] })
                    .collect()
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_lp() {
        // min -3x - 5y  s.t. x <= 4, 2y <= 12, 3x + 2y <= 18, x, y >= 0  ->  (2, 6), -36
        let a = [1.0, 0.0, 3.0, 0.0, 2.0, 2.0];
        let rows = Rows { data: &a, rows: 3, ineq: 3 };
        let s = solve(&[-3.0, -5.0], &rows, &[4.0, 12.0, 18.0], &[0.0, 0.0], &[f64::INFINITY; 2], Options { maxit: 100, tol: 1e-9, basis: None });
        assert_eq!(s.status, Status::Optimal);
        assert!((s.x[0] - 2.0).abs() < 1e-12 && (s.x[1] - 6.0).abs() < 1e-12, "{:?}", s.x);
    }

    #[test]
    fn lu_solves() {
        let a = vec![2.0, 4.0, -2.0, 1.0, -6.0, 7.0, 1.0, 0.0, 2.0]; // columns
        let lu = Lu::factor(3, a.clone()).unwrap();
        let mut b = vec![5.0, -2.0, 9.0];
        lu.solve(&mut b);
        let r: Vec<f64> = (0..3).map(|i| (0..3).map(|j| a[j * 3 + i] * b[j]).sum::<f64>()).collect();
        assert!(r.iter().zip([5.0, -2.0, 9.0]).all(|(x, y)| (x - y).abs() < 1e-12));
        let mut c = vec![1.0, 2.0, 3.0];
        lu.solve_t(&mut c);
        let r: Vec<f64> = (0..3).map(|j| (0..3).map(|i| a[j * 3 + i] * c[i]).sum::<f64>()).collect();
        assert!(r.iter().zip([1.0, 2.0, 3.0]).all(|(x, y)| (x - y).abs() < 1e-12));
    }
}
