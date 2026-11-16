//! Optimization Toolbox solvers, ported from the MATLAB-language draft at
//! github.com/pedrohenrique-dias8/matlivre (branch `optim-toolbox`, folder `toolbox/optim`).
//!
//! Each solver has its own module; shared numerics live in `simplex.rs`. Clean room: the
//! algorithms follow the published references cited in each module and the interfaces follow
//! MATLAB's documentation. No MathWorks code.
//!
//! `optimoptions` returns a structure (field `SolverName` plus the options given) rather than an
//! options object; the solvers read option names in either the optimoptions style
//! (`MaxIterations`) or the optimset style (`MaxIter`).

pub mod linprog;
pub mod simplex;

use crate::builtins::{one, Ret};
use crate::builtins_funfun::opt;
use crate::error::{rt, rt_id, LangError};
use crate::interp::Interpreter;
use crate::value::*;

type R<T> = Result<T, LangError>;

pub const NAMES: &[&str] = &["linprog", "optimoptions"];

pub fn call(it: &mut Interpreter, name: &str, a: &[Value], nargout: usize) -> Ret {
    match name {
        "linprog" => linprog::linprog(it, a, nargout),
        "optimoptions" => optimoptions(it, a),
        _ => Ok(None),
    }
}

/// The options argument of a solver: a structure from optimoptions or optimset, or [].
pub(crate) fn options_arg(v: &Value) -> R<Option<&StructArray>> {
    match v {
        Value::Struct(s) => Ok(Some(s)),
        Value::Num(m) if m.is_empty() => Ok(None),
        _ => rt("OPTIONS must be a structure created with OPTIMOPTIONS or OPTIMSET."),
    }
}

/// The first non-empty option among `names` (case-insensitive).
pub(crate) fn getopt<'a>(o: Option<&'a StructArray>, names: &[&str]) -> Option<&'a Value> {
    names.iter().find_map(|n| opt(o, n))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Display {
    Off,
    Final,
    Iter,
    /// Only when the solver fails.
    Notify,
}

pub(crate) fn display_level(o: Option<&StructArray>, default: Display) -> R<Display> {
    let Some(v) = getopt(o, &["Display"]) else { return Ok(default) };
    Ok(match v.text()?.to_lowercase().as_str() {
        "off" | "none" => Display::Off,
        "iter" | "iter-detailed" => Display::Iter,
        "notify" | "notify-detailed" => Display::Notify,
        _ => Display::Final,
    })
}

/// optimoptions(solver, Name, Value, ...), optimoptions(oldoptions, Name, Value, ...),
/// optimoptions(solver, oldoptions).
fn optimoptions(it: &mut Interpreter, a: &[Value]) -> Ret {
    let Some(first) = a.first() else { return rt("Not enough input arguments.") };
    let mut o = match first {
        Value::Struct(s) => s.clone(),
        Value::Func(_) => {
            let mut s = StructArray::scalar();
            let name = crate::builtins_funfun::func_text(it, first);
            s.set("SolverName", Value::chars(name.trim_start_matches('@')));
            s
        }
        v if v.is_text() => {
            let mut s = StructArray::scalar();
            s.set("SolverName", Value::chars(&v.text()?));
            s
        }
        _ => return rt("The first argument of OPTIMOPTIONS must be a solver name, a function handle or an options structure."),
    };
    let mut rest = &a[1..];
    // optimoptions(solver, oldoptions): copy the old options under the new solver name.
    if let (false, Some(Value::Struct(old))) = (matches!(first, Value::Struct(_)), rest.first()) {
        for (k, name) in old.fields.iter().enumerate() {
            if name != "SolverName" {
                o.set(name, old.elems[0][k].clone());
            }
        }
        rest = &rest[1..];
    }
    let solver = match o.get("SolverName") {
        Some(v) => v.text()?,
        None => String::new(),
    };
    for pair in rest.chunks(2) {
        if !pair[0].is_text() {
            return rt("Option names must be character vectors or strings.");
        }
        let name = pair[0].text()?;
        if pair.len() < 2 {
            return rt_id("optimlib:optimoptions:ParameterMissingValue", format!("Option '{name}' does not have a value."));
        }
        // option names are matched case-insensitively to the solver's list, when it is known
        let canonical = match known_options(&solver) {
            Some(list) => match list.iter().find(|n| n.eq_ignore_ascii_case(&name)) {
                Some(n) => n.to_string(),
                None => {
                    let s = solver.to_uppercase();
                    return rt_id(
                        "optimlib:optimoptions:UnmatchedParameter",
                        format!("'{name}' is not an option for {s}.\nA list of options can be found on the {s} <a href = \"matlab: helpview('optim','{solver}_opts');\">documentation page</a>."),
                    );
                }
            },
            None => name,
        };
        o.set(&canonical, pair[1].clone());
    }
    one(Value::Struct(o))
}

/// The options optimoptions accepts for a solver, with the optimset-style names it also maps
/// (None for solvers whose list is not recorded yet: any name is then accepted).
fn known_options(solver: &str) -> Option<&'static [&'static str]> {
    match solver {
        // linprog: the options of its algorithms, and the legacy names
        "linprog" => Some(&[
            "Algorithm",
            "ConstraintTolerance",
            "Display",
            "MaxIterations",
            "MaxTime",
            "OptimalityTolerance",
            "Preprocess",
            "MaxIter",
            "TolCon",
            "TolFun",
        ]),
        _ => None,
    }
}
