//! Value sets of query data and describe records (spec §03).

use crate::error::BridgeError;
use crate::syntax::{literal_f64, literal_int};
use flatppl_core::{CallHead, Module, NamedKind, Node, NodeId};
use std::fmt;

/// A value set the server can check. Sets outside this list are refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Set {
    Reals,
    PosReals,
    NonNegReals,
    UnitInterval,
    Integers,
    PosIntegers,
    NonNegIntegers,
    Booleans,
    Interval(f64, f64),
    Anything,
    /// `cartpow(set, [d1, ..., dk])`.
    Pow(Box<Set>, Vec<usize>),
    /// `cartprod(a = S1, ...)`.
    Record(Vec<(String, Set)>),
}

/// The Arrow element kind of a scalar set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    Real,
    Integer,
    Boolean,
}

impl Set {
    pub fn scalar(&self) -> Option<Scalar> {
        Some(match self {
            Set::Reals
            | Set::PosReals
            | Set::NonNegReals
            | Set::UnitInterval
            | Set::Interval(..) => Scalar::Real,
            Set::Integers | Set::PosIntegers | Set::NonNegIntegers => Scalar::Integer,
            Set::Booleans => Scalar::Boolean,
            _ => return None,
        })
    }

    /// Whether the real `x` lies in this scalar set.
    pub fn contains_real(&self, x: f64) -> bool {
        match self {
            // Reals admit NaN: a degenerate value travels as a value.
            Set::Reals => true,
            Set::PosReals => x > 0.0,
            Set::NonNegReals => x >= 0.0,
            Set::UnitInterval => (0.0..=1.0).contains(&x),
            Set::Interval(lo, hi) => *lo <= x && x <= *hi,
            _ => false,
        }
    }

    pub fn contains_int(&self, n: i64) -> bool {
        match self {
            Set::Integers => true,
            Set::PosIntegers => n > 0,
            Set::NonNegIntegers => n >= 0,
            _ => false,
        }
    }

    /// Evaluate a set expression of a parsed module.
    pub fn from_node(module: &Module, id: NodeId) -> Result<Set, BridgeError> {
        let unsupported =
            || BridgeError::Unsupported(format!("unsupported value set `{}`", show(module, id)));
        match module.node(id) {
            Node::Const(sym) => Ok(match module.resolve(*sym) {
                "reals" => Set::Reals,
                "posreals" => Set::PosReals,
                "nonnegreals" => Set::NonNegReals,
                "unitinterval" => Set::UnitInterval,
                "integers" => Set::Integers,
                "posintegers" => Set::PosIntegers,
                "nonnegintegers" => Set::NonNegIntegers,
                "booleans" => Set::Booleans,
                "anything" => Set::Anything,
                _ => return Err(unsupported()),
            }),
            Node::Call(call) => {
                let CallHead::Builtin(head) = call.head else {
                    return Err(unsupported());
                };
                match (module.resolve(head), &call.args[..], &call.named[..]) {
                    ("cartpow", [elem, size], []) => {
                        let dims = match module.node(*size) {
                            Node::Call(v) if matches!(v.head, CallHead::Builtin(h) if module.resolve(h) == "vector") => {
                                v.args
                                    .iter()
                                    .map(|&d| dim(module, d))
                                    .collect::<Result<Vec<_>, _>>()?
                            }
                            _ => vec![dim(module, *size)?],
                        };
                        // cartpow(cartpow(S, a), b) is an array of arrays, not a
                        // matrix (spec §03).
                        Ok(Set::Pow(Box::new(Set::from_node(module, *elem)?), dims))
                    }
                    ("cartprod", [], named) if !named.is_empty() => named
                        .iter()
                        .map(|arg| {
                            if arg.kind == NamedKind::Field || arg.kind == NamedKind::Kwarg {
                                Ok((
                                    module.resolve(arg.name).to_owned(),
                                    Set::from_node(module, arg.value)?,
                                ))
                            } else {
                                Err(unsupported())
                            }
                        })
                        .collect::<Result<_, _>>()
                        .map(Set::Record),
                    ("interval", [lo, hi], []) => {
                        let (lo, hi) = (literal_f64(module, *lo), literal_f64(module, *hi));
                        match (lo, hi) {
                            (Some(lo), Some(hi)) if lo <= hi => Ok(Set::Interval(lo, hi)),
                            _ => Err(BridgeError::BadRequest(format!(
                                "`{}` needs literal bounds lo <= hi",
                                show(module, id)
                            ))),
                        }
                    }
                    _ => Err(unsupported()),
                }
            }
            _ => Err(unsupported()),
        }
    }
}

fn dim(module: &Module, id: NodeId) -> Result<usize, BridgeError> {
    literal_int(module, id)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            BridgeError::BadRequest(format!(
                "`{}` is not a positive integer literal",
                show(module, id)
            ))
        })
}

fn show(module: &Module, id: NodeId) -> String {
    crate::syntax::show(module, id)
}

impl fmt::Display for Set {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Set::Reals => f.write_str("reals"),
            Set::PosReals => f.write_str("posreals"),
            Set::NonNegReals => f.write_str("nonnegreals"),
            Set::UnitInterval => f.write_str("unitinterval"),
            Set::Integers => f.write_str("integers"),
            Set::PosIntegers => f.write_str("posintegers"),
            Set::NonNegIntegers => f.write_str("nonnegintegers"),
            Set::Booleans => f.write_str("booleans"),
            Set::Anything => f.write_str("anything"),
            Set::Interval(lo, hi) => write!(f, "interval({}, {})", real(*lo), real(*hi)),
            Set::Pow(elem, dims) if dims.len() == 1 => write!(f, "cartpow({elem}, {})", dims[0]),
            Set::Pow(elem, dims) => {
                let dims: Vec<String> = dims.iter().map(ToString::to_string).collect();
                write!(f, "cartpow({elem}, [{}])", dims.join(", "))
            }
            Set::Record(fields) => {
                let fields: Vec<String> =
                    fields.iter().map(|(n, s)| format!("{n} = {s}")).collect();
                write!(f, "cartprod({})", fields.join(", "))
            }
        }
    }
}

fn real(x: f64) -> String {
    if x == f64::INFINITY {
        "inf".into()
    } else if x == f64::NEG_INFINITY {
        "-inf".into()
    } else {
        format!("{x:?}")
    }
}
