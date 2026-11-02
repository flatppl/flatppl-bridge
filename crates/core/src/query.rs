//! The black-box query profile: parse query text and check its shape.

use crate::error::{BridgeError, bad, unsupported};
use crate::set::Set;
use crate::syntax::{builtin_head, literal_f64, literal_int, show};
use crate::value::{Value, flatten, nest};
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array};
use flatppl_bridge_abi::Op;
use flatppl_core::{CallHead, Module, NamedKind, Node, NodeId, RefNs, Scalar};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Where a `load_data` binding reads from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Part(String),
    Hash(String),
}

/// An RNG state expression.
#[derive(Clone, Debug)]
pub enum RngSpec {
    Seed(Vec<u8>),
    /// Bytes from a literal or a data binding.
    State(Operand),
    /// `broadcast(rnginit, seeds)`: one state per entry of `seeds`.
    Seeds(Operand),
    /// `broadcast(rngstate, states)`: one state per byte vector of `states`.
    States(Operand),
}

/// An argument of the result call.
#[derive(Clone, Debug)]
pub enum Operand {
    Binding(String),
    Literal(Value),
    Record(Vec<(String, Operand)>),
    Rng(Box<RngSpec>),
}

/// The result call of a query.
#[derive(Clone, Debug)]
pub struct ResultCall {
    pub op: Op,
    pub member: String,
    pub broadcast: bool,
    /// The arguments after the member. For `builtin_sample`, the RNG state comes first.
    pub args: Vec<Operand>,
    pub kwargs: Vec<(String, Operand)>,
}

/// A query in the black-box query profile.
#[derive(Clone, Debug)]
pub struct Query {
    pub box_name: String,
    pub data: BTreeMap<String, (Source, Set)>,
    pub literals: BTreeMap<String, Value>,
    pub rngs: BTreeMap<String, RngSpec>,
    pub result: ResultCall,
}

const TRANSPORTS: [(&str, Op); 4] = [
    ("builtin_touniform", Op::ToUniform),
    ("builtin_fromuniform", Op::FromUniform),
    ("builtin_tonormal", Op::ToNormal),
    ("builtin_fromnormal", Op::FromNormal),
];

fn density_op(name: &str) -> Option<Op> {
    if name == "builtin_logdensityof" {
        return Some(Op::LogDensity);
    }
    TRANSPORTS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, op)| *op)
}

/// Parse query text and check that it lies in the profile.
pub fn parse(text: &str) -> Result<Query, BridgeError> {
    let module = flatppl_syntax::parse(text).map_err(|e| bad(format!("query: {e}")))?;
    let mut alias = None;
    let mut box_name = None;
    let mut data = BTreeMap::new();
    let mut literals = BTreeMap::new();
    let mut rngs = BTreeMap::new();
    let mut result = None;
    for (_, binding) in module.bindings() {
        let name = module.resolve(binding.name).to_owned();
        let rhs = binding.rhs;
        if name == "result" {
            result = Some(rhs);
            continue;
        }
        if rng_head(&module, rhs).is_some() {
            rngs.insert(name, rng_spec(&module, rhs)?);
            continue;
        }
        match builtin_head(&module, rhs) {
            Some("load_module") => {
                let source = string_arg(&module, rhs, "load_module")?;
                let Some(boxed) = source.strip_prefix("box:") else {
                    return Err(unsupported(format!(
                        "`load_module(\"{source}\")`: a query loads only `box:<name>`"
                    )));
                };
                if box_name.is_some() {
                    return Err(unsupported("a query loads exactly one box"));
                }
                box_name = Some(boxed.to_owned());
                alias = Some(binding.name);
            }
            Some("load_data") => {
                let Node::Call(call) = module.node(rhs) else {
                    unreachable!()
                };
                let ([source, set], []) = (&call.args[..], &call.named[..]) else {
                    return Err(bad("`load_data` takes a source and a value set"));
                };
                let source = match module.node(*source) {
                    Node::Lit(Scalar::Str(s)) => parse_source(s)?,
                    _ => return Err(bad("the `load_data` source must be a string literal")),
                };
                data.insert(name, (source, Set::from_node(&module, *set)?));
            }
            _ => match literal_value(&module, rhs) {
                Some(value) => {
                    literals.insert(name, value);
                }
                None => {
                    return Err(unsupported(format!(
                        "binding `{name} = {}` is outside the black-box query profile",
                        show(&module, rhs)
                    )));
                }
            },
        }
    }
    let box_name =
        box_name.ok_or_else(|| bad("the query loads no box (`load_module(\"box:<name>\")`)"))?;
    let alias = alias.expect("set with the box");
    let result = result.ok_or_else(|| bad("the query has no `result` binding"))?;
    let result = result_call(&module, result, alias)?;
    Ok(Query {
        box_name,
        data,
        literals,
        rngs,
        result,
    })
}

fn parse_source(source: &str) -> Result<Source, BridgeError> {
    if let Some(name) = source.strip_prefix("part:") {
        if !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
        {
            return Ok(Source::Part(name.to_owned()));
        }
        return Err(bad(format!("bad part name in `{source}`")));
    }
    if let Some(hex) = source.strip_prefix("sha256:") {
        if is_hash(hex) {
            return Ok(Source::Hash(hex.to_owned()));
        }
        return Err(bad(format!(
            "bad hash in `{source}`: expected 64 lowercase hex digits"
        )));
    }
    Err(unsupported(format!(
        "`load_data(\"{source}\")`: a query reads only `part:` and `sha256:` sources"
    )))
}

pub fn is_hash(hex: &str) -> bool {
    hex.len() == 64
        && hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn string_arg(module: &Module, id: NodeId, head: &str) -> Result<String, BridgeError> {
    match module.node(id) {
        Node::Call(call) if call.args.len() == 1 && call.named.is_empty() => {
            match module.node(call.args[0]) {
                Node::Lit(Scalar::Str(s)) => Ok(s.to_string()),
                _ => Err(bad(format!("`{head}` takes a string literal"))),
            }
        }
        _ => Err(unsupported(format!(
            "`{}` is outside the profile",
            show(module, id)
        ))),
    }
}

/// `rnginit` or `rngstate`, called directly or through `broadcast` (then with a
/// trailing `.`).
fn rng_head(module: &Module, id: NodeId) -> Option<&'static str> {
    let Node::Call(call) = module.node(id) else {
        return None;
    };
    let name = |sym| match module.resolve(sym) {
        "rnginit" => Some("rnginit"),
        "rngstate" => Some("rngstate"),
        _ => None,
    };
    match builtin_head(module, id)? {
        "broadcast" => match call.args.first().map(|&c| module.node(c)) {
            Some(Node::Const(sym)) => name(*sym).map(|n| {
                if n == "rnginit" {
                    "rnginit."
                } else {
                    "rngstate."
                }
            }),
            _ => None,
        },
        _ => match call.head {
            CallHead::Builtin(head) => name(head),
            CallHead::User(_) => None,
        },
    }
}

fn rng_spec(module: &Module, id: NodeId) -> Result<RngSpec, BridgeError> {
    let Node::Call(call) = module.node(id) else {
        unreachable!()
    };
    let head = rng_head(module, id).expect("an RNG expression");
    let args = match head {
        "rnginit." | "rngstate." => &call.args[1..],
        _ => &call.args[..],
    };
    let [arg] = (args.len() == 1 && call.named.is_empty())
        .then(|| [args[0]])
        .ok_or_else(|| bad(format!("`{}` takes one argument", show(module, id))))?;
    match head {
        "rnginit" => {
            if let Some(n) = literal_int(module, arg) {
                let n = u64::try_from(n).map_err(|_| bad("an integer seed lies in 0..2^64-1"))?;
                return Ok(RngSpec::Seed(n.to_le_bytes().to_vec()));
            }
            let bytes = literal_value(module, arg).and_then(|v| bytes_of(&v));
            match bytes {
                Some(bytes) if !bytes.is_empty() => Ok(RngSpec::Seed(bytes)),
                _ => Err(bad(
                    "`rnginit` takes an integer or a non-empty byte-vector literal",
                )),
            }
        }
        "rnginit." => Ok(RngSpec::Seeds(operand(module, arg)?)),
        "rngstate." => Ok(RngSpec::States(operand(module, arg)?)),
        _ => Ok(RngSpec::State(operand(module, arg)?)),
    }
}

/// The bytes of an integer vector value, if every entry lies in 0..=255.
pub fn bytes_of(value: &Value) -> Option<Vec<u8>> {
    let Value::Array { shape, data } = value else {
        return None;
    };
    if shape.len() != 1 {
        return None;
    }
    let ints = data.as_any().downcast_ref::<Int64Array>()?;
    ints.values()
        .iter()
        .map(|&b| u8::try_from(b).ok())
        .collect()
}

fn result_call(
    module: &Module,
    id: NodeId,
    alias: flatppl_core::Symbol,
) -> Result<ResultCall, BridgeError> {
    let outside = || {
        unsupported(format!(
            "`result = {}` is outside the black-box query profile",
            show(module, id)
        ))
    };
    let member_of = |id: NodeId| match module.node(id) {
        Node::Ref(r) if r.ns == RefNs::Module(alias) => Some(module.resolve(r.name).to_owned()),
        _ => None,
    };
    let Node::Call(call) = module.node(id) else {
        return Err(outside());
    };
    let kwargs = || -> Result<Vec<(String, Operand)>, BridgeError> {
        call.named
            .iter()
            .map(|a| {
                if a.kind != NamedKind::Kwarg {
                    return Err(outside());
                }
                Ok((module.resolve(a.name).to_owned(), operand(module, a.value)?))
            })
            .collect()
    };
    let operands = |ids: &[NodeId]| {
        ids.iter()
            .map(|&a| operand(module, a))
            .collect::<Result<Vec<_>, _>>()
    };
    match call.head {
        CallHead::User(callee) => {
            let member = member_of(callee).ok_or_else(outside)?;
            Ok(ResultCall {
                op: Op::Apply,
                member,
                broadcast: false,
                args: operands(&call.args)?,
                kwargs: kwargs()?,
            })
        }
        CallHead::Builtin(head) => match module.resolve(head) {
            "builtin_sample" => {
                let [rng, member, rest @ ..] = &call.args[..] else {
                    return Err(outside());
                };
                if !call.named.is_empty() {
                    return Err(outside());
                }
                let member = member_of(*member).ok_or_else(outside)?;
                let mut args = vec![operand(module, *rng)?];
                args.extend(operands(rest)?);
                Ok(ResultCall {
                    op: Op::Sample,
                    member,
                    broadcast: false,
                    args,
                    kwargs: Vec::new(),
                })
            }
            "broadcast" => {
                let [callee, rest @ ..] = &call.args[..] else {
                    return Err(outside());
                };
                if let Some(member) = member_of(*callee) {
                    return Ok(ResultCall {
                        op: Op::Apply,
                        member,
                        broadcast: true,
                        args: operands(rest)?,
                        kwargs: kwargs()?,
                    });
                }
                let op = match module.node(*callee) {
                    Node::Const(sym) => density_op(module.resolve(*sym)),
                    _ => None,
                };
                if let Node::Const(sym) = module.node(*callee)
                    && module.resolve(*sym) == "builtin_sample"
                {
                    let ([rng, member, rest @ ..], true) = (rest, call.named.is_empty()) else {
                        return Err(outside());
                    };
                    let member = member_of(*member).ok_or_else(outside)?;
                    let mut args = vec![operand(module, *rng)?];
                    args.extend(operands(rest)?);
                    return Ok(ResultCall {
                        op: Op::Sample,
                        member,
                        broadcast: true,
                        args,
                        kwargs: Vec::new(),
                    });
                }
                let (Some(op), [member, rest @ ..], true) = (op, rest, call.named.is_empty())
                else {
                    return Err(outside());
                };
                let member = member_of(*member).ok_or_else(outside)?;
                Ok(ResultCall {
                    op,
                    member,
                    broadcast: true,
                    args: operands(rest)?,
                    kwargs: Vec::new(),
                })
            }
            name => {
                let op = density_op(name).ok_or_else(outside)?;
                let ([member, rest @ ..], true) = (&call.args[..], call.named.is_empty()) else {
                    return Err(outside());
                };
                let member = member_of(*member).ok_or_else(outside)?;
                Ok(ResultCall {
                    op,
                    member,
                    broadcast: false,
                    args: operands(rest)?,
                    kwargs: Vec::new(),
                })
            }
        },
    }
}

fn operand(module: &Module, id: NodeId) -> Result<Operand, BridgeError> {
    if let Some(value) = literal_value(module, id) {
        return Ok(Operand::Literal(value));
    }
    if rng_head(module, id).is_some() {
        return Ok(Operand::Rng(Box::new(rng_spec(module, id)?)));
    }
    match module.node(id) {
        Node::Ref(r) if r.ns == RefNs::SelfMod => {
            Ok(Operand::Binding(module.resolve(r.name).to_owned()))
        }
        // The parser reads an unbound name as a builtin constant.
        Node::Const(sym) => Err(bad(format!(
            "`{}` is not a binding of the query",
            module.resolve(*sym)
        ))),
        Node::Call(call) => match builtin_head(module, id) {
            Some("record") if call.args.is_empty() => call
                .named
                .iter()
                .map(|a| Ok((module.resolve(a.name).to_owned(), operand(module, a.value)?)))
                .collect::<Result<Vec<_>, _>>()
                .map(Operand::Record),
            _ => Err(unsupported(format!(
                "argument `{}` is outside the black-box query profile",
                show(module, id)
            ))),
        },
        _ => Err(unsupported(format!(
            "argument `{}` is outside the black-box query profile",
            show(module, id)
        ))),
    }
}

/// A literal scalar, an array literal of these or of array literals, or a
/// record literal of these.
pub fn literal_value(module: &Module, id: NodeId) -> Option<Value> {
    let scalar = |array: ArrayRef| Some(Value::scalar(array));
    match module.node(id) {
        Node::Lit(Scalar::Int(n)) => scalar(Arc::new(Int64Array::from(vec![*n]))),
        Node::Lit(Scalar::Real(x)) => scalar(Arc::new(Float64Array::from(vec![*x]))),
        Node::Lit(Scalar::Bool(b)) => scalar(Arc::new(BooleanArray::from(vec![*b]))),
        Node::Const(_) => {
            literal_f64(module, id).and_then(|x| scalar(Arc::new(Float64Array::from(vec![x]))))
        }
        Node::Call(call) => match builtin_head(module, id)? {
            "neg" => {
                if let Some(n) = literal_int(module, id) {
                    return scalar(Arc::new(Int64Array::from(vec![n])));
                }
                literal_f64(module, id).and_then(|x| scalar(Arc::new(Float64Array::from(vec![x]))))
            }
            "vector" => {
                let items: Vec<Value> = call
                    .args
                    .iter()
                    .map(|&a| literal_value(module, a))
                    .collect::<Option<_>>()?;
                stack(&items)
            }
            "record" if call.args.is_empty() => call
                .named
                .iter()
                .map(|a| {
                    Some((
                        module.resolve(a.name).to_owned(),
                        literal_value(module, a.value)?,
                    ))
                })
                .collect::<Option<Vec<_>>>()
                .map(Value::Record),
            _ => None,
        },
        _ => None,
    }
}

/// The array whose elements are `items`: equal-shape scalars or arrays.
/// An array of arrays is not a matrix (spec §03).
fn stack(items: &[Value]) -> Option<Value> {
    let mut dims = None;
    let mut ints = Vec::new();
    let mut reals = Vec::new();
    let mut bools = Vec::new();
    let mut all_int = true;
    let mut all_bool = true;
    for item in items {
        let Value::Array { shape, data } = item else {
            return None;
        };
        let (inner, data) = flatten(data);
        let full: Vec<usize> = shape.iter().copied().chain(inner).collect();
        if *dims.get_or_insert_with(|| full.clone()) != full {
            return None;
        }
        if let Some(a) = data.as_any().downcast_ref::<Int64Array>() {
            ints.extend(a.values().iter().copied());
            reals.extend(a.values().iter().map(|&n| n as f64));
            all_bool = false;
        } else if let Some(a) = data.as_any().downcast_ref::<Float64Array>() {
            reals.extend(a.values().iter().copied());
            all_int = false;
            all_bool = false;
        } else {
            let a = data.as_any().downcast_ref::<BooleanArray>()?;
            bools.extend(a.iter().map(|b| b.unwrap_or(false)));
            all_int = false;
        }
    }
    let data: ArrayRef = if all_bool {
        Arc::new(BooleanArray::from(bools))
    } else if all_int {
        Arc::new(Int64Array::from(ints))
    } else if bools.is_empty() {
        Arc::new(Float64Array::from(reals))
    } else {
        return None;
    };
    Some(Value::Array {
        shape: vec![items.len()],
        data: nest(data, &dims?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_profile_forms() {
        let q = parse(
            "bb = load_module(\"box:normal\")\n\
             theta = load_data(\"part:theta\", cartpow(reals, 2))\n\
             rng = rnginit(42)\n\
             result = builtin_logdensityof(bb.K, theta, [1.0, 2.0])\n",
        )
        .unwrap();
        assert_eq!(q.box_name, "normal");
        assert_eq!(q.result.op, Op::LogDensity);
        assert_eq!(q.result.args.len(), 2);
        assert_eq!(q.data["theta"].0, Source::Part("theta".into()));
        assert!(matches!(q.rngs["rng"], RngSpec::Seed(ref b) if b == &42u64.to_le_bytes()));
    }

    #[test]
    fn broadcast_and_apply() {
        let q =
            parse("bb = load_module(\"box:b\")\nresult = bb.f.(a = [1.0, 2.0], b = 3)\n").unwrap();
        assert!(q.result.broadcast);
        assert_eq!(q.result.kwargs.len(), 2);
        let q = parse(
            "bb = load_module(\"box:b\")\nresult = builtin_logdensityof.(bb.K, [1, 2], 0.5)\n",
        )
        .unwrap();
        assert!(q.result.broadcast && q.result.op == Op::LogDensity);
    }

    #[test]
    fn refusals() {
        let status = |text: &str| parse(text).unwrap_err().status();
        assert_eq!(status("result = 1.0\n"), 400);
        assert_eq!(status("bb = load_module(\"box:b\")\n"), 400);
        assert_eq!(status("bb = load_module(\"box:b\")\nresult = (\n"), 400);
        assert_eq!(
            status("bb = load_module(\"box:b\")\nresult = logdensityof(bb.M, 1.0)\n"),
            422
        );
        assert_eq!(
            status("bb = load_module(\"box:b\")\nx = 1.0 + 2.0\nresult = bb.f(x)\n"),
            422
        );
        assert_eq!(
            status("bb = load_module(\"m.flatppl\")\nresult = bb.f(1.0)\n"),
            422
        );
        assert_eq!(
            status(
                "bb = load_module(\"box:b\")\nx = load_data(\"x.csv\", reals)\nresult = bb.f(x)\n"
            ),
            422
        );
    }

    #[test]
    fn batched_sampling_and_rng_arrays() {
        let q = parse(
            "bb = load_module(\"box:b\")\n\
             rngs = rnginit.([1, 2])\n\
             result = broadcast(builtin_sample, rngs, bb.K, [0.0, 1.0], 3)\n",
        )
        .unwrap();
        assert!(q.result.broadcast && q.result.op == Op::Sample);
        assert!(matches!(q.rngs["rngs"], RngSpec::Seeds(_)));
        let q = parse(
            "bb = load_module(\"box:b\")\n\
             s = load_data(\"part:s\", cartpow(cartpow(integers, 4), 2))\n\
             result = broadcast(builtin_sample, broadcast(rngstate, s), bb.M)\n",
        )
        .unwrap();
        assert!(
            matches!(&q.result.args[0], Operand::Rng(spec) if matches!(**spec, RngSpec::States(_)))
        );
    }

    #[test]
    fn a_nested_literal_is_an_array_of_arrays() {
        let module = flatppl_syntax::parse("x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]\n").unwrap();
        let (_, binding) = module.bindings().next().unwrap();
        let value = literal_value(&module, binding.rhs).unwrap();
        assert_eq!(value.shape(), Some(&[3][..]));
        assert_eq!(value.set().to_string(), "cartpow(cartpow(reals, 2), 3)");
    }
}
