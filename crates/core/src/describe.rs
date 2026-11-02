//! Describe JSON: what each member of a box supports.

use crate::error::{BridgeError, bad};
use crate::set::Set;
use flatppl_bridge_abi::Op;
use serde_json::Value as Json;
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Function,
    Kernel,
    Measure,
}

/// One member of a box.
#[derive(Clone, Debug)]
pub struct Member {
    pub kind: Kind,
    pub queries: HashSet<Op>,
    pub gradient: bool,
    pub vjp: bool,
    pub input: Option<Set>,
    pub output: Option<Set>,
    pub variate: Option<Set>,
    pub normalized: bool,
}

/// Parse and check the describe JSON of a box.
pub fn parse(text: &str) -> Result<BTreeMap<String, Member>, BridgeError> {
    let json: Json = serde_json::from_str(text).map_err(|e| bad(format!("describe JSON: {e}")))?;
    let Json::Object(map) = json else {
        return Err(bad("describe JSON: expected an object of members"));
    };
    let mut members = BTreeMap::new();
    for (name, value) in map {
        let valid = name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !valid {
            return Err(bad(format!("describe: bad member name `{name}`")));
        }
        let member = member(value).map_err(|e| bad(format!("describe `{name}`: {e}")))?;
        members.insert(name, member);
    }
    Ok(members)
}

/// The canonical describe JSON of parsed members.
pub fn to_json(members: &BTreeMap<String, Member>) -> String {
    let map: serde_json::Map<String, Json> = members
        .iter()
        .map(|(name, m)| {
            let mut object = serde_json::Map::new();
            let kind = match m.kind {
                Kind::Function => "function",
                Kind::Kernel => "kernel",
                Kind::Measure => "measure",
            };
            object.insert("kind".into(), kind.into());
            let mut queries: Vec<Op> = m.queries.iter().copied().collect();
            queries.sort_by_key(|op| op.code());
            object.insert(
                "queries".into(),
                queries.iter().map(|op| op.query_name()).collect(),
            );
            object.insert("gradient".into(), m.gradient.into());
            object.insert("vjp".into(), m.vjp.into());
            for (key, set) in [
                ("input", &m.input),
                ("output", &m.output),
                ("variate", &m.variate),
            ] {
                if let Some(set) = set {
                    object.insert(key.into(), set.to_string().into());
                }
            }
            if m.kind != Kind::Function {
                object.insert("normalized".into(), m.normalized.into());
            }
            (name.clone(), Json::Object(object))
        })
        .collect();
    Json::Object(map).to_string()
}

const OPS: [Op; 7] = [
    Op::Apply,
    Op::LogDensity,
    Op::Sample,
    Op::ToUniform,
    Op::FromUniform,
    Op::ToNormal,
    Op::FromNormal,
];

fn member(value: Json) -> Result<Member, String> {
    let Json::Object(map) = value else {
        return Err("expected a member object".into());
    };
    let mut kind = None;
    let mut queries = None;
    let (mut gradient, mut vjp, mut normalized) = (false, false, false);
    let (mut input, mut output, mut variate) = (None, None, None);
    for (key, value) in map {
        let boolean = |v: &Json| v.as_bool().ok_or(format!("`{key}` must be a boolean"));
        let set = |v: &Json| -> Result<Option<Set>, String> {
            let text = v.as_str().ok_or(format!("`{key}` must be a string"))?;
            set_expression(text)
                .map(Some)
                .map_err(|e| format!("`{key}`: {e}"))
        };
        match key.as_str() {
            "kind" => {
                kind = Some(match value.as_str() {
                    Some("function") => Kind::Function,
                    Some("kernel") => Kind::Kernel,
                    Some("measure") => Kind::Measure,
                    _ => return Err(format!("unknown kind {value}")),
                })
            }
            "queries" => {
                let list = value
                    .as_array()
                    .ok_or("`queries` must be an array of strings")?;
                let mut ops = HashSet::new();
                for item in list {
                    let name = item
                        .as_str()
                        .ok_or("`queries` must be an array of strings")?;
                    let op = OPS
                        .into_iter()
                        .find(|op| op.query_name() == name)
                        .ok_or_else(|| format!("unknown query `{name}`"))?;
                    ops.insert(op);
                }
                queries = Some(ops);
            }
            "gradient" => gradient = boolean(&value)?,
            "vjp" => vjp = boolean(&value)?,
            "normalized" => normalized = boolean(&value)?,
            "input" => input = set(&value)?,
            "output" => output = set(&value)?,
            "variate" => variate = set(&value)?,
            other => return Err(format!("unknown key `{other}`")),
        }
    }
    let kind = kind.ok_or("missing `kind`")?;
    let member = Member {
        kind,
        queries: queries.ok_or("missing `queries`")?,
        gradient,
        vjp,
        input,
        output,
        variate,
        normalized,
    };
    let apply = member.queries.contains(&Op::Apply);
    if apply != (kind == Kind::Function) || (kind == Kind::Function && member.queries.len() > 1) {
        return Err("only functions answer `apply`, and functions answer nothing else".into());
    }
    if member.queries.contains(&Op::Sample) && !member.normalized {
        return Err("`builtin_sample` needs a normalized measure".into());
    }
    Ok(member)
}

/// Parse a FlatPPL set expression with the FlatPPL parser.
fn set_expression(text: &str) -> Result<Set, String> {
    let module = flatppl_syntax::parse(&format!("s = {text}\n")).map_err(|e| e.to_string())?;
    let mut bindings = module.bindings();
    match (bindings.next(), bindings.next()) {
        (Some((_, binding)), None) => {
            Set::from_node(&module, binding.rhs).map_err(|e| e.to_string())
        }
        _ => Err(format!("`{text}` is not one set expression")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_prints_canonical_json() {
        let text = r#"{"K": {"kind": "kernel", "queries": ["builtin_sample", "builtin_logdensityof"], "gradient": true, "input": "cartprod(mu = reals, sigma = posreals)", "variate": "cartpow(cartpow(reals, 3), 2)", "normalized": true}}"#;
        let members = parse(text).unwrap();
        let k = &members["K"];
        assert_eq!(k.kind, Kind::Kernel);
        assert!(k.gradient && !k.vjp && k.normalized);
        assert_eq!(
            k.variate,
            Some(Set::Pow(
                Box::new(Set::Pow(Box::new(Set::Reals), vec![3])),
                vec![2]
            ))
        );
        let json: Json = serde_json::from_str(&to_json(&members)).unwrap();
        assert_eq!(
            json["K"]["queries"],
            serde_json::json!(["builtin_logdensityof", "builtin_sample"])
        );
        assert_eq!(json["K"]["variate"], "cartpow(cartpow(reals, 3), 2)");
        assert_eq!(parse(&to_json(&members)).unwrap()["K"].input, k.input);
    }

    #[test]
    fn refuses_bad_members() {
        let refused = |member: &str| parse(&format!(r#"{{"m": {member}}}"#)).is_err();
        assert!(refused(
            r#"{"kind": "function", "queries": ["apply"], "input": "reals\nx = 1"}"#
        ));
        assert!(refused(
            r#"{"kind": "function", "queries": ["apply"], "input": "foo(1)"}"#
        ));
        assert!(refused(
            r#"{"kind": "function", "queries": ["apply"], "colour": "red"}"#
        ));
        assert!(refused(
            r#"{"kind": "measure", "queries": ["builtin_sample"]}"#
        ));
        assert!(refused(r#"{"kind": "measure", "queries": ["apply"]}"#));
        assert!(refused(
            r#"{"kind": "measure", "queries": "builtin_logdensityof"}"#
        ));
        assert!(parse("m = record(kind = \"measure\")").is_err());
        assert!(!refused(
            r#"{"kind": "function", "queries": ["apply"], "input": "anything"}"#
        ));
    }
}
