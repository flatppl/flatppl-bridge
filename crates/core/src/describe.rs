//! Describe records: what each member of a box supports.

use crate::error::{BridgeError, bad};
use crate::set::Set;
use crate::syntax::builtin_head;
use flatppl_bridge_abi::Op;
use flatppl_core::{Module, Node, NodeId, Scalar};
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

/// Parse and check the describe text of a box.
pub fn parse(text: &str) -> Result<BTreeMap<String, Member>, BridgeError> {
    let module = flatppl_syntax::parse(text).map_err(|e| bad(format!("describe text: {e}")))?;
    let mut members = BTreeMap::new();
    for (_, binding) in module.bindings() {
        let name = module.resolve(binding.name).to_owned();
        let member =
            member(&module, binding.rhs).map_err(|e| bad(format!("describe `{name}`: {e}")))?;
        members.insert(name, member);
    }
    Ok(members)
}

fn member(module: &Module, id: NodeId) -> Result<Member, String> {
    let Node::Call(call) = module.node(id) else {
        return Err("expected a record".into());
    };
    if builtin_head(module, id) != Some("record") {
        return Err("expected a record".into());
    }
    let field = |name: &str| {
        call.named
            .iter()
            .find(|a| module.resolve(a.name) == name)
            .map(|a| a.value)
    };
    let string = |id: NodeId| match module.node(id) {
        Node::Lit(Scalar::Str(s)) => Ok(s.to_string()),
        _ => Err("expected a string".to_string()),
    };
    let boolean = |name: &str| match field(name).map(|id| module.node(id)) {
        None => Ok(false),
        Some(Node::Lit(Scalar::Bool(b))) => Ok(*b),
        Some(_) => Err(format!("`{name}` must be a boolean")),
    };
    let set = |name: &str| {
        field(name)
            .map(|id| Set::from_node(module, id).map_err(|e| e.to_string()))
            .transpose()
    };
    let kind = match string(field("kind").ok_or("missing `kind`")?)?.as_str() {
        "function" => Kind::Function,
        "kernel" => Kind::Kernel,
        "measure" => Kind::Measure,
        other => return Err(format!("unknown kind `{other}`")),
    };
    let mut queries = HashSet::new();
    let list = field("queries").ok_or("missing `queries`")?;
    let items: Vec<NodeId> = match module.node(list) {
        Node::Call(v) if builtin_head(module, list) == Some("vector") => v.args.to_vec(),
        _ => vec![list],
    };
    for item in items {
        let name = string(item)?;
        let op = [
            Op::Apply,
            Op::LogDensity,
            Op::Sample,
            Op::ToUniform,
            Op::FromUniform,
            Op::ToNormal,
            Op::FromNormal,
        ]
        .into_iter()
        .find(|op| op.query_name() == name)
        .ok_or_else(|| format!("unknown query `{name}`"))?;
        queries.insert(op);
    }
    let member = Member {
        kind,
        gradient: boolean("gradient")?,
        vjp: boolean("vjp")?,
        input: set("input")?,
        output: set("output")?,
        variate: set("variate")?,
        normalized: boolean("normalized")?,
        queries,
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
