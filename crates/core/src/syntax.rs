//! Small helpers over parsed FlatPPL nodes.

use flatppl_core::{CallHead, Module, Node, NodeId, RefNs, Scalar};

/// The builtin head of a call node.
pub fn builtin_head(module: &Module, id: NodeId) -> Option<&str> {
    match module.node(id) {
        Node::Call(call) => match call.head {
            CallHead::Builtin(head) => Some(module.resolve(head)),
            CallHead::User(_) => None,
        },
        _ => None,
    }
}

/// A real literal: a number, `inf`, or their negation.
pub fn literal_f64(module: &Module, id: NodeId) -> Option<f64> {
    match module.node(id) {
        Node::Lit(Scalar::Real(x)) => Some(*x),
        Node::Lit(Scalar::Int(n)) => Some(*n as f64),
        Node::Const(sym) if module.resolve(*sym) == "inf" => Some(f64::INFINITY),
        Node::Call(call) if builtin_head(module, id) == Some("neg") && call.args.len() == 1 => {
            literal_f64(module, call.args[0]).map(|x| -x)
        }
        _ => None,
    }
}

/// An integer literal, possibly negated.
pub fn literal_int(module: &Module, id: NodeId) -> Option<i64> {
    match module.node(id) {
        Node::Lit(Scalar::Int(n)) => Some(*n),
        Node::Call(call) if builtin_head(module, id) == Some("neg") && call.args.len() == 1 => {
            literal_int(module, call.args[0]).and_then(i64::checked_neg)
        }
        _ => None,
    }
}

/// A short rendering of an expression for error messages.
pub fn show(module: &Module, id: NodeId) -> String {
    match module.node(id) {
        Node::Lit(Scalar::Int(n)) => n.to_string(),
        Node::Lit(Scalar::Real(x)) => format!("{x:?}"),
        Node::Lit(Scalar::Bool(b)) => b.to_string(),
        Node::Lit(Scalar::Str(s)) => format!("{s:?}"),
        Node::Const(sym) => module.resolve(*sym).to_owned(),
        Node::Ref(r) => match r.ns {
            RefNs::Module(alias) => format!("{}.{}", module.resolve(alias), module.resolve(r.name)),
            _ => module.resolve(r.name).to_owned(),
        },
        Node::Hole => "_".into(),
        Node::Axis(_) => "axis".into(),
        Node::Call(call) => {
            let head = match call.head {
                CallHead::Builtin(head) => module.resolve(head).to_owned(),
                CallHead::User(callee) => show(module, callee),
            };
            let mut args: Vec<String> = call.args.iter().map(|&a| show(module, a)).collect();
            args.extend(
                call.named.iter().map(|arg| {
                    format!("{} = {}", module.resolve(arg.name), show(module, arg.value))
                }),
            );
            format!("{head}({})", args.join(", "))
        }
    }
}
