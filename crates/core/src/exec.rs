//! Run a checked query against a plugin: build the decoded call, batch a
//! broadcast, and assemble the reply values and derivatives.

use crate::describe::{Kind, Member};
use crate::error::{BridgeError, bad, unsupported};
use crate::query::{Operand, Query, RngSpec, bytes_of};
use crate::value::{Value, flatten, from_elements, nest, struct_array};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, Int64Array, UInt64Array};
use arrow_schema::DataType;
use flatppl_bridge_abi::{Call, Derivative, Op, Plugin, Rng};
use serde_json::Value as Json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Request options (the `options` part).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    pub derivative: Option<DerivativeOption>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DerivativeOption {
    pub vjp: bool,
    pub wrt: Vec<String>,
    pub cotangent: Option<String>,
}

impl Options {
    pub fn from_json(text: &[u8]) -> Result<Options, BridgeError> {
        let json: Json = serde_json::from_slice(text).map_err(|e| bad(format!("options: {e}")))?;
        let Json::Object(map) = json else {
            return Err(bad("options must be a JSON object"));
        };
        let mut options = Options::default();
        for (key, value) in map {
            match key.as_str() {
                "derivative" => options.derivative = Some(derivative_option(value)?),
                other => return Err(bad(format!("unknown option `{other}`"))),
            }
        }
        Ok(options)
    }

    /// The canonical JSON form, for cache keys.
    pub fn canonical(&self) -> String {
        match &self.derivative {
            None => "{}".into(),
            Some(d) => serde_json::json!({
                "derivative": {
                    "kind": if d.vjp { "vjp" } else { "gradient" },
                    "wrt": d.wrt,
                    "cotangent": d.cotangent,
                }
            })
            .to_string(),
        }
    }
}

fn derivative_option(value: Json) -> Result<DerivativeOption, BridgeError> {
    let Json::Object(map) = value else {
        return Err(bad("`derivative` must be an object"));
    };
    let mut kind = None;
    let mut wrt = None;
    let mut cotangent = None;
    for (key, value) in map {
        match (key.as_str(), value) {
            ("kind", Json::String(k)) if k == "gradient" || k == "vjp" => kind = Some(k == "vjp"),
            ("wrt", Json::Array(names)) => {
                wrt = Some(
                    names
                        .into_iter()
                        .map(|n| match n {
                            Json::String(s) => Ok(s),
                            _ => Err(bad("`wrt` lists binding names")),
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            ("cotangent", Json::String(c)) => cotangent = Some(c),
            (key, _) => return Err(bad(format!("bad derivative option `{key}`"))),
        }
    }
    let vjp = kind.ok_or_else(|| bad("`derivative.kind` must be \"gradient\" or \"vjp\""))?;
    let wrt = wrt
        .filter(|w| !w.is_empty())
        .ok_or_else(|| bad("`derivative.wrt` must list bindings"))?;
    if vjp != cotangent.is_some() {
        return Err(bad("a vjp needs a `cotangent`, and a gradient takes none"));
    }
    Ok(DerivativeOption {
        vjp,
        wrt,
        cotangent,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Input,
    Point,
}

/// The argument of one slot: a whole operand or the fields of a record input.
enum SlotArg<'a> {
    Whole(&'a Operand),
    Fields(Vec<(&'a str, &'a Operand)>, bool),
}

/// Where a binding reaches the plugin, and which source element each row takes.
struct Occurrence {
    name: String,
    slot: Slot,
    field: Option<String>,
    indices: Vec<usize>,
}

/// Run `query` on `plugin`. `values` holds the data and literal bindings.
pub fn execute(
    query: &Query,
    members: &BTreeMap<String, Member>,
    values: &HashMap<String, Value>,
    options: &Options,
    plugin: &dyn Plugin,
) -> Result<Vec<(String, Value)>, BridgeError> {
    let result = &query.result;
    let member = members.get(&result.member).ok_or_else(|| {
        unsupported(format!(
            "box `{}` has no member `{}`",
            query.box_name, result.member
        ))
    })?;
    if !member.queries.contains(&result.op) {
        return Err(unsupported(format!(
            "member `{}` does not answer `{}`",
            result.member,
            result.op.query_name()
        )));
    }
    if let Some(d) = &options.derivative {
        if result.op == Op::Sample {
            return Err(unsupported("`builtin_sample` has no derivative"));
        }
        if (d.vjp && !member.vjp) || (!d.vjp && !member.gradient) {
            let kind = if d.vjp { "vjp" } else { "gradient" };
            return Err(unsupported(format!(
                "member `{}` has no {kind}",
                result.member
            )));
        }
    }

    let (input, point, rest) = slots(query, member)?;
    let mut sizes = Vec::new();
    let mut rng = None;
    if result.op == Op::Sample {
        rng = Some(rng_of(query, values, &result.args[0])?);
        for size in rest {
            let n = match resolve(values, size)? {
                Value::Array { shape, data }
                    if shape.is_empty() && data.data_type() == &DataType::Int64 =>
                {
                    data.as_primitive::<arrow_array::types::Int64Type>()
                        .value(0)
                }
                _ => -1,
            };
            sizes.push(
                u64::try_from(n)
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| bad("sample sizes are positive integers"))?,
            );
        }
    } else if !rest.is_empty() {
        return Err(bad(format!(
            "too many arguments for `{}`",
            result.op.query_name()
        )));
    }

    let mut occurrences = Vec::new();
    let (rows, batch, input, point) = if result.broadcast {
        broadcast(values, input, point, &mut occurrences)?
    } else {
        let mut whole = |slot, arg: Option<SlotArg>| -> Result<Option<ArrayRef>, BridgeError> {
            let Some(arg) = arg else { return Ok(None) };
            Ok(Some(match arg {
                SlotArg::Whole(op) => {
                    if let Operand::Binding(name) = op {
                        occurrences.push(Occurrence {
                            name: name.clone(),
                            slot,
                            field: None,
                            indices: vec![0],
                        });
                    }
                    resolve(values, op)?.to_element()?
                }
                SlotArg::Fields(fields, _) => {
                    let mut arrays = Vec::new();
                    for (field, op) in &fields {
                        if let Operand::Binding(name) = op {
                            occurrences.push(Occurrence {
                                name: name.clone(),
                                slot,
                                field: Some(field.to_string()),
                                indices: vec![0],
                            });
                        }
                        arrays.push(resolve(values, op)?.to_element()?);
                    }
                    Arc::new(struct_array(fields.iter().map(|(f, _)| *f), arrays, 1))
                }
            }))
        };
        let input = whole(Slot::Input, input)?;
        let point = whole(Slot::Point, point)?;
        (1, Vec::new(), input, point)
    };

    let derivative = match &options.derivative {
        None => None,
        Some(d) => {
            for name in &d.wrt {
                if !values.contains_key(name) {
                    return Err(bad(format!(
                        "`wrt` names `{name}`, which is not a data or literal binding"
                    )));
                }
                if !occurrences.iter().any(|o| &o.name == name) {
                    return Err(bad(format!("`{name}` is not an argument of the result")));
                }
            }
            let cotangent = d
                .cotangent
                .as_ref()
                .map(|name| cotangent(values, name, &batch, rows))
                .transpose()?;
            Some(Derivative {
                cotangent,
                wrt_input: occurrences
                    .iter()
                    .any(|o| o.slot == Slot::Input && d.wrt.contains(&o.name)),
                wrt_point: occurrences
                    .iter()
                    .any(|o| o.slot == Slot::Point && d.wrt.contains(&o.name)),
            })
        }
    };

    let output = plugin.call(Call {
        op: result.op,
        member: result.member.clone(),
        rows,
        input,
        point,
        sizes: sizes.clone(),
        rng,
        derivative,
    })?;

    let value_batch: Vec<usize> = if result.op == Op::Sample {
        sizes.iter().map(|&n| n as usize).collect()
    } else {
        batch.clone()
    };
    let value = from_elements(&output.value, &value_batch)?;
    let mut parts = Vec::new();
    if let Some(d) = &options.derivative {
        if d.vjp {
            let ct = &values[d.cotangent.as_ref().expect("checked")];
            if ct.shape() != value.shape() {
                return Err(bad("the cotangent does not have the shape of the result"));
            }
        } else if !matches!(&value, Value::Array { data, .. } if is_real(data.data_type()))
            || output.value.data_type() != value_data_type(&value)
        {
            return Err(bad(
                "a gradient needs a real scalar result per row; use a vjp",
            ));
        }
    }
    let real32 =
        matches!(&value, Value::Array { data, .. } if data.data_type() == &DataType::Float32);
    parts.push(("value".to_owned(), value));
    if let Some(d) = &options.derivative {
        for name in &d.wrt {
            let derivative = accumulate(name, &values[name], &occurrences, &output, rows, real32)?;
            parts.push((format!("d_{name}"), derivative));
        }
    }
    if result.op == Op::Sample {
        let state = output
            .rng
            .ok_or_else(|| BridgeError::Failure("the box returned no RNG state".into()))?;
        let data: ArrayRef = Arc::new(Int64Array::from(
            state.iter().map(|&b| b as i64).collect::<Vec<_>>(),
        ));
        parts.push((
            "rngstate".to_owned(),
            Value::Array {
                shape: vec![state.len()],
                data,
            },
        ));
    }
    Ok(parts)
}

fn is_real(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Float64 | DataType::Float32)
}

fn value_data_type(value: &Value) -> &DataType {
    match value {
        Value::Array { data, .. } => data.data_type(),
        _ => &DataType::Null,
    }
}

/// Map the arguments of the result call to the input and point slots.
type Slots<'a> = (Option<SlotArg<'a>>, Option<SlotArg<'a>>, &'a [Operand]);

fn slots<'a>(query: &'a Query, member: &'a Member) -> Result<Slots<'a>, BridgeError> {
    let result = &query.result;
    let args = &result.args[..];
    let whole = |op: &'a Operand| match op {
        Operand::Record(fields) => {
            SlotArg::Fields(fields.iter().map(|(f, o)| (f.as_str(), o)).collect(), true)
        }
        op => SlotArg::Whole(op),
    };
    let arity = |n: usize| {
        bad(format!(
            "`{}` on a {:?} member takes {n} arguments",
            result.op.query_name(),
            member.kind
        ))
    };
    Ok(match result.op {
        Op::Apply => {
            let input = match (args, &result.kwargs[..]) {
                ([], kwargs) if !kwargs.is_empty() => {
                    SlotArg::Fields(kwargs.iter().map(|(f, o)| (f.as_str(), o)).collect(), false)
                }
                ([arg], []) => whole(arg),
                (args, []) if args.len() > 1 => match &member.input {
                    Some(crate::set::Set::Record(fields)) if fields.len() == args.len() => {
                        SlotArg::Fields(
                            fields.iter().map(|(f, _)| f.as_str()).zip(args).collect(),
                            false,
                        )
                    }
                    _ => {
                        return Err(unsupported(
                            "positional arguments need a record input of the same length",
                        ));
                    }
                },
                _ => {
                    return Err(bad(
                        "an application takes positional or keyword arguments, not both",
                    ));
                }
            };
            (Some(input), None, &[][..])
        }
        Op::Sample => match member.kind {
            Kind::Kernel => {
                let [_, input, rest @ ..] = args else {
                    return Err(arity(2));
                };
                (Some(whole(input)), None, rest)
            }
            _ => (None, None, &args[1..]),
        },
        _ => match (member.kind, args) {
            (Kind::Kernel, [input, x]) => (Some(whole(input)), Some(whole(x)), &[][..]),
            (Kind::Measure, [x]) => (None, Some(whole(x)), &[][..]),
            (Kind::Kernel, _) => return Err(arity(2)),
            _ => return Err(arity(1)),
        },
    })
}

fn resolve(values: &HashMap<String, Value>, op: &Operand) -> Result<Value, BridgeError> {
    match op {
        Operand::Binding(name) => values
            .get(name)
            .cloned()
            .ok_or_else(|| bad(format!("`{name}` is not a data or literal binding"))),
        Operand::Literal(value) => Ok(value.clone()),
        Operand::Record(fields) => fields
            .iter()
            .map(|(f, o)| Ok((f.clone(), resolve(values, o)?)))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Record),
        Operand::Rng(_) => Err(bad(
            "an RNG state is only the first argument of `builtin_sample`",
        )),
    }
}

fn rng_of(
    query: &Query,
    values: &HashMap<String, Value>,
    op: &Operand,
) -> Result<Rng, BridgeError> {
    let spec = match op {
        Operand::Rng(spec) => spec.as_ref(),
        Operand::Binding(name) => query
            .rngs
            .get(name)
            .ok_or_else(|| bad(format!("`{name}` is not an RNG binding")))?,
        _ => return Err(bad("`builtin_sample` takes an RNG state first")),
    };
    match spec {
        RngSpec::Seed(bytes) => Ok(Rng::Seed(bytes.clone())),
        RngSpec::State(op) => bytes_of(&resolve(values, op)?)
            .filter(|b| !b.is_empty())
            .map(Rng::State)
            .ok_or_else(|| bad("`rngstate` takes a non-empty vector of integers in 0..255")),
    }
}

type Batched = (usize, Vec<usize>, Option<ArrayRef>, Option<ArrayRef>);

/// Iterate collection operands elementwise (spec §04) and gather each row.
fn broadcast(
    values: &HashMap<String, Value>,
    input: Option<SlotArg>,
    point: Option<SlotArg>,
    occurrences: &mut Vec<Occurrence>,
) -> Result<Batched, BridgeError> {
    struct Leaf {
        slot: Slot,
        field: Option<String>,
        name: Option<String>,
        shape: Option<Vec<usize>>,
        elements: ArrayRef,
    }
    let mut leaves = Vec::new();
    let mut fields_of = HashMap::new();
    for (slot, arg) in [(Slot::Input, input), (Slot::Point, point)] {
        let ops: Vec<(Option<String>, &Operand)> = match arg {
            None => continue,
            Some(SlotArg::Whole(op)) => vec![(None, op)],
            Some(SlotArg::Fields(_, true)) => {
                return Err(bad("records are not broadcast operands (spec §04)"));
            }
            Some(SlotArg::Fields(fields, false)) => {
                fields_of.insert(
                    slot as u8,
                    fields
                        .iter()
                        .map(|(f, _)| f.to_string())
                        .collect::<Vec<_>>(),
                );
                fields
                    .into_iter()
                    .map(|(f, o)| (Some(f.to_owned()), o))
                    .collect()
            }
        };
        for (field, op) in ops {
            let (shape, elements) = match resolve(values, op)? {
                Value::Array { shape, data } if shape.is_empty() => (None, data),
                Value::Array { shape, data } => (Some(shape), data),
                Value::Table(rows) => (Some(vec![rows.len()]), Arc::new(rows) as ArrayRef),
                Value::Record(_) => {
                    return Err(bad("records are not broadcast operands (spec §04)"));
                }
            };
            let name = match op {
                Operand::Binding(name) => Some(name.clone()),
                _ => None,
            };
            leaves.push(Leaf {
                slot,
                field,
                name,
                shape,
                elements,
            });
        }
    }
    let shapes: Vec<&Vec<usize>> = leaves.iter().filter_map(|l| l.shape.as_ref()).collect();
    let axes = shapes.first().map_or(0, |s| s.len());
    if shapes.iter().any(|s| s.len() != axes) {
        return Err(bad(
            "broadcast collections need the same number of axes (spec §04)",
        ));
    }
    let mut batch = vec![1usize; axes];
    for shape in &shapes {
        for (b, &s) in batch.iter_mut().zip(shape.iter()) {
            if *b == 1 {
                *b = s;
            } else if s != 1 && s != *b {
                return Err(bad(
                    "broadcast axes differ in size and are not size one (spec §04)",
                ));
            }
        }
    }
    // A size-one axis expanded against a zero-length axis has length zero.
    for shape in &shapes {
        for (b, &s) in batch.iter_mut().zip(shape.iter()) {
            if s == 0 {
                *b = 0;
            }
        }
    }
    let rows: usize = batch.iter().product();
    let mut gathered: HashMap<(u8, Option<String>), ArrayRef> = HashMap::new();
    for leaf in leaves {
        let indices: Vec<usize> = match &leaf.shape {
            None => vec![0; rows],
            Some(shape) => (0..rows).map(|r| source_index(r, &batch, shape)).collect(),
        };
        let take = UInt64Array::from(indices.iter().map(|&i| i as u64).collect::<Vec<_>>());
        let array = arrow_select::take::take(&leaf.elements, &take, None)
            .map_err(|e| BridgeError::Failure(format!("Arrow take: {e}")))?;
        if let Some(name) = leaf.name {
            occurrences.push(Occurrence {
                name,
                slot: leaf.slot,
                field: leaf.field.clone(),
                indices,
            });
        }
        gathered.insert((leaf.slot as u8, leaf.field), array);
    }
    let mut slot_array = |slot: Slot| -> Option<ArrayRef> {
        match fields_of.get(&(slot as u8)) {
            Some(fields) => {
                let arrays = fields
                    .iter()
                    .map(|f| {
                        gathered
                            .remove(&(slot as u8, Some(f.clone())))
                            .expect("gathered")
                    })
                    .collect();
                Some(Arc::new(struct_array(
                    fields.iter().map(String::as_str),
                    arrays,
                    rows,
                )))
            }
            None => gathered.remove(&(slot as u8, None)),
        }
    };
    let input = slot_array(Slot::Input);
    let point = slot_array(Slot::Point);
    Ok((rows, batch, input, point))
}

/// The flat source index of batch row `row` in a collection of `shape`.
fn source_index(row: usize, batch: &[usize], shape: &[usize]) -> usize {
    let mut rest = row;
    let mut index = 0;
    let mut stride = 1;
    for k in (0..batch.len()).rev() {
        let i = rest % batch[k];
        rest /= batch[k];
        if shape[k] != 1 {
            index += i * stride;
        }
        stride *= shape[k];
    }
    index
}

/// The cotangent binding as `rows` elements of the result element shape.
fn cotangent(
    values: &HashMap<String, Value>,
    name: &str,
    batch: &[usize],
    rows: usize,
) -> Result<ArrayRef, BridgeError> {
    let Some(Value::Array { shape, data }) = values.get(name) else {
        return Err(bad(format!(
            "cotangent `{name}` must be a real array binding"
        )));
    };
    if !is_real(data.data_type()) || shape.len() < batch.len() || shape[..batch.len()] != *batch {
        return Err(bad(format!(
            "cotangent `{name}` does not have the shape of the result"
        )));
    }
    let array = nest(data.clone(), &shape[batch.len()..]);
    debug_assert_eq!(array.len(), rows);
    Ok(array)
}

/// Sum the per-row derivatives of the plugin into the shape of binding `name`.
fn accumulate(
    name: &str,
    binding: &Value,
    occurrences: &[Occurrence],
    output: &flatppl_bridge_abi::Output,
    rows: usize,
    real32: bool,
) -> Result<Value, BridgeError> {
    let Value::Array { shape, data } = binding else {
        return Err(unsupported(format!(
            "a derivative w.r.t. `{name}` needs an array binding"
        )));
    };
    if !is_real(data.data_type()) {
        return Err(unsupported(format!(
            "`{name}` is not real, so it has no derivative"
        )));
    }
    let failure = |m: &str| BridgeError::Failure(format!("derivative w.r.t. `{name}`: {m}"));
    let mut total = vec![0.0f64; shape.iter().product()];
    for occurrence in occurrences.iter().filter(|o| o.name == name) {
        let array = match occurrence.slot {
            Slot::Input => output.d_input.as_ref(),
            Slot::Point => output.d_point.as_ref(),
        }
        .ok_or_else(|| failure("the box returned no derivative"))?;
        let array = match &occurrence.field {
            None => array.clone(),
            Some(field) => array
                .as_struct_opt()
                .and_then(|s| s.column_by_name(field))
                .cloned()
                .ok_or_else(|| failure("the box returned no field derivative"))?,
        };
        if array.len() != rows {
            return Err(failure("the box returned the wrong number of rows"));
        }
        let (_, flat) = flatten(&array);
        let flat = arrow_cast::cast(&flat, &DataType::Float64).map_err(|_| failure("not real"))?;
        let flat = flat.as_primitive::<Float64Type>().values();
        let width = flat.len().checked_div(rows).unwrap_or(0);
        let sources = total.len().checked_div(width.max(1)).unwrap_or(0);
        if rows > 0 && (width * rows != flat.len() || sources * width != total.len()) {
            return Err(failure("the box returned the wrong shape"));
        }
        for (row, &source) in occurrence.indices.iter().enumerate() {
            let (into, from) = (
                &mut total[source * width..][..width],
                &flat[row * width..][..width],
            );
            into.iter_mut().zip(from).for_each(|(t, f)| *t += f);
        }
    }
    let data: ArrayRef = if real32 {
        Arc::new(Float32Array::from(
            total.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        ))
    } else {
        Arc::new(Float64Array::from(total))
    };
    Ok(Value::Array {
        shape: shape.clone(),
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_index_expands_size_one_axes() {
        assert_eq!(source_index(4, &[2, 3], &[2, 3]), 4);
        assert_eq!(source_index(4, &[2, 3], &[1, 3]), 1);
        assert_eq!(source_index(4, &[2, 3], &[2, 1]), 1);
    }

    #[test]
    fn options_parse_and_refuse() {
        let o = Options::from_json(br#"{"derivative": {"kind": "gradient", "wrt": ["theta"]}}"#)
            .unwrap();
        assert_eq!(o.derivative.unwrap().wrt, vec!["theta"]);
        assert!(Options::from_json(br#"{"derivative": {"kind": "vjp", "wrt": ["x"]}}"#).is_err());
        assert!(Options::from_json(br#"{"cache": false}"#).is_err());
    }
}
