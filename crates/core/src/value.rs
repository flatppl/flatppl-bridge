//! FlatPPL values in Arrow: the part layout and the element encoding of
//! `docs/protocol.md`.

use crate::error::{BridgeError, bad, unsupported};
use crate::set::{Scalar, Set};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, RecordBatch, StructArray};
use arrow_ipc::reader::{FileReader, StreamReader};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Fields, Schema};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

/// The schema metadata key that carries a part's value set.
pub const VALUESET_KEY: &str = "flatppl.valueset";

/// Rows per record batch in a reply part.
const BATCH_ROWS: usize = 65_536;

/// A FlatPPL value.
#[derive(Clone, Debug)]
pub enum Value {
    /// A scalar (empty shape) or an array: row-major data of a primitive type
    /// (`Float64`, `Float32`, `Int64` or `Boolean`).
    Array {
        shape: Vec<usize>,
        data: ArrayRef,
    },
    Record(Vec<(String, Value)>),
    /// A table, one struct element per row.
    Table(StructArray),
}

impl Value {
    pub fn scalar(data: ArrayRef) -> Value {
        Value::Array {
            shape: Vec::new(),
            data,
        }
    }

    pub fn shape(&self) -> Option<&[usize]> {
        match self {
            Value::Array { shape, .. } => Some(shape),
            _ => None,
        }
    }

    /// The value set of this value, from its Arrow types.
    pub fn set(&self) -> Set {
        match self {
            Value::Array { shape, data } => pow(scalar_set(data.data_type()), shape),
            Value::Record(fields) => {
                Set::Record(fields.iter().map(|(n, v)| (n.clone(), v.set())).collect())
            }
            Value::Table(rows) => {
                Set::Pow(Box::new(element_set(rows.data_type())), vec![rows.len()])
            }
        }
    }

    /// This value as one Arrow element (an array of length 1).
    pub fn to_element(&self) -> Result<ArrayRef, BridgeError> {
        match self {
            Value::Array { shape, data } => Ok(nest(data.clone(), shape)),
            Value::Record(fields) => {
                let arrays = fields
                    .iter()
                    .map(|(_, v)| v.to_element())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Arc::new(struct_array(
                    fields.iter().map(|(n, _)| n.as_str()),
                    arrays,
                    1,
                )))
            }
            Value::Table(_) => Err(unsupported(
                "a table is accepted only as a broadcast operand",
            )),
        }
    }
}

fn scalar_set(data_type: &DataType) -> Set {
    match data_type {
        DataType::Int64 => Set::Integers,
        DataType::Boolean => Set::Booleans,
        _ => Set::Reals,
    }
}

fn pow(elem: Set, shape: &[usize]) -> Set {
    if shape.is_empty() {
        elem
    } else {
        Set::Pow(Box::new(elem), shape.to_vec())
    }
}

/// The value set of one element of an Arrow type.
fn element_set(data_type: &DataType) -> Set {
    match data_type {
        DataType::FixedSizeList(..) => {
            let (dims, inner) = list_dims(data_type);
            pow(scalar_set(inner), &dims)
        }
        DataType::Struct(fields) => Set::Record(
            fields
                .iter()
                .map(|f| (f.name().clone(), element_set(f.data_type())))
                .collect(),
        ),
        other => scalar_set(other),
    }
}

fn list_dims(mut data_type: &DataType) -> (Vec<usize>, &DataType) {
    let mut dims = Vec::new();
    while let DataType::FixedSizeList(field, size) = data_type {
        dims.push(*size as usize);
        data_type = field.data_type();
    }
    (dims, data_type)
}

/// Wrap flat row-major data into nested fixed-size lists of `dims`.
pub fn nest(data: ArrayRef, dims: &[usize]) -> ArrayRef {
    dims.iter().rev().fold(data, |inner, &d| {
        let field = Arc::new(Field::new("item", inner.data_type().clone(), false));
        Arc::new(FixedSizeListArray::new(field, d as i32, inner, None))
    })
}

/// Split nested fixed-size lists into element dims and flat data.
pub fn flatten(array: &ArrayRef) -> (Vec<usize>, ArrayRef) {
    match array.data_type() {
        DataType::FixedSizeList(_, size) => {
            let list = array.as_fixed_size_list();
            let (mut dims, data) = flatten(list.values());
            dims.insert(0, *size as usize);
            (dims, data)
        }
        _ => (Vec::new(), array.clone()),
    }
}

/// Build a struct array from named children.
pub fn struct_array<'a>(
    names: impl Iterator<Item = &'a str>,
    arrays: Vec<ArrayRef>,
    len: usize,
) -> StructArray {
    let fields: Fields = names
        .zip(&arrays)
        .map(|(name, array)| Field::new(name, array.data_type().clone(), false))
        .collect();
    StructArray::try_new_with_length(fields, arrays, None, len).expect("children match the fields")
}

/// Decode `rows` elements of an Arrow array into a value of batch shape `batch`.
pub fn from_elements(array: &ArrayRef, batch: &[usize]) -> Result<Value, BridgeError> {
    let rows: usize = batch.iter().product();
    if array.len() != rows {
        return Err(BridgeError::Failure(format!(
            "the box returned {} values where {rows} were expected",
            array.len()
        )));
    }
    if array.null_count() > 0 {
        return Err(BridgeError::Failure("the box returned null values".into()));
    }
    match array.data_type() {
        DataType::Struct(_) if batch.is_empty() => element_value(array, 0),
        DataType::Struct(_) if batch.len() == 1 => Ok(Value::Table(array.as_struct().clone())),
        DataType::Struct(_) => Err(unsupported(
            "a record result needs a single broadcast axis (spec §04)",
        )),
        _ => {
            let (dims, data) = flatten(array);
            let data = normalize(&data).map_err(BridgeError::Failure)?;
            Ok(Value::Array {
                shape: batch.iter().copied().chain(dims).collect(),
                data,
            })
        }
    }
}

/// The value of element `index` of an Arrow array.
pub fn element_value(array: &ArrayRef, index: usize) -> Result<Value, BridgeError> {
    match array.data_type() {
        DataType::Struct(fields) => {
            let rows = array.as_struct();
            fields
                .iter()
                .zip(rows.columns())
                .map(|(f, column)| Ok((f.name().clone(), element_value(column, index)?)))
                .collect::<Result<Vec<_>, BridgeError>>()
                .map(Value::Record)
        }
        _ => {
            let (dims, data) = flatten(&array.slice(index, 1));
            Ok(Value::Array {
                shape: dims,
                data: normalize(&data).map_err(bad)?,
            })
        }
    }
}

/// Cast integer types to `Int64` and `Float16` to `Float32`. Refuse other types.
fn normalize(data: &ArrayRef) -> Result<ArrayRef, String> {
    match data.data_type() {
        DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Boolean => {
            Ok(data.clone())
        }
        DataType::Float16 => cast(data, &DataType::Float32),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => cast(data, &DataType::Int64),
        other => Err(format!("Arrow type {other} holds no FlatPPL value")),
    }
}

fn cast(data: &ArrayRef, to: &DataType) -> Result<ArrayRef, String> {
    let options = arrow_cast::CastOptions {
        safe: false,
        ..Default::default()
    };
    arrow_cast::cast_with_options(data, to, &options).map_err(|e| format!("Arrow cast: {e}"))
}

/// Read every record batch of an Arrow IPC stream or file.
fn read_batch(bytes: &[u8]) -> Result<RecordBatch, BridgeError> {
    let error = |e: arrow_schema::ArrowError| bad(format!("Arrow IPC: {e}"));
    let (schema, batches) = if bytes.starts_with(b"ARROW1") {
        let reader = FileReader::try_new(Cursor::new(bytes), None).map_err(error)?;
        let schema = reader.schema();
        (
            schema,
            reader.collect::<Result<Vec<_>, _>>().map_err(error)?,
        )
    } else {
        let reader = StreamReader::try_new(Cursor::new(bytes), None).map_err(error)?;
        let schema = reader.schema();
        (
            schema,
            reader.collect::<Result<Vec<_>, _>>().map_err(error)?,
        )
    };
    arrow_select::concat::concat_batches(&schema, &batches).map_err(error)
}

/// Decode a data part as a value of set `set` (spec §07 `load_data`).
pub fn decode_part(bytes: &[u8], set: &Set) -> Result<Value, BridgeError> {
    let batch = read_batch(bytes)?;
    let value = match set {
        Set::Anything => {
            return Err(unsupported(
                "`load_data` with value set `anything` cannot be read (spec §07)",
            ));
        }
        Set::Record(fields) => {
            if batch.num_rows() == 0 {
                return Err(bad("a record part needs one row"));
            }
            let fields = fields
                .iter()
                .map(|(name, _)| {
                    let column = batch
                        .column_by_name(name)
                        .ok_or_else(|| bad(format!("the part has no column `{name}`")))?;
                    Ok((name.clone(), element_value(column, 0)?))
                })
                .collect::<Result<Vec<_>, BridgeError>>()?;
            Value::Record(fields)
        }
        Set::Pow(elem, dims) if matches!(**elem, Set::Record(_)) => {
            let Set::Record(fields) = &**elem else {
                unreachable!()
            };
            let [rows] = dims[..] else {
                return Err(unsupported("a table has one axis"));
            };
            if batch.num_rows() < rows {
                return Err(bad(format!(
                    "the part has {} rows, the value set needs {rows}",
                    batch.num_rows()
                )));
            }
            let mut arrays = Vec::new();
            for (name, _) in fields {
                let column = batch
                    .column_by_name(name)
                    .ok_or_else(|| bad(format!("the part has no column `{name}`")))?;
                let (dims, data) = flatten(&column.slice(0, rows));
                arrays.push(nest(normalize(&data).map_err(bad)?, &dims));
            }
            Value::Table(struct_array(
                fields.iter().map(|(n, _)| n.as_str()),
                arrays,
                rows,
            ))
        }
        Set::Pow(_, dims) => {
            let column = single_column(&batch)?;
            if column.len() < dims[0] {
                return Err(bad(format!(
                    "the part has {} rows, the value set needs {}",
                    column.len(),
                    dims[0]
                )));
            }
            let (inner, data) = flatten(&column.slice(0, dims[0]));
            Value::Array {
                shape: std::iter::once(dims[0]).chain(inner).collect(),
                data: normalize(&data).map_err(bad)?,
            }
        }
        _ => {
            let column = single_column(&batch)?;
            if column.is_empty() {
                return Err(bad("a scalar part needs one row"));
            }
            element_value(column, 0)?
        }
    };
    let value = coerce(value, set)?;
    check(&value, set)?;
    Ok(value)
}

fn single_column(batch: &RecordBatch) -> Result<&ArrayRef, BridgeError> {
    match batch.columns() {
        [column] => Ok(column),
        columns => Err(bad(format!(
            "the part has {} columns, the value set needs one",
            columns.len()
        ))),
    }
}

/// Convert integer data to reals where the set asks for reals.
pub fn coerce(value: Value, set: &Set) -> Result<Value, BridgeError> {
    let elem = match set {
        Set::Pow(elem, _) => elem.as_ref(),
        other => other,
    };
    match (value, set) {
        (Value::Array { shape, data }, _)
            if elem.scalar() == Some(Scalar::Real) && data.data_type() == &DataType::Int64 =>
        {
            Ok(Value::Array {
                shape,
                data: cast(&data, &DataType::Float64).map_err(bad)?,
            })
        }
        (Value::Record(values), Set::Record(fields)) => values
            .into_iter()
            .map(|(name, v)| match fields.iter().find(|(n, _)| *n == name) {
                Some((_, s)) => Ok((name, coerce(v, s)?)),
                None => Ok((name, v)),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Record),
        (value, _) => Ok(value),
    }
}

/// Check that `value` lies in `set`.
pub fn check(value: &Value, set: &Set) -> Result<(), BridgeError> {
    let mismatch = || {
        bad(format!(
            "the value of set `{}` is not in `{set}`",
            value.set()
        ))
    };
    match (value, set) {
        (_, Set::Anything) => Ok(()),
        (Value::Array { shape, data }, _) => {
            let (elem, dims): (&Set, &[usize]) = match set {
                Set::Pow(elem, dims) => (elem, dims),
                scalar => (scalar, &[]),
            };
            if shape != dims {
                return Err(mismatch());
            }
            check_data(data, elem).map_err(|_| mismatch())
        }
        (Value::Record(values), Set::Record(fields)) => {
            for (name, s) in fields {
                let v = values
                    .iter()
                    .find(|(n, _)| n == name)
                    .ok_or_else(|| bad(format!("the record has no field `{name}`")))?;
                check(&v.1, s)?;
            }
            Ok(())
        }
        (Value::Table(rows), Set::Pow(elem, dims)) => {
            let Set::Record(fields) = &**elem else {
                return Err(mismatch());
            };
            if dims[..] != [rows.len()] {
                return Err(mismatch());
            }
            for (name, s) in fields {
                let column = rows
                    .column_by_name(name)
                    .ok_or_else(|| bad(format!("the table has no column `{name}`")))?;
                let (dims, data) = flatten(column);
                check(&Value::Array { shape: dims, data }, s).map_err(|_| mismatch())?;
            }
            Ok(())
        }
        _ => Err(mismatch()),
    }
}

fn check_data(data: &ArrayRef, elem: &Set) -> Result<(), ()> {
    if data.null_count() > 0 {
        return Err(());
    }
    let ok = match (elem.scalar(), data.data_type()) {
        (Some(Scalar::Real), DataType::Float64) => data
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .all(|&x| elem.contains_real(x)),
        (Some(Scalar::Real), DataType::Float32) => data
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .all(|&x| elem.contains_real(x as f64)),
        (Some(Scalar::Integer), DataType::Int64) => data
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .all(|&n| elem.contains_int(n)),
        (Some(Scalar::Boolean), DataType::Boolean) => true,
        _ => false,
    };
    if ok { Ok(()) } else { Err(()) }
}

/// Encode a value in the part layout as an Arrow IPC stream.
pub fn encode_part(value: &Value) -> Result<Vec<u8>, BridgeError> {
    let (names, columns, rows): (Vec<String>, Vec<ArrayRef>, usize) = match value {
        Value::Array { shape, data } if shape.is_empty() => {
            (vec!["value".into()], vec![data.clone()], 1)
        }
        Value::Array { shape, data } => (
            vec!["value".into()],
            vec![nest(data.clone(), &shape[1..])],
            shape[0],
        ),
        Value::Record(fields) => (
            fields.iter().map(|(n, _)| n.clone()).collect(),
            fields
                .iter()
                .map(|(_, v)| v.to_element())
                .collect::<Result<_, _>>()?,
            1,
        ),
        Value::Table(rows) => (
            rows.fields().iter().map(|f| f.name().clone()).collect(),
            rows.columns().to_vec(),
            rows.len(),
        ),
    };
    let fields: Vec<Field> = names
        .iter()
        .zip(&columns)
        .map(|(n, c)| Field::new(n, c.data_type().clone(), false))
        .collect();
    let metadata = HashMap::from([(VALUESET_KEY.to_owned(), value.set().to_string())]);
    let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
    let failure = |e: arrow_schema::ArrowError| BridgeError::Failure(format!("Arrow IPC: {e}"));
    let mut writer = StreamWriter::try_new(Vec::new(), &schema).map_err(failure)?;
    let mut start = 0;
    loop {
        let len = BATCH_ROWS.min(rows - start);
        let batch = RecordBatch::try_new(
            schema.clone(),
            columns.iter().map(|c| c.slice(start, len)).collect(),
        )
        .map_err(failure)?;
        writer.write(&batch).map_err(failure)?;
        start += len;
        if start >= rows {
            break;
        }
    }
    writer.into_inner().map_err(failure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int64Array};

    fn reals(values: Vec<f64>) -> ArrayRef {
        Arc::new(Float64Array::from(values))
    }

    #[test]
    fn matrix_round_trips_through_a_part() {
        let value = Value::Array {
            shape: vec![2, 3],
            data: reals(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let bytes = encode_part(&value).unwrap();
        let set = Set::Pow(Box::new(Set::Reals), vec![2, 3]);
        let back = decode_part(&bytes, &set).unwrap();
        let Value::Array { shape, data } = back else {
            panic!()
        };
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(
            data.as_primitive::<Float64Type>().values().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn integers_read_as_reals() {
        let value = Value::Array {
            shape: vec![2],
            data: Arc::new(Int64Array::from(vec![1, 2])),
        };
        let bytes = encode_part(&value).unwrap();
        let back = decode_part(&bytes, &Set::Pow(Box::new(Set::Reals), vec![2])).unwrap();
        assert_eq!(back.set(), Set::Pow(Box::new(Set::Reals), vec![2]));
    }

    #[test]
    fn refuses_values_outside_the_set() {
        let value = Value::scalar(reals(vec![-1.0]));
        let bytes = encode_part(&value).unwrap();
        assert!(matches!(
            decode_part(&bytes, &Set::PosReals),
            Err(BridgeError::BadRequest(_))
        ));
    }

    #[test]
    fn table_and_record_round_trip() {
        let rows = struct_array(
            ["mu", "sigma"].into_iter(),
            vec![reals(vec![0.0, 1.0]), reals(vec![1.0, 2.0])],
            2,
        );
        let bytes = encode_part(&Value::Table(rows)).unwrap();
        let record = Set::Record(vec![
            ("mu".into(), Set::Reals),
            ("sigma".into(), Set::PosReals),
        ]);
        let table = decode_part(&bytes, &Set::Pow(Box::new(record.clone()), vec![2])).unwrap();
        assert!(matches!(table, Value::Table(ref t) if t.len() == 2));
        let first = decode_part(&bytes, &record).unwrap();
        assert!(matches!(first, Value::Record(ref f) if f.len() == 2));
    }
}
