//! The flatppl-bridge plugin interface: decoded calls, the [`Plugin`] trait, the
//! C ABI of `include/flatppl_bridge.h`, and [`export_plugin!`], which exposes a
//! Rust [`Plugin`] through that ABI. See `docs/plugin-abi.md`.

use arrow_array::{ArrayRef, make_array};
use arrow_data::ffi::FFI_ArrowArray;
use arrow_schema::ffi::FFI_ArrowSchema;
use std::ffi::{c_char, c_void};

pub use arrow_array;

pub const ABI_VERSION: u32 = 1;

pub const OP_APPLY: u32 = 0;
pub const OP_LOGDENSITY: u32 = 1;
pub const OP_SAMPLE: u32 = 2;
pub const OP_TOUNIFORM: u32 = 3;
pub const OP_FROMUNIFORM: u32 = 4;
pub const OP_TONORMAL: u32 = 5;
pub const OP_FROMNORMAL: u32 = 6;

pub const DERIV_NONE: u32 = 0;
pub const DERIV_GRADIENT: u32 = 1;
pub const DERIV_VJP: u32 = 2;

pub const WRT_INPUT: u32 = 1;
pub const WRT_POINT: u32 = 2;

pub const RNG_NONE: u32 = 0;
pub const RNG_SEED: u32 = 1;
pub const RNG_STATE: u32 = 2;

pub const OK: i32 = 0;
pub const INVALID: i32 = 1;
pub const UNSUPPORTED: i32 = 2;
pub const FAILURE: i32 = 3;

pub const INIT_SYMBOL: &[u8] = b"fpb_plugin_init_v1\0";
pub const FREE_ERROR_SYMBOL: &[u8] = b"fpb_plugin_free_error_v1\0";

/// A query operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Apply,
    LogDensity,
    Sample,
    ToUniform,
    FromUniform,
    ToNormal,
    FromNormal,
}

impl Op {
    pub fn code(self) -> u32 {
        match self {
            Op::Apply => OP_APPLY,
            Op::LogDensity => OP_LOGDENSITY,
            Op::Sample => OP_SAMPLE,
            Op::ToUniform => OP_TOUNIFORM,
            Op::FromUniform => OP_FROMUNIFORM,
            Op::ToNormal => OP_TONORMAL,
            Op::FromNormal => OP_FROMNORMAL,
        }
    }

    pub fn from_code(code: u32) -> Option<Op> {
        Some(match code {
            OP_APPLY => Op::Apply,
            OP_LOGDENSITY => Op::LogDensity,
            OP_SAMPLE => Op::Sample,
            OP_TOUNIFORM => Op::ToUniform,
            OP_FROMUNIFORM => Op::FromUniform,
            OP_TONORMAL => Op::ToNormal,
            OP_FROMNORMAL => Op::FromNormal,
            _ => return None,
        })
    }

    /// The query name of the operation, as describe lists it.
    pub fn query_name(self) -> &'static str {
        match self {
            Op::Apply => "apply",
            Op::LogDensity => "builtin_logdensityof",
            Op::Sample => "builtin_sample",
            Op::ToUniform => "builtin_touniform",
            Op::FromUniform => "builtin_fromuniform",
            Op::ToNormal => "builtin_tonormal",
            Op::FromNormal => "builtin_fromnormal",
        }
    }
}

/// The RNG state of one row of a sampling call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rng {
    /// A fresh state from these seed bytes (`rnginit`).
    Seed(Vec<u8>),
    /// A state that a previous reply returned (`rngstate`).
    State(Vec<u8>),
}

/// A derivative request.
#[derive(Clone, Debug)]
pub struct Derivative {
    /// `None` for a gradient, the cotangent for a VJP.
    pub cotangent: Option<ArrayRef>,
    pub wrt_input: bool,
    pub wrt_point: bool,
}

/// A decoded call. Each array holds `rows` FlatPPL values.
#[derive(Clone, Debug)]
pub struct Call {
    pub op: Op,
    pub member: String,
    pub rows: usize,
    pub input: Option<ArrayRef>,
    pub point: Option<ArrayRef>,
    pub sizes: Vec<u64>,
    /// One RNG state per row for [`Op::Sample`], empty otherwise. Every row
    /// has the same kind and byte length.
    pub rng: Vec<Rng>,
    pub derivative: Option<Derivative>,
}

/// The result of a call.
#[derive(Clone, Debug)]
pub struct Output {
    /// `rows` values, or for a sample `rows` times the product of the sizes.
    pub value: ArrayRef,
    pub d_input: Option<ArrayRef>,
    pub d_point: Option<ArrayRef>,
    /// For a sample, the new RNG state of each row, all of one byte length.
    pub rng: Vec<Vec<u8>>,
}

/// A call failure, by HTTP class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Bad input values (400).
    Invalid(String),
    /// An unsupported query (422).
    Unsupported(String),
    /// A box failure (500).
    Failure(String),
}

impl Error {
    pub fn message(&self) -> &str {
        match self {
            Error::Invalid(m) | Error::Unsupported(m) | Error::Failure(m) => m,
        }
    }

    pub fn code(&self) -> i32 {
        match self {
            Error::Invalid(_) => INVALID,
            Error::Unsupported(_) => UNSUPPORTED,
            Error::Failure(_) => FAILURE,
        }
    }

    pub fn from_code(code: i32, message: String) -> Error {
        match code {
            INVALID => Error::Invalid(message),
            UNSUPPORTED => Error::Unsupported(message),
            _ => Error::Failure(message),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}

/// A box implementation. Calls may arrive from several threads at once.
pub trait Plugin: Send + Sync {
    /// The describe JSON of `docs/protocol.md`: one member object per member.
    fn describe(&self) -> Result<String, Error>;
    fn call(&self, call: Call) -> Result<Output, Error>;
}

/// `fpb_array`: an Arrow array with its schema.
#[repr(C)]
pub struct FpbArray {
    pub schema: FFI_ArrowSchema,
    pub array: FFI_ArrowArray,
}

impl FpbArray {
    pub fn empty() -> FpbArray {
        FpbArray {
            schema: FFI_ArrowSchema::empty(),
            array: FFI_ArrowArray::empty(),
        }
    }

    pub fn export(array: &ArrayRef) -> Result<FpbArray, Error> {
        let (array, schema) = arrow_array::ffi::to_ffi(&array.to_data())
            .map_err(|e| Error::Failure(format!("Arrow export: {e}")))?;
        Ok(FpbArray { schema, array })
    }

    /// Import the array, or `None` if it is empty or released.
    pub fn import(self) -> Result<Option<ArrayRef>, Error> {
        if self.array.is_released() {
            return Ok(None);
        }
        let FpbArray { schema, array } = self;
        // SAFETY: the producer filled both structs by the C Data Interface.
        let data = unsafe { arrow_array::ffi::from_ffi(array, &schema) }
            .map_err(|e| Error::Invalid(format!("Arrow import: {e}")))?;
        Ok(Some(make_array(data)))
    }

    /// Move the array out of `ptr`, leaving a released array in place.
    ///
    /// # Safety
    /// `ptr` is null or points to a valid `fpb_array`.
    pub unsafe fn take(ptr: *mut FpbArray) -> Result<Option<ArrayRef>, Error> {
        if ptr.is_null() {
            return Ok(None);
        }
        // SAFETY: the caller guarantees a valid pointer.
        unsafe { std::ptr::replace(ptr, FpbArray::empty()) }.import()
    }
}

/// `fpb_call`.
#[repr(C)]
pub struct FpbCall {
    pub op: u32,
    pub member: *const c_char,
    pub rows: i64,
    pub input: *mut FpbArray,
    pub point: *mut FpbArray,
    pub sizes: *const u64,
    pub n_sizes: usize,
    pub rng_kind: u32,
    pub rng: *const u8,
    pub rng_len: usize,
    pub deriv: u32,
    pub wrt: u32,
    pub cotangent: *mut FpbArray,
}

/// `fpb_result`.
#[repr(C)]
pub struct FpbResult {
    pub value: FpbArray,
    pub d_input: FpbArray,
    pub d_point: FpbArray,
    pub rng: *mut u8,
    pub rng_len: usize,
    pub error: *mut c_char,
    pub private_data: *mut c_void,
}

impl FpbResult {
    pub fn empty() -> FpbResult {
        FpbResult {
            value: FpbArray::empty(),
            d_input: FpbArray::empty(),
            d_point: FpbArray::empty(),
            rng: std::ptr::null_mut(),
            rng_len: 0,
            error: std::ptr::null_mut(),
            private_data: std::ptr::null_mut(),
        }
    }
}

/// `fpb_plugin`.
#[repr(C)]
pub struct FpbPlugin {
    pub abi_version: u32,
    pub state: *mut c_void,
    pub describe: unsafe extern "C" fn(state: *mut c_void) -> *const c_char,
    pub call:
        unsafe extern "C" fn(state: *mut c_void, call: *const FpbCall, out: *mut FpbResult) -> i32,
    pub release: unsafe extern "C" fn(state: *mut c_void, out: *mut FpbResult),
    pub destroy: unsafe extern "C" fn(state: *mut c_void),
}

pub type InitFn =
    unsafe extern "C" fn(config: *const c_char, error: *mut *mut c_char) -> *const FpbPlugin;
pub type FreeErrorFn = unsafe extern "C" fn(error: *mut c_char);

/// Decode a C call. Moves the input arrays out of `call`.
///
/// # Safety
/// `call` points to a valid `fpb_call`.
pub unsafe fn decode_call(call: *const FpbCall) -> Result<Call, Error> {
    // SAFETY: the caller guarantees a valid call; pointers follow the header.
    unsafe {
        let call = &*call;
        let op = Op::from_code(call.op)
            .ok_or_else(|| Error::Unsupported(format!("unknown operation {}", call.op)))?;
        let member = std::ffi::CStr::from_ptr(call.member)
            .to_str()
            .map_err(|_| Error::Invalid("member name is not UTF-8".into()))?
            .to_owned();
        let rows =
            usize::try_from(call.rows).map_err(|_| Error::Invalid("negative rows".into()))?;
        let states = || -> Vec<Vec<u8>> {
            if call.rng_len == 0 {
                return vec![Vec::new(); rows];
            }
            std::slice::from_raw_parts(call.rng, rows * call.rng_len)
                .chunks(call.rng_len)
                .map(<[u8]>::to_vec)
                .collect()
        };
        let rng = match call.rng_kind {
            RNG_SEED => states().into_iter().map(Rng::Seed).collect(),
            RNG_STATE => states().into_iter().map(Rng::State).collect(),
            _ => Vec::new(),
        };
        let sizes = if call.n_sizes == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(call.sizes, call.n_sizes).to_vec()
        };
        let derivative = match call.deriv {
            DERIV_NONE => None,
            DERIV_GRADIENT | DERIV_VJP => Some(Derivative {
                cotangent: if call.deriv == DERIV_VJP {
                    FpbArray::take(call.cotangent)?
                } else {
                    None
                },
                wrt_input: call.wrt & WRT_INPUT != 0,
                wrt_point: call.wrt & WRT_POINT != 0,
            }),
            other => {
                return Err(Error::Unsupported(format!(
                    "unknown derivative kind {other}"
                )));
            }
        };
        Ok(Call {
            op,
            member,
            rows,
            input: FpbArray::take(call.input)?,
            point: FpbArray::take(call.point)?,
            sizes,
            rng,
            derivative,
        })
    }
}

/// Glue for [`export_plugin!`]. Not for direct use.
#[doc(hidden)]
pub mod export {
    use super::*;
    use std::ffi::{CString, c_char, c_void};

    pub struct State {
        table: FpbPlugin,
        plugin: Box<dyn Plugin>,
        describe: CString,
    }

    struct Owned {
        rng: Vec<u8>,
        error: Option<CString>,
    }

    fn message(text: &str) -> CString {
        CString::new(text.replace('\0', " ")).expect("NUL bytes removed")
    }

    unsafe extern "C" fn describe(state: *mut c_void) -> *const c_char {
        // SAFETY: `state` is the `State` that `init` boxed.
        unsafe { (*(state as *const State)).describe.as_ptr() }
    }

    unsafe extern "C" fn call(
        state: *mut c_void,
        call: *const FpbCall,
        out: *mut FpbResult,
    ) -> i32 {
        // SAFETY: the core passes the state from `init` and valid call and result structs.
        let state = unsafe { &*(state as *const State) };
        let result = unsafe { decode_call(call) }.and_then(|call| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.plugin.call(call)))
                .unwrap_or_else(|_| Err(Error::Failure("plugin panicked".into())))
        });
        let out = unsafe { &mut *out };
        let mut owned = Box::new(Owned {
            rng: Vec::new(),
            error: None,
        });
        let code = match result.and_then(|output| {
            let export = |array: &Option<ArrayRef>| match array {
                Some(array) => FpbArray::export(array),
                None => Ok(FpbArray::empty()),
            };
            Ok((
                FpbArray::export(&output.value)?,
                export(&output.d_input)?,
                export(&output.d_point)?,
                output.rng,
            ))
        }) {
            Ok((value, d_input, d_point, rng)) => {
                let len = rng.first().map_or(0, Vec::len);
                if rng.iter().any(|r| r.len() != len) {
                    let text = message("the plugin returned RNG states of different lengths");
                    out.error = text.as_ptr() as *mut c_char;
                    owned.error = Some(text);
                    out.private_data = Box::into_raw(owned) as *mut c_void;
                    return FAILURE;
                }
                out.value = value;
                out.d_input = d_input;
                out.d_point = d_point;
                if !rng.is_empty() {
                    owned.rng = rng.concat();
                    out.rng = owned.rng.as_mut_ptr();
                    out.rng_len = len;
                }
                OK
            }
            Err(error) => {
                let text = message(error.message());
                out.error = text.as_ptr() as *mut c_char;
                owned.error = Some(text);
                error.code()
            }
        };
        out.private_data = Box::into_raw(owned) as *mut c_void;
        code
    }

    unsafe extern "C" fn release(_state: *mut c_void, out: *mut FpbResult) {
        // SAFETY: `private_data` is the `Owned` box that `call` leaked.
        unsafe {
            let out = &mut *out;
            if !out.private_data.is_null() {
                drop(Box::from_raw(out.private_data as *mut Owned));
                out.private_data = std::ptr::null_mut();
            }
            out.rng = std::ptr::null_mut();
            out.rng_len = 0;
            out.error = std::ptr::null_mut();
        }
    }

    unsafe extern "C" fn destroy(state: *mut c_void) {
        // SAFETY: `state` is the `State` that `table` boxed.
        unsafe { drop(Box::from_raw(state as *mut State)) }
    }

    /// Build the function table for a plugin.
    pub fn table(plugin: Box<dyn Plugin>) -> Result<*const FpbPlugin, String> {
        let text = plugin.describe().map_err(|e| e.message().to_owned())?;
        let state = Box::into_raw(Box::new(State {
            table: FpbPlugin {
                abi_version: ABI_VERSION,
                state: std::ptr::null_mut(),
                describe,
                call,
                release,
                destroy,
            },
            plugin,
            describe: message(&text),
        }));
        // SAFETY: `state` was just boxed. destroy frees it, table included.
        unsafe {
            (*state).table.state = state as *mut c_void;
            Ok(&(*state).table)
        }
    }

    pub fn error(text: &str) -> *mut c_char {
        message(text).into_raw()
    }

    /// # Safety
    /// `error` came from [`error`].
    pub unsafe fn free_error(error: *mut c_char) {
        if !error.is_null() {
            // SAFETY: the caller guarantees the pointer came from `CString::into_raw`.
            drop(unsafe { CString::from_raw(error) });
        }
    }
}

/// Export the C entry points for a plugin type. `$init` is a function
/// `fn(&str) -> Result<impl Plugin, String>` that builds the plugin from its
/// configuration string.
#[macro_export]
macro_rules! export_plugin {
    ($init:path) => {
        /// # Safety
        /// `config` is null or a NUL-terminated string. `error` is valid for writes.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn fpb_plugin_init_v1(
            config: *const ::std::ffi::c_char,
            error: *mut *mut ::std::ffi::c_char,
        ) -> *const $crate::FpbPlugin {
            let config = if config.is_null() {
                ""
            } else {
                // SAFETY: the caller passes a NUL-terminated string.
                match unsafe { ::std::ffi::CStr::from_ptr(config) }.to_str() {
                    Ok(text) => text,
                    Err(_) => {
                        // SAFETY: the caller passes a writable pointer.
                        unsafe { *error = $crate::export::error("config is not UTF-8") };
                        return ::std::ptr::null();
                    }
                }
            };
            let built = ::std::panic::catch_unwind(|| $init(config))
                .unwrap_or_else(|_| Err("plugin init panicked".to_string()))
                .and_then(|plugin| $crate::export::table(Box::new(plugin)));
            match built {
                Ok(table) => table,
                Err(text) => {
                    // SAFETY: the caller passes a writable pointer.
                    unsafe { *error = $crate::export::error(&text) };
                    ::std::ptr::null()
                }
            }
        }

        /// # Safety
        /// `error` came from `fpb_plugin_init_v1`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn fpb_plugin_free_error_v1(error: *mut ::std::ffi::c_char) {
            // SAFETY: forwarded from the caller.
            unsafe { $crate::export::free_error(error) }
        }
    };
}
