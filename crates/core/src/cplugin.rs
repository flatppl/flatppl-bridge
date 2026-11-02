//! Load a plugin library through the C ABI.

use flatppl_bridge_abi::{
    Call, DERIV_GRADIENT, DERIV_NONE, DERIV_VJP, Error, FREE_ERROR_SYMBOL, FpbArray, FpbCall,
    FpbPlugin, FpbResult, FreeErrorFn, INIT_SYMBOL, InitFn, OK, Output, Plugin, RNG_NONE, RNG_SEED,
    RNG_STATE, Rng, WRT_INPUT, WRT_POINT,
};
use std::ffi::{CStr, CString, c_char};
use std::path::Path;

/// A plugin library loaded through `fpb_plugin_init_v1`.
pub struct CPlugin {
    table: *const FpbPlugin,
    describe: String,
    // Never unloaded: a library's threads and destructors may outlive dlclose.
    _library: std::mem::ManuallyDrop<libloading::Library>,
}

// SAFETY: the ABI requires a plugin to accept calls from several threads at once.
unsafe impl Send for CPlugin {}
// SAFETY: as above.
unsafe impl Sync for CPlugin {}

impl CPlugin {
    /// Load the library at `path` and initialize it with `config`.
    pub fn load(path: &Path, config: &str) -> Result<CPlugin, String> {
        let config = CString::new(config)
            .map_err(|_| "the plugin config contains a NUL byte".to_string())?;
        // SAFETY: loading runs the library's initializers. The operator chose the library.
        let library = unsafe { libloading::Library::new(path) }
            .map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: the symbols have the types the header declares.
        let (init, free_error) = unsafe {
            let init: libloading::Symbol<InitFn> = library
                .get(INIT_SYMBOL)
                .map_err(|e| format!("{}: no fpb_plugin_init_v1: {e}", path.display()))?;
            let free_error: Option<libloading::Symbol<FreeErrorFn>> =
                library.get(FREE_ERROR_SYMBOL).ok();
            (*init, free_error.map(|f| *f))
        };
        let mut error: *mut c_char = std::ptr::null_mut();
        // SAFETY: `config` is NUL-terminated and `error` is writable.
        let table = unsafe { init(config.as_ptr(), &mut error) };
        if table.is_null() {
            let message = if error.is_null() {
                "plugin init failed".to_string()
            } else {
                // SAFETY: the plugin set a NUL-terminated message, freed by its own function.
                let text = unsafe { CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned();
                if let Some(free_error) = free_error {
                    unsafe { free_error(error) };
                }
                text
            };
            return Err(format!("{}: {message}", path.display()));
        }
        // SAFETY: a non-null table is valid until destroy.
        let table_ref = unsafe { &*table };
        if table_ref.abi_version != flatppl_bridge_abi::ABI_VERSION {
            let version = table_ref.abi_version;
            unsafe { (table_ref.destroy)(table_ref.state) };
            return Err(format!(
                "{}: ABI version {version}, expected {}",
                path.display(),
                flatppl_bridge_abi::ABI_VERSION
            ));
        }
        // SAFETY: describe returns a NUL-terminated string valid until destroy, or NULL.
        let text = unsafe { (table_ref.describe)(table_ref.state) };
        if text.is_null() {
            unsafe { (table_ref.destroy)(table_ref.state) };
            return Err(format!("{}: describe failed", path.display()));
        }
        let describe = unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned();
        Ok(CPlugin {
            table,
            describe,
            _library: std::mem::ManuallyDrop::new(library),
        })
    }

    fn table(&self) -> &FpbPlugin {
        // SAFETY: the table is valid until drop.
        unsafe { &*self.table }
    }
}

impl Drop for CPlugin {
    fn drop(&mut self) {
        let table = self.table();
        // SAFETY: destroy runs once, before the library unloads.
        unsafe { (table.destroy)(table.state) };
    }
}

impl Plugin for CPlugin {
    fn describe(&self) -> Result<String, Error> {
        Ok(self.describe.clone())
    }

    fn call(&self, call: Call) -> Result<Output, Error> {
        let export = |array: &Option<arrow_array::ArrayRef>| {
            array.as_ref().map(FpbArray::export).transpose()
        };
        let mut input = export(&call.input)?;
        let mut point = export(&call.point)?;
        let mut cotangent = export(&call.derivative.as_ref().and_then(|d| d.cotangent.clone()))?;
        let member = CString::new(call.member.as_str())
            .map_err(|_| Error::Invalid("member name has a NUL".into()))?;
        let (rng_kind, rng): (u32, &[u8]) = match &call.rng {
            None => (RNG_NONE, &[]),
            Some(Rng::Seed(b)) => (RNG_SEED, b),
            Some(Rng::State(b)) => (RNG_STATE, b),
        };
        let (deriv, wrt) = match &call.derivative {
            None => (DERIV_NONE, 0),
            Some(d) => (
                if d.cotangent.is_some() {
                    DERIV_VJP
                } else {
                    DERIV_GRADIENT
                },
                (if d.wrt_input { WRT_INPUT } else { 0 })
                    | (if d.wrt_point { WRT_POINT } else { 0 }),
            ),
        };
        let ptr = |a: &mut Option<FpbArray>| {
            a.as_mut()
                .map_or(std::ptr::null_mut(), |a| a as *mut FpbArray)
        };
        let c_call = FpbCall {
            op: call.op.code(),
            member: member.as_ptr(),
            rows: call.rows as i64,
            input: ptr(&mut input),
            point: ptr(&mut point),
            sizes: call.sizes.as_ptr(),
            n_sizes: call.sizes.len(),
            rng_kind,
            rng: rng.as_ptr(),
            rng_len: rng.len(),
            deriv,
            wrt,
            cotangent: ptr(&mut cotangent),
        };
        let table = self.table();
        let mut out = FpbResult::empty();
        // SAFETY: the call and result follow the header. Inputs left in place drop below.
        let code = unsafe { (table.call)(table.state, &c_call, &mut out) };
        let take = |a: &mut FpbArray| std::mem::replace(a, FpbArray::empty()).import();
        let result = if code == OK {
            let value = take(&mut out.value);
            let d_input = take(&mut out.d_input);
            let d_point = take(&mut out.d_point);
            let rng = (!out.rng.is_null()).then(|| {
                // SAFETY: the plugin set `rng_len` bytes at `rng`, valid until release.
                unsafe { std::slice::from_raw_parts(out.rng, out.rng_len) }.to_vec()
            });
            (|| {
                Ok(Output {
                    value: value?
                        .ok_or_else(|| Error::Failure("the plugin returned no value".into()))?,
                    d_input: d_input?,
                    d_point: d_point?,
                    rng,
                })
            })()
        } else {
            let message = if out.error.is_null() {
                "the plugin failed without a message".to_string()
            } else {
                // SAFETY: the plugin set a NUL-terminated message, valid until release.
                unsafe { CStr::from_ptr(out.error) }
                    .to_string_lossy()
                    .into_owned()
            };
            // Arrays a failing plugin filled anyway are released on drop.
            Err(Error::from_code(code, message))
        };
        // SAFETY: release frees what the plugin allocated for this result.
        unsafe { (table.release)(table.state, &mut out) };
        result
    }
}

/// Load a plugin from `path` with `config`.
pub fn load(path: impl AsRef<Path>, config: &str) -> Result<CPlugin, String> {
    CPlugin::load(path.as_ref(), config)
}
