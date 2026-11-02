//! The BridgeStan plugin through the C ABI. Needs compiled models, so the tests
//! are ignored by default. Run them with:
//!
//!   build-model.sh tests/models/normal.stan DIR
//!   build-model.sh tests/models/scalar.stan DIR
//!   FPB_BRIDGESTAN_MODELS=DIR cargo test -p fpb-bridgestan -- --ignored
//!
//! Expected numbers come from scipy.stats and from BridgeStan called directly.

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array};
use flatppl_bridge::store::Store;
use flatppl_bridge::value::{Value, encode_part, flatten};
use flatppl_bridge::{Bridge, Request};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::Arc;

/// The plugin library, built first: `cargo test` builds no cdylib.
fn built_library(package: &str, file: &str) -> PathBuf {
    static BUILT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    BUILT.get_or_init(|| {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
        let mut command = std::process::Command::new(cargo);
        command.args(["build", "-q", "-p", package]);
        if dir.file_name().is_some_and(|n| n == "release") {
            command.arg("--release");
        }
        assert!(
            command.status().unwrap().success(),
            "cargo build -p {package} failed"
        );
    });
    let name = if cfg!(target_os = "macos") {
        format!("lib{file}.dylib")
    } else {
        format!("lib{file}.so")
    };
    dir.join(name)
}

fn plugin_library() -> PathBuf {
    built_library("fpb-bridgestan", "fpb_bridgestan")
}

fn model(name: &str) -> String {
    let dir = std::env::var("FPB_BRIDGESTAN_MODELS")
        .expect("set FPB_BRIDGESTAN_MODELS to the compiled model directory");
    format!("{dir}/{name}_model.so")
}

fn bridge(name: &str) -> Bridge {
    let config = format!("{{\"model\": \"{}\"}}", model(name));
    let plugin = flatppl_bridge::cplugin::load(plugin_library(), &config).unwrap();
    let mut bridge = Bridge::new(Store::memory(1 << 20), 0);
    bridge.add_box("stan", Arc::new(plugin)).unwrap();
    bridge
}

fn part(shape: Vec<usize>, values: Vec<f64>) -> Vec<u8> {
    encode_part(&Value::Array {
        shape,
        data: Arc::new(Float64Array::from(values)),
    })
    .unwrap()
}

fn reals(bytes: &[u8]) -> Vec<f64> {
    let reader = arrow_ipc_reader(bytes);
    let (_, flat) = flatten(&reader);
    flat.as_primitive::<Float64Type>().values().to_vec()
}

fn arrow_ipc_reader(bytes: &[u8]) -> ArrayRef {
    let mut reader =
        arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes.to_vec()), None)
            .unwrap();
    reader.next().unwrap().unwrap().column(0).clone()
}

fn query(
    bridge: &Bridge,
    text: &str,
    options: Option<&str>,
    parts: Vec<(&str, Vec<u8>)>,
) -> HashMap<String, Vec<f64>> {
    let request = Request {
        query: format!("stan = load_module(\"box:stan\")\n{text}"),
        options: options.map(|o| o.as_bytes().to_vec()),
        parts: parts.into_iter().map(|(n, b)| (n.to_owned(), b)).collect(),
    };
    let reply = bridge.query(&request).unwrap();
    reply.iter().map(|(n, b)| (n.clone(), reals(b))).collect()
}

/// BridgeStan called directly, without the plugin.
fn direct(name: &str, theta: &[f64]) -> (f64, Vec<f64>) {
    type Construct = unsafe extern "C" fn(*const c_char, u32, *mut *mut c_char) -> *mut c_void;
    type Gradient = unsafe extern "C" fn(
        *const c_void,
        bool,
        bool,
        *const f64,
        *mut f64,
        *mut f64,
        *mut *mut c_char,
    ) -> c_int;
    unsafe {
        let library = libloading::Library::new(model(name)).unwrap();
        let construct = library.get::<Construct>(b"bs_model_construct").unwrap();
        let gradient = library.get::<Gradient>(b"bs_log_density_gradient").unwrap();
        let mut error = std::ptr::null_mut();
        let m = construct(c"".as_ptr(), 0, &mut error);
        assert!(!m.is_null());
        let (mut value, mut grad) = (0.0, vec![0.0; theta.len()]);
        assert_eq!(
            gradient(
                m,
                false,
                true,
                theta.as_ptr(),
                &mut value,
                grad.as_mut_ptr(),
                &mut error
            ),
            0
        );
        (value, grad)
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + b.abs())
}

#[test]
#[ignore = "needs compiled BridgeStan models"]
fn log_density_and_gradient_match_scipy_and_bridgestan() {
    let bridge = bridge("normal");
    let text = "u = load_data(\"part:u\", cartpow(reals, 2))\nresult = builtin_logdensityof(stan.posterior, u)\n";
    let options = r#"{"derivative": {"kind": "gradient", "wrt": ["u"]}}"#;
    let reply = query(
        &bridge,
        text,
        Some(options),
        vec![("u", part(vec![2], vec![0.3, 0.2]))],
    );
    // norm.logpdf(0.3, 1.5, 2) + norm.logpdf(0.2): the lognormal prior with its Jacobian.
    assert!(close(reply["value"][0], -2.7310242469692905));
    assert!(close(reply["d_u"][0], 0.3) && close(reply["d_u"][1], -0.2));
    let (value, grad) = direct("normal", &[0.3, 0.2]);
    assert_eq!(reply["value"][0], value);
    assert_eq!(reply["d_u"], grad);
}

#[test]
#[ignore = "needs compiled BridgeStan models"]
fn constrain_applies_the_transform() {
    let bridge = bridge("normal");
    let reply = query(
        &bridge,
        "result = stan.constrain([0.3, 0.2])\n",
        None,
        vec![],
    );
    assert_eq!(reply["value"][0], 0.3);
    assert!(close(reply["value"][1], 1.2214027581601699));
}

#[test]
#[ignore = "needs compiled BridgeStan models"]
fn broadcast_over_a_batch_of_theta_vectors() {
    let bridge = bridge("normal");
    let thetas = [0.3, 0.2, -1.0, 0.5, 2.0, -0.7];
    let text = "thetas = load_data(\"part:thetas\", cartpow(cartpow(reals, 2), 3))\nresult = builtin_logdensityof.(stan.posterior, thetas)\n";
    let options = r#"{"derivative": {"kind": "gradient", "wrt": ["thetas"]}}"#;
    let reply = query(
        &bridge,
        text,
        Some(options),
        vec![("thetas", part(vec![3, 2], thetas.to_vec()))],
    );
    assert_eq!(reply["value"].len(), 3);
    for (row, theta) in thetas.chunks(2).enumerate() {
        let (value, grad) = direct("normal", theta);
        assert_eq!(reply["value"][row], value);
        assert_eq!(reply["d_thetas"][2 * row..][..2], grad);
    }
}

#[test]
#[ignore = "needs compiled BridgeStan models"]
fn broadcast_over_one_parameter_vectors() {
    let bridge = bridge("scalar");
    let text = "us = load_data(\"part:us\", cartpow(cartpow(reals, 1), 3))\nresult = builtin_logdensityof.(stan.posterior, us)\n";
    let options = r#"{"derivative": {"kind": "gradient", "wrt": ["us"]}}"#;
    let reply = query(
        &bridge,
        text,
        Some(options),
        vec![("us", part(vec![3, 1], vec![-1.0, 0.0, 1.0]))],
    );
    // expon.logpdf(exp(u), scale = 0.5) + u, and its derivative 1 - 2 exp(u).
    let value = [-1.0426117017829393, -1.3068528194400546, -3.743416476358145];
    let grad = [0.26424111765711533, -1.0, -4.43656365691809];
    assert!(reply["value"].iter().zip(value).all(|(&a, b)| close(a, b)));
    assert!(reply["d_us"].iter().zip(grad).all(|(&a, b)| close(a, b)));
}

#[test]
#[ignore = "needs compiled BridgeStan models"]
fn refusals() {
    let bridge = bridge("normal");
    let status = |text: &str| {
        let request = Request {
            query: format!("stan = load_module(\"box:stan\")\n{text}"),
            ..Request::default()
        };
        bridge.query(&request).unwrap_err().status()
    };
    assert_eq!(
        status("result = builtin_logdensityof(stan.posterior, [1.0, 2.0, 3.0])\n"),
        400
    );
    assert_eq!(
        status("result = builtin_sample(rnginit(1), stan.posterior)\n"),
        422
    );
    // A matrix broadcasts to scalars, which are not points of the posterior.
    let request = Request {
        query: "stan = load_module(\"box:stan\")\nthetas = load_data(\"part:thetas\", cartpow(reals, [3, 2]))\nresult = builtin_logdensityof.(stan.posterior, thetas)\n".into(),
        parts: [("thetas".to_owned(), part(vec![3, 2], vec![0.0; 6]))].into(),
        ..Request::default()
    };
    assert_eq!(bridge.query(&request).unwrap_err().status(), 400);
}
