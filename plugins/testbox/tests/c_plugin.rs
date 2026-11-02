//! The C header matches the Rust ABI: a plugin written in C serves a query.

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use flatppl_bridge::store::Store;
use flatppl_bridge::{Bridge, Request};
use std::path::PathBuf;
use std::sync::Arc;

#[test]
fn a_c_plugin_answers_a_query() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("libstdnormal_c.so");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = std::process::Command::new(cc)
        .args([
            "-std=c11",
            "-D_DEFAULT_SOURCE",
            "-Wall",
            "-Werror",
            "-shared",
            "-fPIC",
            "-I",
        ])
        .arg(root.join("include"))
        .arg(root.join("examples/c-plugin/stdnormal.c"))
        .args(["-lm", "-o"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let plugin = flatppl_bridge::cplugin::load(&out, "").unwrap();
    let mut bridge = Bridge::new(Store::memory(1 << 20), 0);
    bridge.add_box("c", Arc::new(plugin)).unwrap();
    let request = Request {
        query:
            "c = load_module(\"box:c\")\nresult = builtin_logdensityof.(c.stdnormal, [0.0, 1.0])\n"
                .into(),
        ..Request::default()
    };
    let reply = bridge.query(&request).unwrap();
    let mut reader =
        arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(reply[0].1.clone()), None)
            .unwrap();
    let batch = reader.next().unwrap().unwrap();
    let values = batch
        .column(0)
        .as_primitive::<Float64Type>()
        .values()
        .to_vec();
    // scipy.stats.norm.logpdf([0, 1])
    assert_eq!(values, vec![-0.9189385332046727, -1.4189385332046727]);
    let request = Request {
        query: "c = load_module(\"box:c\")\nresult = builtin_logdensityof(c.stdnormal, true)\n"
            .into(),
        ..Request::default()
    };
    assert_eq!(bridge.query(&request).unwrap_err().status(), 400);
}
