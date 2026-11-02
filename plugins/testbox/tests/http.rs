//! The protocol end to end: an HTTP server with the test box loaded through the C ABI.
//! Expected numbers come from scipy.stats.norm, not from the box.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, ListArray, RecordBatch};
use arrow_schema::{Field, Schema};
use flatppl_bridge::Bridge;
use flatppl_bridge::server::Config;
use flatppl_bridge::store::{Store, sha256_hex};
use flatppl_bridge::value::{Value, encode_part, flatten, nest, struct_array};
use reqwest::blocking::{Client, multipart};
use std::path::PathBuf;
use std::sync::Arc;

const LOG_N: f64 = -1.7920857137646178; // norm.logpdf(0.3, 1.5, 2.0)
const CDF_N: f64 = 0.2742531177500736; // norm.cdf(0.3, 1.5, 2.0)

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

fn library() -> PathBuf {
    built_library("fpb-testbox", "fpb_testbox")
}

struct Server {
    url: String,
    client: Client,
    token: Option<String>,
}

fn start(token: Option<&str>) -> Server {
    let plugin = flatppl_bridge::cplugin::load(library(), "").expect("load the test box");
    let mut bridge = Bridge::new(Store::memory(1 << 26), 1 << 26);
    bridge.add_box("tb", Arc::new(plugin)).unwrap();
    let bridge = Arc::new(bridge);
    let config = Config {
        token: token.map(str::to_owned),
        ..Config::default()
    };
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(listener.local_addr().unwrap()).unwrap();
            flatppl_bridge::server::serve(listener, bridge, config)
                .await
                .unwrap();
        });
    });
    let addr = rx.recv().unwrap();
    Server {
        url: format!("http://{addr}"),
        client: Client::new(),
        token: token.map(str::to_owned),
    }
}

struct Reply {
    status: u16,
    body: Vec<u8>,
    content_type: String,
}

impl Reply {
    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap()
    }

    /// The named parts of a multipart/mixed reply.
    fn parts(&self) -> Vec<(String, Vec<u8>)> {
        assert_eq!(self.status, 200, "{}", String::from_utf8_lossy(&self.body));
        let boundary = self.content_type.split("boundary=").nth(1).unwrap();
        let delimiter = format!("--{boundary}");
        let body = &self.body;
        let mut parts = Vec::new();
        let mut starts: Vec<usize> = find_all(body, delimiter.as_bytes());
        let end = starts.pop().unwrap();
        assert_eq!(&body[end..], format!("{delimiter}--\r\n").as_bytes());
        starts.push(end);
        for pair in starts.windows(2) {
            let part = &body[pair[0] + delimiter.len() + 2..pair[1] - 2];
            let split = find_all(part, b"\r\n\r\n")[0];
            let headers = String::from_utf8_lossy(&part[..split]).to_string();
            let name = headers
                .split("name=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap()
                .to_owned();
            parts.push((name, part[split + 4..].to_vec()));
        }
        parts
    }

    fn part(&self, name: &str) -> RecordBatch {
        let parts = self.parts();
        let bytes = &parts
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no part {name}"))
            .1;
        let reader =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes.clone()), None)
                .unwrap();
        let schema = reader.schema();
        let batches: Vec<_> = reader.map(Result::unwrap).collect();
        arrow_select::concat::concat_batches(&schema, &batches).unwrap()
    }

    fn reals(&self, name: &str) -> Vec<f64> {
        let batch = self.part(name);
        let (_, flat) = flatten(batch.column(0));
        flat.as_primitive::<Float64Type>().values().to_vec()
    }

    /// An integer part, one vector per row (a single vector is one row).
    fn int_rows(&self, name: &str) -> Vec<Vec<i64>> {
        let batch = self.part(name);
        let (dims, flat) = flatten(batch.column(0));
        let flat = flat.as_primitive::<Int64Type>().values().to_vec();
        match dims[..] {
            [] => vec![flat],
            [len] => flat.chunks(len).map(<[i64]>::to_vec).collect(),
            _ => panic!("not a vector of vectors"),
        }
    }

    fn valueset(&self, name: &str) -> String {
        self.part(name).schema().metadata()["flatppl.valueset"].clone()
    }
}

fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    haystack
        .windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle)
        .map(|(i, _)| i)
        .collect()
}

impl Server {
    fn auth(
        &self,
        request: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        match &self.token {
            Some(t) => request.bearer_auth(t),
            None => request,
        }
    }

    fn query(&self, query: &str, options: Option<&str>, parts: &[(&str, Vec<u8>)]) -> Reply {
        let mut form = multipart::Form::new().part(
            "query",
            multipart::Part::text(query.to_owned())
                .mime_str("text/x-flatppl; charset=utf-8")
                .unwrap(),
        );
        if let Some(options) = options {
            form = form.part(
                "options",
                multipart::Part::text(options.to_owned())
                    .mime_str("application/json")
                    .unwrap(),
            );
        }
        for (name, bytes) in parts {
            form = form.part(
                name.to_string(),
                multipart::Part::bytes(bytes.clone())
                    .mime_str("application/vnd.apache.arrow.stream")
                    .unwrap(),
            );
        }
        let response = self
            .auth(self.client.post(format!("{}/v1/query", self.url)))
            .multipart(form)
            .send()
            .unwrap();
        reply(response)
    }

    fn put(&self, hex: &str, bytes: Vec<u8>) -> u16 {
        let request = self
            .client
            .put(format!("{}/v1/objects/sha256/{hex}", self.url))
            .body(bytes);
        self.auth(request).send().unwrap().status().as_u16()
    }

    fn get(&self, path: &str) -> Reply {
        reply(
            self.auth(self.client.get(format!("{}{path}", self.url)))
                .send()
                .unwrap(),
        )
    }
}

fn reply(response: reqwest::blocking::Response) -> Reply {
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default();
    Reply {
        status,
        body: response.bytes().unwrap().to_vec(),
        content_type,
    }
}

fn reals(shape: Vec<usize>, values: Vec<f64>) -> Vec<u8> {
    encode_part(&Value::Array {
        shape,
        data: Arc::new(Float64Array::from(values)),
    })
    .unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + b.abs())
}

const HEAD: &str = "tb = load_module(\"box:tb\")\n";

#[test]
fn boxes_and_describe() {
    let server = start(None);
    assert_eq!(server.get("/v1/boxes").body, b"tb\n");
    let describe = server.get("/v1/boxes/tb/describe");
    assert_eq!(describe.content_type, "application/json");
    let json = describe.json();
    assert_eq!(json["normal"]["kind"], "kernel");
    assert_eq!(
        json["normal"]["input"],
        "cartprod(mu = reals, sigma = posreals)"
    );
    assert_eq!(json["stdnormal2"]["variate"], "cartpow(reals, 2)");
    assert_eq!(json["square"]["queries"], serde_json::json!(["apply"]));
    assert_eq!(server.get("/v1/boxes/none/describe").status, 404);
}

#[test]
fn kernel_log_density_and_gradient() {
    let server = start(None);
    let q = format!(
        "{HEAD}mu = 1.5\nx = load_data(\"part:x\", reals)\nresult = builtin_logdensityof(tb.normal, (mu = mu, sigma = 2.0), x)\n"
    );
    let x = reals(vec![], vec![0.3]);
    let r = server.query(&q, None, &[("x", x.clone())]);
    assert!(close(r.reals("value")[0], LOG_N));
    let options = r#"{"derivative": {"kind": "gradient", "wrt": ["mu", "x"]}}"#;
    let r = server.query(&q, Some(options), &[("x", x)]);
    assert!(close(r.reals("d_mu")[0], -0.3));
    assert!(close(r.reals("d_x")[0], 0.3));
}

#[test]
fn transport() {
    let server = start(None);
    let q = format!("{HEAD}result = builtin_touniform(tb.normal, (mu = 1.5, sigma = 2.0), 0.3)\n");
    assert!(close(server.query(&q, None, &[]).reals("value")[0], CDF_N));
}

#[test]
fn broadcast_matrix_over_a_measure() {
    let server = start(None);
    let xs = vec![-1.0, 0.0, 0.5, 1.0, 2.0, -3.0];
    let expected = [
        -1.4189385332046727,
        -0.9189385332046727,
        -1.0439385332046727,
        -1.4189385332046727,
        -2.9189385332046727,
        -5.418938533204672,
    ];
    let q = format!(
        "{HEAD}xs = load_data(\"part:xs\", cartpow(reals, [2, 3]))\nresult = builtin_logdensityof.(tb.stdnormal, xs)\n"
    );
    let options = r#"{"derivative": {"kind": "gradient", "wrt": ["xs"]}}"#;
    let r = server.query(&q, Some(options), &[("xs", reals(vec![2, 3], xs.clone()))]);
    let value = r.reals("value");
    assert!(value.iter().zip(expected).all(|(&a, b)| close(a, b)));
    let batch = r.part("value");
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(
        batch.schema().metadata()["flatppl.valueset"],
        "cartpow(reals, [2, 3])"
    );
    let d: Vec<f64> = xs.iter().map(|x| -x).collect();
    assert_eq!(r.reals("d_xs"), d);
}

#[test]
fn broadcast_table_of_inputs_with_shared_scalar() {
    let server = start(None);
    let rows = struct_array(
        ["mu", "sigma"].into_iter(),
        vec![
            Arc::new(Float64Array::from(vec![1.5, 1.5, 0.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![2.0, 2.0, 1.0])) as ArrayRef,
        ],
        3,
    );
    let table = encode_part(&Value::Table(rows)).unwrap();
    let q = format!(
        "{HEAD}t = load_data(\"part:t\", cartpow(cartprod(mu = reals, sigma = posreals), 3))\nx = 0.3\nresult = broadcast(builtin_logdensityof, tb.normal, t, x)\n"
    );
    let options = r#"{"derivative": {"kind": "gradient", "wrt": ["x"]}}"#;
    let r = server.query(&q, Some(options), &[("t", table)]);
    let value = r.reals("value");
    assert!(close(value[0], LOG_N) && close(value[1], LOG_N));
    assert!(close(value[2], -0.9639385332046727));
    // The gradient of the sum: the shared x collects every row.
    assert!(close(r.reals("d_x")[0], 0.3 + 0.3 - 0.3));
}

#[test]
fn vjp_of_a_function() {
    let server = start(None);
    let q = format!(
        "{HEAD}a = load_data(\"part:a\", cartpow(reals, [2, 2]))\nct = load_data(\"part:ct\", cartpow(reals, [2, 2]))\nresult = tb.square(a)\n"
    );
    let options = r#"{"derivative": {"kind": "vjp", "wrt": ["a"], "cotangent": "ct"}}"#;
    let parts = [
        ("a", reals(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])),
        ("ct", reals(vec![2, 2], vec![1.0, 0.5, 0.0, -1.0])),
    ];
    let r = server.query(&q, Some(options), &parts);
    assert_eq!(r.reals("value"), vec![1.0, 4.0, 9.0, 16.0]);
    assert_eq!(r.reals("d_a"), vec![2.0, 2.0, 0.0, -8.0]);
    // Broadcast applies the function per element.
    let q = format!(
        "{HEAD}a = load_data(\"part:a\", cartpow(reals, [2, 2]))\nct = load_data(\"part:ct\", cartpow(reals, [2, 2]))\nresult = tb.square.(a)\n"
    );
    let r = server.query(&q, Some(options), &parts);
    assert_eq!(r.reals("d_a"), vec![2.0, 2.0, 0.0, -8.0]);
}

#[test]
fn sampling_threads_the_rng_state() {
    let server = start(None);
    let draw = |rng: &str| {
        server.query(&format!("{HEAD}rng = {rng}\nresult = builtin_sample(rng, tb.normal, (mu = 10.0, sigma = 0.1), 4, 2)\n"), None, &[])
    };
    let first = draw("rnginit(7)");
    let x = first.reals("value");
    assert_eq!(x.len(), 8);
    assert!(x.iter().all(|v| (v - 10.0).abs() < 1.0));
    assert_eq!(
        first.part("value").schema().metadata()["flatppl.valueset"],
        "cartpow(reals, [4, 2])"
    );
    let state: Vec<i64> = first
        .part("rngstate")
        .column(0)
        .as_primitive::<Int64Type>()
        .values()
        .to_vec();
    let literal: Vec<String> = state.iter().map(ToString::to_string).collect();
    let second = draw(&format!("rngstate([{}])", literal.join(", ")));
    assert_ne!(second.reals("value"), x);
    assert_eq!(
        draw(&format!("rngstate([{}])", literal.join(", "))).reals("value"),
        second.reals("value")
    );
    assert_eq!(draw("rnginit(7)").reals("value"), x);
    assert_eq!(draw("rngstate([1, 2, 3])").status, 422);
}

#[test]
fn upload_on_missing_hash_then_retry() {
    let server = start(None);
    let x = reals(vec![3], vec![0.0, 1.0, 2.0]);
    let hex = sha256_hex(&x);
    let q = format!(
        "{HEAD}xs = load_data(\"sha256:{hex}\", cartpow(reals, 3))\nresult = builtin_logdensityof.(tb.stdnormal, xs)\n"
    );
    let r = server.query(&q, None, &[]);
    assert_eq!(r.status, 404);
    assert_eq!(
        r.json()["missing"],
        serde_json::json!([format!("sha256:{hex}")])
    );
    assert_eq!(server.put(&"0".repeat(64), x.clone()), 400);
    assert_eq!(server.put(&hex, x.clone()), 201);
    assert_eq!(server.put(&hex, x), 200);
    assert_eq!(server.query(&q, None, &[]).status, 200);
}

#[test]
fn a_part_is_stored_under_its_hash() {
    let server = start(None);
    let x = reals(vec![], vec![0.5]);
    let hex = sha256_hex(&x);
    let q = format!(
        "{HEAD}x = load_data(\"part:x\", reals)\nresult = builtin_logdensityof(tb.stdnormal, x)\n"
    );
    assert_eq!(server.query(&q, None, &[("x", x)]).status, 200);
    let q = format!(
        "{HEAD}x = load_data(\"sha256:{hex}\", reals)\nresult = builtin_logdensityof(tb.stdnormal, x)\n"
    );
    assert_eq!(server.query(&q, None, &[]).status, 200);
}

#[test]
fn identical_queries_return_identical_bytes() {
    let server = start(None);
    let q = format!("{HEAD}rng = rnginit(3)\nresult = builtin_sample(rng, tb.stdnormal, 5)\n");
    let (a, b) = (server.query(&q, None, &[]), server.query(&q, None, &[]));
    assert_eq!(a.body, b.body);
    assert_eq!(a.content_type, b.content_type);
}

#[test]
fn degenerate_values_are_values() {
    let server = start(None);
    let q = format!("{HEAD}result = builtin_logdensityof(tb.stdnormal, inf)\n");
    assert_eq!(
        server.query(&q, None, &[]).reals("value")[0],
        f64::NEG_INFINITY
    );
}

#[test]
fn refusals() {
    let server = start(None);
    let status =
        |q: &str, options: Option<&str>| server.query(&format!("{HEAD}{q}"), options, &[]).status;
    // 400: malformed.
    assert_eq!(
        status("result = builtin_logdensityof(tb.stdnormal, x)\n", None),
        400
    );
    assert_eq!(
        status(
            "x = load_data(\"part:x\", reals)\nresult = builtin_logdensityof(tb.stdnormal, x)\n",
            None
        ),
        400
    );
    assert_eq!(
        status(
            "result = builtin_logdensityof(tb.stdnormal, 1.0)\n",
            Some("{\"x\": 1}")
        ),
        400
    );
    assert_eq!(
        status(
            "result = builtin_logdensityof(tb.stdnormal, 1.0, 2.0)\n",
            None
        ),
        400
    );
    assert_eq!(
        status(
            "result = builtin_logdensityof.(tb.normal, (mu = [0.0], sigma = 1.0), [1.0])\n",
            None
        ),
        400
    );
    // 422: unsupported.
    assert_eq!(
        status("result = builtin_logdensityof(tb.nothing, 1.0)\n", None),
        422
    );
    assert_eq!(
        status("result = builtin_sample(rnginit(1), tb.halfnormal)\n", None),
        422
    );
    assert_eq!(
        status("result = logdensityof(tb.stdnormal, 1.0)\n", None),
        422
    );
    assert_eq!(
        status(
            "result = builtin_logdensityof(normalize(tb.halfnormal), 1.0)\n",
            None
        ),
        422
    );
    assert_eq!(
        status(
            "result = builtin_tonormal(tb.normal, (mu = 0.0, sigma = 1.0), 1.0)\n",
            None
        ),
        422
    );
    let grad = Some(r#"{"derivative": {"kind": "gradient", "wrt": ["x"]}}"#);
    assert_eq!(
        status(
            "x = 1.0\nresult = builtin_logdensityof(tb.halfnormal, x)\n",
            grad
        ),
        422
    );
    assert_eq!(
        status(
            "x = 1.0\nresult = builtin_sample(rnginit(1), tb.stdnormal)\n",
            grad
        ),
        422
    );
    let other = server.query(
        "tb = load_module(\"box:other\")\nresult = builtin_logdensityof(tb.stdnormal, 1.0)\n",
        None,
        &[],
    );
    assert_eq!(other.status, 422);
    assert_eq!(other.content_type, "application/problem+json");
}

#[test]
fn bearer_token() {
    let server = start(Some("s3cret"));
    assert_eq!(server.get("/v1/boxes").status, 200);
    let anonymous = Client::new()
        .get(format!("{}/v1/boxes", server.url))
        .send()
        .unwrap();
    assert_eq!(anonymous.status().as_u16(), 401);
}

fn table(mu: Vec<f64>, sigma: Vec<f64>) -> Vec<u8> {
    let rows = mu.len();
    encode_part(&Value::Table(struct_array(
        ["mu", "sigma"].into_iter(),
        vec![
            Arc::new(Float64Array::from(mu)) as ArrayRef,
            Arc::new(Float64Array::from(sigma)) as ArrayRef,
        ],
        rows,
    )))
    .unwrap()
}

fn literal(state: &[i64]) -> String {
    let bytes: Vec<String> = state.iter().map(ToString::to_string).collect();
    format!("[{}]", bytes.join(", "))
}

#[test]
fn batched_sampling_equals_unbatched_draws_per_row() {
    let server = start(None);
    let t = table(vec![0.0, 10.0, -5.0], vec![1.0, 0.1, 2.0]);
    let head = format!(
        "{HEAD}t = load_data(\"part:t\", cartpow(cartprod(mu = reals, sigma = posreals), 3))\n"
    );
    let batched = server.query(
        &format!(
            "{head}rngs = rnginit.([11, 12, 13])\nresult = broadcast(builtin_sample, rngs, tb.normal, t, 4)\n"
        ),
        None,
        &[("t", t.clone())],
    );
    let value = batched.reals("value");
    assert_eq!(batched.valueset("value"), "cartpow(cartpow(reals, 4), 3)");
    let states = batched.int_rows("rngstate");
    assert_eq!(states.len(), 3);
    let rows = [(11, 0.0, 1.0), (12, 10.0, 0.1), (13, -5.0, 2.0)];
    let single = |rng: &str, mu: f64, sigma: f64| {
        server.query(
            &format!(
                "{HEAD}result = builtin_sample({rng}, tb.normal, (mu = {mu:?}, sigma = {sigma:?}), 4)\n"
            ),
            None,
            &[],
        )
    };
    for (i, &(seed, mu, sigma)) in rows.iter().enumerate() {
        let one = single(&format!("rnginit({seed})"), mu, sigma);
        assert_eq!(one.reals("value"), value[i * 4..][..4]);
        assert_eq!(one.int_rows("rngstate")[0], states[i]);
    }

    // Resume every row from the state the batched reply returned.
    let len = states[0].len();
    let s = encode_part(&Value::Array {
        shape: vec![3],
        data: nest(Arc::new(Int64Array::from(states.concat())), &[len]),
    })
    .unwrap();
    let resumed = server.query(
        &format!(
            "{head}s = load_data(\"part:s\", cartpow(cartpow(integers, {len}), 3))\nresult = broadcast(builtin_sample, rngstate.(s), tb.normal, t, 4)\n"
        ),
        None,
        &[("t", t), ("s", s)],
    );
    let value = resumed.reals("value");
    for (i, &(_, mu, sigma)) in rows.iter().enumerate() {
        let one = single(&format!("rngstate({})", literal(&states[i])), mu, sigma);
        assert_eq!(one.reals("value"), value[i * 4..][..4]);
    }

    // A measure member, and seeds given as byte vectors.
    let batched = server.query(
        &format!("{HEAD}result = broadcast(builtin_sample, rnginit.([[1, 2], [3, 4]]), tb.stdnormal, 3)\n"),
        None,
        &[],
    );
    let value = batched.reals("value");
    assert_eq!(batched.valueset("value"), "cartpow(cartpow(reals, 3), 2)");
    for (i, seed) in ["[1, 2]", "[3, 4]"].into_iter().enumerate() {
        let one = server.query(
            &format!("{HEAD}result = builtin_sample(rnginit({seed}), tb.stdnormal, 3)\n"),
            None,
            &[],
        );
        assert_eq!(one.reals("value"), value[i * 3..][..3]);
    }
}

#[test]
fn batched_sampling_refuses_a_shared_rng_state() {
    let server = start(None);
    let t = table(vec![0.0, 1.0, 2.0], vec![1.0, 1.0, 1.0]);
    let status = |q: &str| {
        let q = format!(
            "{HEAD}t = load_data(\"part:t\", cartpow(cartprod(mu = reals, sigma = posreals), 3))\n{q}"
        );
        server.query(&q, None, &[("t", t.clone())]).status
    };
    assert_eq!(
        status("result = broadcast(builtin_sample, rnginit(1), tb.normal, t)\n"),
        422
    );
    // A size-one array of states would expand to every row.
    assert_eq!(
        status("result = broadcast(builtin_sample, rnginit.([1]), tb.normal, t)\n"),
        422
    );
    assert_eq!(
        status("result = broadcast(builtin_sample, rnginit.([1, 2]), tb.normal, t)\n"),
        400
    );
    assert_eq!(
        status("result = builtin_sample(rnginit.([1, 2]), tb.stdnormal)\n"),
        400
    );
    assert_eq!(
        status("result = broadcast(builtin_sample, rnginit.([1, 2, 3]), tb.normal, t)\n"),
        200
    );
}

/// The standard normal log density of a 2-vector: norm.logpdf(a) + norm.logpdf(b).
fn lognormal2(a: f64, b: f64) -> f64 {
    const LN_2PI: f64 = 1.8378770664093453;
    -0.5 * (a * a + b * b) - LN_2PI
}

#[test]
fn a_batch_of_vectors_broadcasts_per_vector_and_a_matrix_per_scalar() {
    let server = start(None);
    let xs = vec![0.5, -1.0, 2.0, 0.0, 1.5, 3.0];
    let expected: Vec<f64> = xs.chunks(2).map(|v| lognormal2(v[0], v[1])).collect();
    let options = r#"{"derivative": {"kind": "gradient", "wrt": ["xs"]}}"#;
    let vectors = format!(
        "{HEAD}xs = load_data(\"part:xs\", cartpow(cartpow(reals, 2), 3))\nresult = builtin_logdensityof.(tb.stdnormal2, xs)\n"
    );

    // A FixedSizeList column.
    let r = server.query(
        &vectors,
        Some(options),
        &[("xs", reals(vec![3, 2], xs.clone()))],
    );
    assert_eq!(r.reals("value").len(), 3);
    assert!(
        r.reals("value")
            .iter()
            .zip(&expected)
            .all(|(&a, &b)| close(a, b))
    );
    assert_eq!(r.reals("d_xs"), xs.iter().map(|x| -x).collect::<Vec<_>>());
    assert_eq!(r.valueset("d_xs"), "cartpow(cartpow(reals, 2), 3)");

    // A List column.
    let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, false));
    let list = ListArray::new(
        field,
        arrow_buffer::OffsetBuffer::from_lengths([2, 2, 2]),
        Arc::new(Float64Array::from(xs.clone())),
        None,
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "xs",
        list.data_type().clone(),
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list)]).unwrap();
    let mut writer = arrow_ipc::writer::StreamWriter::try_new(Vec::new(), &schema).unwrap();
    writer.write(&batch).unwrap();
    let r = server.query(&vectors, None, &[("xs", writer.into_inner().unwrap())]);
    assert!(
        r.reals("value")
            .iter()
            .zip(&expected)
            .all(|(&a, &b)| close(a, b))
    );

    // A literal array of vectors.
    let r = server.query(
        &format!(
            "{HEAD}result = builtin_logdensityof.(tb.stdnormal2, [[0.5, -1.0], [2.0, 0.0]])\n"
        ),
        None,
        &[],
    );
    assert!(
        r.reals("value")
            .iter()
            .zip(&expected)
            .all(|(&a, &b)| close(a, b))
    );

    // A matrix broadcasts to its scalars: six rows, none of them a 2-vector.
    let matrix = |member: &str| {
        let q = format!(
            "{HEAD}xs = load_data(\"part:xs\", cartpow(reals, [3, 2]))\nresult = builtin_logdensityof.(tb.{member}, xs)\n"
        );
        server.query(&q, None, &[("xs", reals(vec![3, 2], xs.clone()))])
    };
    assert_eq!(matrix("stdnormal").reals("value").len(), 6);
    assert_eq!(matrix("stdnormal2").status, 400);
}

#[test]
fn the_core_checks_arguments_against_declared_sets() {
    let server = start(None);
    let query = |q: &str| server.query(&format!("{HEAD}{q}"), None, &[]);
    // testbox would return NaN for a negative sigma: the core refuses first.
    let r = query("result = builtin_logdensityof(tb.normal, (mu = 0.0, sigma = -1.0), 0.3)\n");
    assert_eq!(r.status, 400);
    assert!(r.json()["detail"].as_str().unwrap().contains("posreals"));
    for q in [
        "result = builtin_logdensityof(tb.normal, (mu = 0.0), 0.3)\n",
        "result = builtin_logdensityof(tb.stdnormal, true)\n",
        "result = builtin_logdensityof(tb.stdnormal2, [1.0, 2.0, 3.0])\n",
        "result = builtin_logdensityof(tb.stdnormal2, [true, false])\n",
        "result = builtin_sample(rnginit(1), tb.normal, (mu = 0.0, sigma = 0.0))\n",
    ] {
        assert_eq!(query(q).status, 400, "{q}");
    }
    let t = table(vec![0.0, 0.0], vec![1.0, -1.0]);
    let q = format!(
        "{HEAD}t = load_data(\"part:t\", cartpow(cartprod(mu = reals, sigma = reals), 2))\nresult = broadcast(builtin_logdensityof, tb.normal, t, 0.3)\n"
    );
    assert_eq!(server.query(&q, None, &[("t", t)]).status, 400);
    // An integer is a real (spec §03).
    let r = query("result = builtin_logdensityof(tb.stdnormal, 1)\n");
    assert!(close(r.reals("value")[0], -1.4189385332046727));
}

#[test]
fn a_broadcast_of_vector_results_is_an_array_of_vectors() {
    let server = start(None);
    let r = server.query(
        &format!("{HEAD}result = tb.pair.([1.0, 2.0, 3.0])\n"),
        None,
        &[],
    );
    assert_eq!(r.valueset("value"), "cartpow(cartpow(reals, 2), 3)");
    assert_eq!(r.part("value").num_rows(), 3);
    assert_eq!(r.reals("value"), vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
    // One call is the vector itself.
    let r = server.query(&format!("{HEAD}result = tb.pair(1.0)\n"), None, &[]);
    assert_eq!(r.valueset("value"), "cartpow(reals, 2)");
}

#[test]
fn batched_sampling_of_a_vector_kernel_gives_one_vector_per_row() {
    let server = start(None);
    let mus = "[[0.0, 1.0], [10.0, 20.0], [-5.0, 5.0]]";
    let batched = server.query(
        &format!(
            "{HEAD}result = broadcast(builtin_sample, rnginit.([1, 2, 3]), tb.shifted2, {mus})\n"
        ),
        None,
        &[],
    );
    assert_eq!(batched.valueset("value"), "cartpow(cartpow(reals, 2), 3)");
    assert_eq!(batched.part("value").num_rows(), 3);
    let value = batched.reals("value");
    for (row, mu) in ["[0.0, 1.0]", "[10.0, 20.0]", "[-5.0, 5.0]"]
        .into_iter()
        .enumerate()
    {
        let one = server.query(
            &format!(
                "{HEAD}result = builtin_sample(rnginit({}), tb.shifted2, {mu})\n",
                row + 1
            ),
            None,
            &[],
        );
        assert_eq!(one.valueset("value"), "cartpow(reals, 2)");
        assert_eq!(one.reals("value"), value[2 * row..][..2]);
    }
    // With sizes, each row is an array of 4 vectors (spec §06 iid).
    let batched = server.query(
        &format!(
            "{HEAD}result = broadcast(builtin_sample, rnginit.([1, 2, 3]), tb.shifted2, {mus}, 4)\n"
        ),
        None,
        &[],
    );
    assert_eq!(
        batched.valueset("value"),
        "cartpow(cartpow(cartpow(reals, 2), 4), 3)"
    );
    let one = server.query(
        &format!("{HEAD}result = builtin_sample(rnginit(2), tb.shifted2, [10.0, 20.0], 4)\n"),
        None,
        &[],
    );
    assert_eq!(one.valueset("value"), "cartpow(cartpow(reals, 2), 4)");
    assert_eq!(one.reals("value"), batched.reals("value")[8..16]);
}
