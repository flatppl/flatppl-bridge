//! Memoization: an identical query reaches the plugin once.

use arrow_array::{ArrayRef, Float64Array};
use flatppl_bridge::abi::{Call, Error, Output, Plugin};
use flatppl_bridge::store::{Store, sha256_hex};
use flatppl_bridge::value::{Value, encode_part};
use flatppl_bridge::{Bridge, Request};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting(Arc<AtomicUsize>);

impl Plugin for Counting {
    fn describe(&self) -> Result<String, Error> {
        Ok("f = record(kind = \"function\", queries = [\"apply\"], input = anything, output = anything)\n".into())
    }

    fn call(&self, call: Call) -> Result<Output, Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Output {
            value: call.input.unwrap(),
            d_input: None,
            d_point: None,
            rng: None,
        })
    }
}

fn bridge(cache: usize) -> (Bridge, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let mut bridge = Bridge::new(Store::memory(1 << 20), cache);
    bridge
        .add_box("b", Arc::new(Counting(count.clone())))
        .unwrap();
    (bridge, count)
}

fn part() -> Vec<u8> {
    let data: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
    encode_part(&Value::Array {
        shape: vec![2],
        data,
    })
    .unwrap()
}

fn request(source: &str, parts: HashMap<String, Vec<u8>>) -> Request {
    Request {
        query: format!(
            "bb = load_module(\"box:b\")\nx = load_data(\"{source}\", cartpow(reals, 2))\nresult = bb.f(x)\n"
        ),
        options: None,
        parts,
    }
}

#[test]
fn identical_queries_hit_the_cache() {
    let (bridge, count) = bridge(1 << 20);
    let by_part = request("part:x", HashMap::from([("x".to_owned(), part())]));
    let first = bridge.query(&by_part).unwrap();
    let second = bridge.query(&by_part).unwrap();
    assert_eq!(first, second);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    // A hash reference to the same bytes is the same query.
    let by_hash = request(&format!("sha256:{}", sha256_hex(&part())), HashMap::new());
    assert_eq!(bridge.query(&by_hash).unwrap(), first);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn a_disabled_cache_calls_every_time() {
    let (bridge, count) = bridge(0);
    let by_part = request("part:x", HashMap::from([("x".to_owned(), part())]));
    bridge.query(&by_part).unwrap();
    bridge.query(&by_part).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 2);
}
