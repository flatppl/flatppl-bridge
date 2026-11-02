# flatppl-bridge

Serves black-box models to FlatPPL engines. A black box can be a log density, a sampler or an expensive function, implemented by BridgeStan, a compiled library, an executable or Python code.

A FlatPPL model declares a box as a module of `external(anything)` members and queries it with the usual `logdensityof`, `rand`, `~` and function application. The engine sends the box only primitive queries over HTTP, with data as Arrow IPC.

The server core is Rust. Adapters are C ABI plugins, and a Python layer sits on top.

## Layout

- `docs/protocol.md`: the HTTP and Arrow IPC protocol.
- `docs/plugin-abi.md`: the plugin C ABI. The header is `include/flatppl_bridge.h`.
- `crates/core`: the `flatppl-bridge` library: query checks, the content store, the reply cache, plugin loading and the HTTP binding.
- `crates/abi`: the `flatppl-bridge-abi` crate, for plugins written in Rust.
- `crates/server`: the `flatppl-bridge-server` binary.
- `plugins/bridgestan`: a Stan model through BridgeStan.
- `plugins/testbox`: closed-form test members.
- `examples/c-plugin`: a minimal plugin in C.
- `python`: the `flatppl_bridge` Python package, which serves Python callables as boxes. See `python/README.md`.

## Serve a Stan model

```sh
plugins/bridgestan/build-model.sh model.stan models/
cargo build --release -p flatppl-bridge-server -p fpb-bridgestan
target/release/flatppl-bridge-server \
  --box stan=target/release/libfpb_bridgestan.so \
  --config 'stan={"model": "models/model_model.so", "data": "data.json"}'
```

On macOS the plugin library ends in `.dylib`. `build-model.sh` uses BridgeStan at `$BRIDGESTAN`, or fetches it into `.bridgestan/` in this repository.

## Test

```sh
cargo test --workspace
for m in plugins/bridgestan/tests/models/*.stan; do plugins/bridgestan/build-model.sh "$m" target/bridgestan-models; done
FPB_BRIDGESTAN_MODELS=$PWD/target/bridgestan-models cargo test -p fpb-bridgestan -- --ignored
(cd python && pixi run -e test test)
```

Work in progress.
