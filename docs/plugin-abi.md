# flatppl-bridge plugin ABI, version 1

A plugin is a shared library that implements one box. The server loads it, reads its describe JSON, and passes it decoded calls. A plugin never parses FlatPPL queries, so a closed plugin keeps working when the query language grows.

The C declarations are in `include/flatppl_bridge.h`. Rust plugins implement the `Plugin` trait of the `flatppl-bridge-abi` crate and call `export_plugin!`, which generates the C entry points.

## Entry points

A plugin library exports two functions:

```c
const fpb_plugin* fpb_plugin_init_v1(const char* config, char** error);
void fpb_plugin_free_error_v1(char* error);
```

`config` is a plugin-defined string, for example JSON. On failure, `fpb_plugin_init_v1` returns NULL and sets `*error` to a message. The server frees the message with `fpb_plugin_free_error_v1`.

The server never unloads a plugin library, because library threads and static destructors can outlive `dlclose`.

## The function table

`fpb_plugin` holds `abi_version` (1), an opaque `state`, and four functions:

| Function | Contract |
|---|---|
| `describe(state)` | Return the describe JSON (see `docs/protocol.md`), NUL-terminated and valid until `destroy`. Return NULL on failure. |
| `call(state, call, out)` | Answer one decoded call. Fill `out` and return a status code. |
| `release(state, out)` | Free what `call` allocated in `out`: `rng`, `error` and `private_data`. |
| `destroy(state)` | Free the plugin. The server calls it once, after the last call. |

`call` may run on several threads at once. A plugin that is not thread-safe must lock.

## Calls

`fpb_call` carries one query:

| Field | Meaning |
|---|---|
| `op` | `FPB_OP_APPLY`, `FPB_OP_LOGDENSITY`, `FPB_OP_SAMPLE` or a transport. |
| `member` | The member name. |
| `rows` | The number of rows: 1 for a single query, the element count of a broadcast. |
| `input` | The function or kernel input, `rows` values. NULL for a measure member. |
| `point` | The point of a log density or transport, `rows` values. NULL otherwise. |
| `sizes`, `n_sizes` | The sample sizes of `FPB_OP_SAMPLE`. Empty for one draw. |
| `rng_kind`, `rng`, `rng_len` | `FPB_RNG_SEED` with seed bytes, `FPB_RNG_STATE` with states the plugin returned before, or `FPB_RNG_NONE`. `rng` holds one state per row, `rng_len` bytes each, row-major. |
| `deriv` | `FPB_DERIV_NONE`, `FPB_DERIV_GRADIENT` or `FPB_DERIV_VJP`. |
| `wrt` | `FPB_WRT_INPUT` and `FPB_WRT_POINT`: which derivatives to return. |
| `cotangent` | For a VJP: `rows` values of the output shape. |

Arrays use the Arrow C Data Interface with the element encoding of `docs/protocol.md`: one array element is one FlatPPL value. A real is `Float64` or `Float32`, an integer `Int64`, a boolean `Boolean`, an array of shape `[d1, ..., dk]` is `k` nested `FixedSizeList`s, and a record is a `Struct`.

The server owns the input arrays. A plugin may move an input out: it copies the struct and sets the original's `release` to NULL. The server releases every input left in place after `call` returns.

## What the core guarantees

The core parses the describe JSON once, at load time, and refuses a box whose JSON or value sets are invalid. Before each call, it guarantees:

- The member exists and declares the operation, and the derivative if one is requested.
- Each row of `input` lies in the declared `input` set, and for `FPB_OP_LOGDENSITY`, `FPB_OP_TOUNIFORM` and `FPB_OP_TONORMAL` each row of `point` lies in the `variate` set. The check covers shape, integer versus real, and interval bounds. A mismatch never reaches the plugin: the client gets 400.
- Values arrive in the canonical encoding of their set: an integer as `Int64`, an integer given for a real set as `Float64`, an array as nested `FixedSizeList`s, and a record as a `Struct` with the fields in the order of the set.
- `FPB_OP_SAMPLE` carries `rows` RNG states of one kind and length.

A set of `anything` turns the check off for that argument. A plugin may check more, for example a constraint that no value set expresses, and returns `FPB_INVALID` for a mismatch.

## Results

The server passes `out` zeroed. On success the plugin sets:

| Field | Content |
|---|---|
| `value` | `rows` values. For a sample: per row the product of `sizes` variates, row-major, or one variate if `sizes` is empty. |
| `d_input` | If `wrt` has `FPB_WRT_INPUT`: `rows` values with the input's encoding. |
| `d_point` | If `wrt` has `FPB_WRT_POINT`: `rows` values with the point's encoding. |
| `rng`, `rng_len` | For a sample: the new RNG state of each row, `rng_len` bytes each, row-major. |

Derivatives are per row:

- Gradient: row `i` holds the gradient of the scalar `value[i]` with respect to row `i` of the input or point.
- VJP: row `i` holds the product of `cotangent[i]` with the Jacobian of row `i`.

The server sums rows that share a source value, for example a scalar held constant in a broadcast. So the reply carries the derivative of the whole query.

The server takes the result arrays and releases them. It then calls `release` once. Each `value` element of `FPB_OP_APPLY` must lie in the declared `output` set, and each sampled variate in the `variate` set. Otherwise the query fails with 500.

## Status codes

| Code | HTTP status | Meaning |
|---|---|---|
| `FPB_OK` | 200 | Success. |
| `FPB_INVALID` | 400 | Bad input values that the declared sets do not catch. |
| `FPB_UNSUPPORTED` | 422 | An unsupported query, or a foreign RNG state. |
| `FPB_FAILURE` | 500 | A box failure. |

On failure, the plugin sets `error` to a NUL-terminated message. A degenerate result is a value with `FPB_OK`: a log density outside the support is `-inf`.

## RNG state

The plugin defines its RNG algorithm and state encoding. The state bytes must identify the algorithm, so that the plugin can refuse a foreign state with `FPB_UNSUPPORTED`. A seed `n` given as an integer arrives as its 8-byte little-endian encoding (spec §07 `rnginit`). Equal states and equal calls must give equal samples. Row `i` of a sample draws from state `i` alone, so that it equals a one-row call with that state. All new states of one call have the same length.

## Plugins in this repository

- `plugins/bridgestan`: a Stan model through BridgeStan. Configuration and members are documented in its `src/lib.rs`. Build a model with `plugins/bridgestan/build-model.sh MODEL.stan OUTDIR`.
- `plugins/testbox`: closed-form test members.
