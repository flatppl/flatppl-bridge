# flatppl-bridge protocol, version 1

This document defines how a FlatPPL engine (the client) queries a black-box server (the box server) over HTTP. A box server hosts one or more boxes. A box is a FlatPPL module whose members are functions, kernels or measures that the server implements.

Status: draft. The protocol moves to flatppl-design as a versioned document after milestone 1.

## Overview

- The client sends each query as a multipart HTTP POST.
- The query is FlatPPL text in the black-box query profile.
- Data parts and reply parts are Arrow IPC streams.
- Large values are content-addressed. The client uploads a value once and then refers to it by its SHA-256 hash.
- Every query is pure. A sampling query carries an explicit RNG state. The server may answer an identical query from its cache.

All paths are under the prefix `/v1`. A server that changes this protocol incompatibly uses a new prefix.

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /v1/boxes` | List the boxes, one name per line (`text/plain`). |
| `GET /v1/boxes/{box}/describe` | Describe the members of a box (FlatPPL text). |
| `POST /v1/query` | Evaluate a query (multipart). |
| `HEAD /v1/objects/sha256/{hex}` | Check whether the store holds an object: 200 or 404. |
| `PUT /v1/objects/sha256/{hex}` | Upload an object: 201, or 200 if the store holds it already. |

`{hex}` is the lowercase hexadecimal SHA-256 of the object bytes, 64 characters. Box names match `[A-Za-z_][A-Za-z0-9_]*`.

## Values in Arrow

A FlatPPL value maps to one element of an Arrow array:

| FlatPPL value | Arrow type |
|---|---|
| real | `Float64` or `Float32` |
| integer | `Int64` (a request may also use any other integer type) |
| boolean | `Boolean` |
| array of shape `[d1, ..., dk]` | `k` nested `FixedSizeList`s, outermost of size `d1`, row-major |
| record | `Struct`, one child per field, in field order |

The protocol does not require 64-bit floats. A box answers in the precision it computes. A client converts if it needs another precision.

### Part layout

A data part holds one value as an Arrow IPC stream. The layout follows `load_data` (spec §07) for Arrow sources:

- A scalar: one column, one row.
- A record: one row, one column per field.
- An array of shape `[n, d2, ..., dk]` of scalars: one column with `n` rows. Each row holds the sub-array of shape `[d2, ..., dk]`, or a scalar if `k = 1`.
- A table of `n` rows: one column per table column, `n` rows.

A single-column part may give its column any name. A part may split its rows over several record batches. A reply part carries the schema metadata key `flatppl.valueset`, whose value is the value set of the part as FlatPPL text, for example `cartpow(reals, [100, 3])`.

An Arrow IPC file (magic `ARROW1`) is accepted wherever a stream is.

## Query

### Request

`POST /v1/query` takes `multipart/form-data`. Each part has a `Content-Disposition` name.

| Part name | Content type | Content |
|---|---|---|
| `query` | `text/x-flatppl; charset=utf-8` | The query text. Required. |
| `options` | `application/json` | Request options. Optional. |
| any other name | `application/vnd.apache.arrow.stream` | A data part. |

A data part name matches `[A-Za-z0-9_.-]+`. A request may carry at most one part of each name.

### Query text

The query text is a FlatPPL module. Its bindings are restricted to the black-box query profile:

- **Box.** Exactly one binding `B = load_module("box:<name>")`, where `<name>` is a box on this server. The alias `B` is free.
- **Data.** Any number of bindings `D = load_data(S, V)`. `S` is `"part:<name>"`, a data part of this request, or `"sha256:<hex>"`, an object in the store. `V` is the value set of the data, and must not be `anything` (spec §07 refuses access to such data). The server checks the data against `V`. A request part is also stored under its hash, so a later request may refer to it by hash.
- **Literals.** Any number of bindings to scalar literals, array literals of scalars, and record literals of these.
- **RNG.** Bindings `R = rnginit(seed)` and `R = rngstate(bytes)`, where `seed` is an integer or byte-vector literal and `bytes` is a byte-vector literal or a data binding.
- **Result.** Exactly one binding named `result`, with one of the forms below.

An argument is a binding name, a literal, or a record literal of these. `M` is a member of the box, written `B.M`.

| Result form | Meaning |
|---|---|
| `B.f(args...)` | Apply function member `f`. |
| `builtin_logdensityof(B.K, input, x)` | Log density of kernel member `K` at `x`. |
| `builtin_logdensityof(B.M, x)` | Log density of measure member `M` at `x`. |
| `builtin_sample(rng, B.K, input, n, ...)` | Draw from kernel member `K`. The sizes are optional. |
| `builtin_sample(rng, B.M, n, ...)` | Draw from measure member `M`. The sizes are optional. |
| `builtin_touniform(B.K, input, x)`, and the other three transports | Transport (spec §07), if the member supports it. |
| `broadcast(callee, args...)` | Batch one of the forms above. |

A measure member is a nullary kernel (spec §06). FlatPPL has no empty-record literal, so a measure member takes no `input` argument. The server resolves the arity from the member kind.

`broadcast` follows spec §04. `callee` is a function member, or one of `builtin_logdensityof` and the four transports. Array arguments are iterated elementwise, tables row-wise, and members and scalars are held constant. Size-one axes expand. `builtin_sample` is not batched in version 1, because a shared RNG state would correlate the rows.

A table is accepted only as a `broadcast` operand in version 1.

Version 1 refuses all other forms with 422. Examples are measure algebra on a member, a query on two boxes, and computation on data.

Example:

```flatppl
bb = load_module("box:normal")
theta = load_data("part:theta", cartpow(reals, 2))
x = load_data("sha256:6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d", cartpow(reals, 1000))
result = builtin_logdensityof(bb.K, theta, x)
```

### Options

`options` is a JSON object. Unknown keys are refused with 400.

```json
{"derivative": {"kind": "gradient", "wrt": ["theta"]}}
{"derivative": {"kind": "vjp", "wrt": ["theta"], "cotangent": "ct"}}
```

- `wrt` lists data or literal binding names of the query. The server returns one derivative per name.
- `gradient` returns the gradient of the sum of all result entries. For a scalar result this is the ordinary gradient. For a broadcast it is the per-row gradient for a batched argument, and the sum over rows for a held-constant argument.
- `vjp` returns the vector-Jacobian product with the cotangent. `cotangent` names a data binding of the query whose value has the shape of the result.
- A derivative of `builtin_sample`, a derivative the member does not support, or a derivative with respect to an integer or boolean binding is refused with 422.

### RNG state

- The box defines its RNG algorithm and the byte encoding of its state. The encoding must identify the algorithm, so that the box refuses a foreign state with 422.
- `rnginit(seed)` starts a fresh state. An integer seed `n` means the 8-byte little-endian encoding of `n` (spec §07).
- `rngstate(bytes)` resumes a state that a previous reply returned.
- A sampling reply carries the new state. The client passes it to the next sampling query. Equal states and equal queries give equal samples.

### Reply

A successful query returns 200 with `multipart/mixed`. Each part is an Arrow IPC stream in the part layout, named by its `Content-Disposition` name:

| Part name | When | Content |
|---|---|---|
| `value` | always | The result. For `builtin_sample`, the samples `X`. |
| `d_<name>` | for each `wrt` name | The derivative, with the shape of binding `<name>`. |
| `rngstate` | for `builtin_sample` | The new RNG state: a vector of integers in `0..255`. |

Parts come in this order. The `value` part may stream in several record batches, so a client can read a large result before the reply ends.

A degenerate mathematical result is a value, not an error. For example, a log density outside the support is `-inf`, and an undefined result is `NaN`.

### Errors

An error reply has a status code and an `application/problem+json` body (RFC 9457) with at least `title` and `detail`.

| Status | Meaning | Client action |
|---|---|---|
| 400 | Malformed request: a parse error, a missing `result`, a missing part, a bad option, data that does not match its value set. | Fix the request. |
| 404 | A `sha256:` reference is not in the store. The body lists every missing hash in `missing`. | Upload each object, then retry. |
| 422 | An unsupported query: unknown box or member, a form outside the profile, a query the member does not declare, an unsupported derivative, a foreign RNG state. | Do not retry. |
| 500 | The box failed. | Report the box failure. |

An unknown path also returns 404, without a `missing` field.

## Describe

`GET /v1/boxes/{box}/describe` returns `text/x-flatppl`. The text binds each member name to a record:

```flatppl
posterior = record(kind = "measure", queries = ["builtin_logdensityof"], gradient = true, vjp = false, variate = cartpow(reals, 2), normalized = false)
f = record(kind = "function", queries = ["apply"], gradient = true, vjp = true, input = cartpow(reals, 3), output = reals)
K = record(kind = "kernel", queries = ["builtin_logdensityof", "builtin_sample"], gradient = false, vjp = false, input = cartprod(mu = reals, sigma = posreals), variate = reals, normalized = true)
```

| Field | Members | Meaning |
|---|---|---|
| `kind` | all | `"function"`, `"kernel"` or `"measure"`. |
| `queries` | all | The supported query forms: `"apply"`, `"builtin_logdensityof"`, `"builtin_sample"`, and the transport names. |
| `gradient`, `vjp` | all | Whether the deterministic queries support this derivative. |
| `input` | functions, kernels | The value set of the input. |
| `output` | functions | The value set of the output. |
| `variate` | kernels, measures | The value set of the variate. |
| `normalized` | kernels, measures | `true` if every measure has total mass 1. |

A value set may be `anything`. `builtin_sample` is listed only for a normalized member, because only a probability measure can be sampled.

A function or kernel takes one input. An input with a record set takes keyword arguments, as in `B.f(a = x, b = y)` (spec §04 calling convention).

The describe text is a description, not a model. Its records hold strings and value sets, which FlatPPL values cannot hold. The client parses it with the FlatPPL parser.

## Store

The store holds objects by content hash. Objects never expire on content grounds, because a hash cannot go stale. A server may still evict objects, and then answers 404 for them.

`PUT /v1/objects/sha256/{hex}` takes the raw bytes (`application/octet-stream` or `application/vnd.apache.arrow.stream`). The server computes the SHA-256 of the body and refuses a mismatch with 400. A query may refer only to objects in the store and parts of the same request.

## Caching and memoization

A query is a pure function of its text, its data and its options. The server may cache replies. The cache key is the SHA-256 of the box name, the canonical query text with each `part:` source replaced by its `sha256:` source, and the canonical options. A cache hit returns the same bytes as the original reply.

## Security

- Serve over HTTPS outside a trusted network.
- Use standard HTTP authentication (the `Authorization` header). The server decides the scheme.
- The server verifies every upload against its hash.
- A query reads only uploaded objects and parts of its own request. A query never names a file path or a URL.
