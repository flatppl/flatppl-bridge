//! The HTTP binding of `docs/protocol.md`.

use crate::bridge::{Bridge, Request};
use crate::error::BridgeError;
use crate::query::is_hash;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Multipart, Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use sha2::Digest;
use std::sync::Arc;

/// Server settings.
#[derive(Clone, Debug)]
pub struct Config {
    /// Require `Authorization: Bearer <token>` when set.
    pub token: Option<String>,
    /// The largest request body in bytes.
    pub body_limit: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            token: None,
            body_limit: 1 << 30,
        }
    }
}

/// The router for a bridge.
pub fn router(bridge: Arc<Bridge>, config: Config) -> Router {
    let token = config.token.clone();
    Router::new()
        .route("/v1/boxes", get(boxes))
        .route("/v1/boxes/{name}/describe", get(describe))
        .route("/v1/query", post(query))
        .route(
            "/v1/objects/sha256/{hex}",
            get(object_get).head(object_head).put(object_put),
        )
        .layer(DefaultBodyLimit::max(config.body_limit))
        .layer(middleware::from_fn(
            move |headers: HeaderMap, request, next: Next| {
                let token = token.clone();
                async move {
                    if let Some(token) = token {
                        let expected = format!("Bearer {token}");
                        let given = headers
                            .get(header::AUTHORIZATION)
                            .and_then(|v| v.to_str().ok());
                        if !given
                            .is_some_and(|g| constant_time_eq(g.as_bytes(), expected.as_bytes()))
                        {
                            let mut response = problem(
                                StatusCode::UNAUTHORIZED,
                                "Unauthorized",
                                "a bearer token is required",
                                None,
                            );
                            response.headers_mut().insert(
                                header::WWW_AUTHENTICATE,
                                HeaderValue::from_static("Bearer"),
                            );
                            return response;
                        }
                    }
                    next.run(request).await
                }
            },
        ))
        .with_state(bridge)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Serve the router on `listener` until the process ends.
pub async fn serve(
    listener: tokio::net::TcpListener,
    bridge: Arc<Bridge>,
    config: Config,
) -> std::io::Result<()> {
    axum::serve(listener, router(bridge, config)).await
}

fn problem(status: StatusCode, title: &str, detail: &str, missing: Option<&[String]>) -> Response {
    let mut body =
        serde_json::json!({ "title": title, "status": status.as_u16(), "detail": detail });
    if let Some(missing) = missing {
        body["missing"] = serde_json::json!(missing);
    }
    (
        status,
        [(header::CONTENT_TYPE, "application/problem+json")],
        body.to_string(),
    )
        .into_response()
}

fn error_response(error: &BridgeError) -> Response {
    let status = StatusCode::from_u16(error.status()).expect("valid status");
    let title = match error {
        BridgeError::BadRequest(_) => "Malformed request",
        BridgeError::Missing(_) => "Unknown hash",
        BridgeError::Unsupported(_) => "Unsupported query",
        BridgeError::Failure(_) => "Box failure",
    };
    let missing = match error {
        BridgeError::Missing(m) => Some(m.as_slice()),
        _ => None,
    };
    problem(status, title, &error.to_string(), missing)
}

async fn boxes(State(bridge): State<Arc<Bridge>>) -> Response {
    let mut text: String = bridge.box_names().map(|n| format!("{n}\n")).collect();
    if text.is_empty() {
        text.push('\n');
    }
    ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()
}

async fn describe(State(bridge): State<Arc<Bridge>>, Path(name): Path<String>) -> Response {
    match bridge.describe(&name) {
        Some(text) => (
            [(header::CONTENT_TYPE, "application/json")],
            text.to_owned(),
        )
            .into_response(),
        None => problem(
            StatusCode::NOT_FOUND,
            "Unknown box",
            &format!("no box `{name}`"),
            None,
        ),
    }
}

fn bad_hash(hex: &str) -> Option<Response> {
    (!is_hash(hex)).then(|| {
        problem(
            StatusCode::BAD_REQUEST,
            "Malformed request",
            "expected 64 lowercase hex digits",
            None,
        )
    })
}

async fn object_head(State(bridge): State<Arc<Bridge>>, Path(hex): Path<String>) -> Response {
    if let Some(r) = bad_hash(&hex) {
        return r;
    }
    if bridge.store.contains(&hex) {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
    .into_response()
}

async fn object_get(State(bridge): State<Arc<Bridge>>, Path(hex): Path<String>) -> Response {
    if let Some(r) = bad_hash(&hex) {
        return r;
    }
    match bridge.store.get(&hex) {
        Some(bytes) => (
            [(header::CONTENT_TYPE, "application/octet-stream")],
            bytes.as_ref().clone(),
        )
            .into_response(),
        None => problem(
            StatusCode::NOT_FOUND,
            "Unknown hash",
            &format!("no object sha256:{hex}"),
            None,
        ),
    }
}

async fn object_put(
    State(bridge): State<Arc<Bridge>>,
    Path(hex): Path<String>,
    body: Bytes,
) -> Response {
    if let Some(r) = bad_hash(&hex) {
        return r;
    }
    let actual = crate::store::sha256_hex(&body);
    if actual != hex {
        let detail = format!("the body hashes to sha256:{actual}, not sha256:{hex}");
        return problem(StatusCode::BAD_REQUEST, "Hash mismatch", &detail, None);
    }
    let stored = tokio::task::spawn_blocking(move || bridge.store.put(body.to_vec())).await;
    match stored {
        Ok(Ok((_, true))) => StatusCode::CREATED.into_response(),
        Ok(Ok((_, false))) => StatusCode::OK.into_response(),
        Ok(Err(e)) => problem(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Store failure",
            &e.to_string(),
            None,
        ),
        Err(e) => problem(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Store failure",
            &e.to_string(),
            None,
        ),
    }
}

async fn query(State(bridge): State<Arc<Bridge>>, mut multipart: Multipart) -> Response {
    let mut request = Request::default();
    let mut query = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => return error_response(&BridgeError::BadRequest(format!("multipart: {e}"))),
        };
        let Some(name) = field.name().map(str::to_owned) else {
            return error_response(&BridgeError::BadRequest("a part has no name".into()));
        };
        let bytes = match field.bytes().await {
            Ok(bytes) => bytes.to_vec(),
            Err(e) => return error_response(&BridgeError::BadRequest(format!("multipart: {e}"))),
        };
        let duplicate = match name.as_str() {
            "query" => query.replace(bytes).is_some(),
            "options" => request.options.replace(bytes).is_some(),
            _ => request.parts.insert(name.clone(), bytes).is_some(),
        };
        if duplicate {
            return error_response(&BridgeError::BadRequest(format!(
                "two parts named `{name}`"
            )));
        }
    }
    let Some(text) = query else {
        return error_response(&BridgeError::BadRequest(
            "the request has no `query` part".into(),
        ));
    };
    request.query = match String::from_utf8(text) {
        Ok(text) => text,
        Err(_) => return error_response(&BridgeError::BadRequest("the query is not UTF-8".into())),
    };
    let reply = tokio::task::spawn_blocking(move || bridge.query(&request)).await;
    match reply {
        Ok(Ok(parts)) => multipart_reply(parts),
        Ok(Err(error)) => error_response(&error),
        Err(e) => error_response(&BridgeError::Failure(format!("the box panicked: {e}"))),
    }
}

/// Large parts go out in chunks, so a client can read record batches early.
const CHUNK: usize = 1 << 20;

fn multipart_reply(parts: crate::bridge::Reply) -> Response {
    // A boundary from the content hash keeps cached replies byte-identical.
    let mut hasher = sha2::Sha256::new();
    for (name, bytes) in parts.iter() {
        hasher.update(name.as_bytes());
        hasher.update(bytes);
    }
    let digest = hasher.finalize();
    let boundary = format!(
        "fpb-{}",
        digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    let content_type = format!("multipart/mixed; boundary={boundary}");
    let stream = async_stream(parts, boundary);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from_stream(stream))
        .expect("valid response")
}

fn async_stream(
    parts: crate::bridge::Reply,
    boundary: String,
) -> impl futures_util::Stream<Item = Result<Bytes, std::convert::Infallible>> {
    let mut chunks: Vec<Bytes> = Vec::new();
    for (name, bytes) in parts.iter() {
        chunks.push(Bytes::from(format!(
            "--{boundary}\r\nContent-Disposition: attachment; name=\"{name}\"\r\n\
             Content-Type: application/vnd.apache.arrow.stream\r\n\r\n"
        )));
        for chunk in bytes.chunks(CHUNK) {
            chunks.push(Bytes::copy_from_slice(chunk));
        }
        chunks.push(Bytes::from_static(b"\r\n"));
    }
    chunks.push(Bytes::from(format!("--{boundary}--\r\n")));
    futures_util::stream::iter(chunks.into_iter().map(Ok))
}
