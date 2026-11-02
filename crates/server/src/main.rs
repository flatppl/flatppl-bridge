//! `flatppl-bridge-server`: serve plugin boxes over HTTP. See `docs/protocol.md`.

use clap::Parser;
use flatppl_bridge::Bridge;
use flatppl_bridge::server::{Config, router};
use flatppl_bridge::store::Store;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(version, about = "Serve FlatPPL black boxes over HTTP and Arrow IPC.")]
struct Args {
    /// The address to listen on.
    #[arg(long, default_value = "127.0.0.1:8737")]
    listen: std::net::SocketAddr,
    /// A box as NAME=LIBRARY, a plugin library that exports fpb_plugin_init_v1. Repeat for more boxes.
    #[arg(long = "box", value_name = "NAME=LIBRARY", required = true)]
    boxes: Vec<String>,
    /// The configuration string of a box, as NAME=CONFIG.
    #[arg(long = "config", value_name = "NAME=CONFIG")]
    configs: Vec<String>,
    /// Keep uploaded objects in this directory. Without it, objects live in memory only.
    #[arg(long)]
    store: Option<PathBuf>,
    /// The in-memory object store size in bytes.
    #[arg(long, default_value_t = 1 << 30)]
    store_bytes: usize,
    /// The reply cache size in bytes. 0 turns memoization off.
    #[arg(long, default_value_t = 1 << 28)]
    cache_bytes: usize,
    /// The largest request body in bytes.
    #[arg(long, default_value_t = 1 << 30)]
    body_limit: usize,
    /// Require this bearer token. Prefer the FPB_TOKEN variable, which stays out of process listings.
    #[arg(long, env = "FPB_TOKEN", hide_env_values = true)]
    token: Option<String>,
    /// A PEM certificate chain. Serves HTTPS together with --tls-key.
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<PathBuf>,
    /// A PEM private key.
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<PathBuf>,
}

fn split(spec: &str, what: &str) -> Result<(String, String), String> {
    spec.split_once('=')
        .map(|(n, v)| (n.to_owned(), v.to_owned()))
        .ok_or_else(|| format!("{what} `{spec}` is not NAME=VALUE"))
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();
    let store = match &args.store {
        Some(dir) => {
            Store::directory(dir.clone(), args.store_bytes).map_err(|e| format!("store: {e}"))?
        }
        None => Store::memory(args.store_bytes),
    };
    let mut configs = std::collections::HashMap::new();
    for spec in &args.configs {
        let (name, config) = split(spec, "--config")?;
        configs.insert(name, config);
    }
    let mut bridge = Bridge::new(store, args.cache_bytes);
    for spec in &args.boxes {
        let (name, library) = split(spec, "--box")?;
        let config = configs.remove(&name).unwrap_or_default();
        let plugin = flatppl_bridge::cplugin::load(&library, &config)?;
        bridge.add_box(&name, Arc::new(plugin))?;
    }
    if let Some(name) = configs.keys().next() {
        return Err(format!("--config names `{name}`, which no --box serves"));
    }
    let app = router(
        Arc::new(bridge),
        Config {
            token: args.token,
            body_limit: args.body_limit,
        },
    );
    match (args.tls_cert, args.tls_key) {
        (Some(cert), Some(key)) => {
            let _ = rustls::crypto::ring::default_provider().install_default();
            let tls = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key)
                .await
                .map_err(|e| format!("TLS: {e}"))?;
            eprintln!("serving https://{}", args.listen);
            axum_server::bind_rustls(args.listen, tls)
                .serve(app.into_make_service())
                .await
                .map_err(|e| e.to_string())
        }
        _ => {
            let listener = tokio::net::TcpListener::bind(args.listen)
                .await
                .map_err(|e| e.to_string())?;
            eprintln!(
                "serving http://{}",
                listener.local_addr().map_err(|e| e.to_string())?
            );
            axum::serve(listener, app).await.map_err(|e| e.to_string())
        }
    }
}
