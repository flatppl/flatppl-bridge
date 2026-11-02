//! Core of the FlatPPL black-box server. See `docs/protocol.md` and
//! `docs/plugin-abi.md`.

pub mod bridge;
pub mod cplugin;
pub mod describe;
pub mod error;
pub mod exec;
pub mod query;
pub mod server;
pub mod set;
pub mod store;
mod syntax;
pub mod value;

pub use bridge::{Bridge, Reply, Request};
pub use error::BridgeError;
pub use flatppl_bridge_abi as abi;

/// The protocol version, and the path prefix `/v1`.
pub const PROTOCOL_VERSION: u32 = 1;
