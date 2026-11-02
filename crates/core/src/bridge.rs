//! The server state: boxes, the object store and the reply cache.

use crate::describe::{self, Member};
use crate::error::{BridgeError, bad, unsupported};
use crate::exec::{self, Options};
use crate::query::{self, Source};
use crate::store::{ReplyCache, Store, sha256_hex};
use crate::value::{Value, decode_part, encode_part};
use flatppl_bridge_abi::Plugin;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// A box: a plugin and its parsed describe records.
pub struct BoxEntry {
    pub plugin: Arc<dyn Plugin>,
    pub describe: String,
    pub members: BTreeMap<String, Member>,
}

/// A decoded query request.
#[derive(Clone, Debug, Default)]
pub struct Request {
    pub query: String,
    pub options: Option<Vec<u8>>,
    pub parts: HashMap<String, Vec<u8>>,
}

pub use crate::store::Parts as Reply;

pub struct Bridge {
    boxes: BTreeMap<String, BoxEntry>,
    pub store: Store,
    cache: Option<ReplyCache>,
}

impl Bridge {
    /// A bridge with `store`, and a reply cache of at most `cache_bytes` (0 turns it off).
    pub fn new(store: Store, cache_bytes: usize) -> Bridge {
        Bridge {
            boxes: BTreeMap::new(),
            store,
            cache: (cache_bytes > 0).then(|| ReplyCache::new(cache_bytes)),
        }
    }

    /// Serve `plugin` as box `name`.
    pub fn add_box(&mut self, name: &str, plugin: Arc<dyn Plugin>) -> Result<(), String> {
        let valid = name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !valid {
            return Err(format!("bad box name `{name}`"));
        }
        let describe = plugin
            .describe()
            .map_err(|e| format!("box `{name}`: {e}"))?;
        let members = describe::parse(&describe).map_err(|e| format!("box `{name}`: {e}"))?;
        self.boxes.insert(
            name.to_owned(),
            BoxEntry {
                plugin,
                describe,
                members,
            },
        );
        Ok(())
    }

    pub fn box_names(&self) -> impl Iterator<Item = &str> {
        self.boxes.keys().map(String::as_str)
    }

    pub fn describe(&self, name: &str) -> Option<&str> {
        self.boxes.get(name).map(|b| b.describe.as_str())
    }

    /// Answer a query. Blocks while the plugin runs.
    pub fn query(&self, request: &Request) -> Result<Reply, BridgeError> {
        let query = query::parse(&request.query)?;
        let options = match &request.options {
            Some(bytes) => Options::from_json(bytes)?,
            None => Options::default(),
        };
        let entry = self
            .boxes
            .get(&query.box_name)
            .ok_or_else(|| unsupported(format!("no box `{}` on this server", query.box_name)))?;

        let mut hashes = HashMap::new();
        for (name, bytes) in &request.parts {
            let (hex, _) = self
                .store
                .put(bytes.clone())
                .map_err(|e| BridgeError::Failure(format!("store: {e}")))?;
            hashes.insert(name.clone(), hex);
        }
        let mut bytes = HashMap::new();
        let mut missing = Vec::new();
        for (binding, (source, _)) in &query.data {
            match source {
                Source::Part(part) => {
                    let data = request.parts.get(part).ok_or_else(|| {
                        bad(format!(
                            "`{binding}` reads part `{part}`, which the request lacks"
                        ))
                    })?;
                    bytes.insert(binding.clone(), Arc::new(data.clone()));
                }
                Source::Hash(hex) => match self.store.get(hex) {
                    Some(data) => {
                        bytes.insert(binding.clone(), data);
                    }
                    None => missing.push(format!("sha256:{hex}")),
                },
            }
        }
        if !missing.is_empty() {
            missing.sort();
            missing.dedup();
            return Err(BridgeError::Missing(missing));
        }

        let key = self
            .cache
            .as_ref()
            .map(|_| cache_key(&query.box_name, &request.query, &hashes, &options));
        if let (Some(cache), Some(key)) = (&self.cache, &key)
            && let Some(reply) = cache.get(key)
        {
            return Ok(reply);
        }

        let mut values: HashMap<String, Value> = query.literals.clone().into_iter().collect();
        for (binding, (_, set)) in &query.data {
            let value = decode_part(&bytes[binding], set).map_err(|e| match e {
                BridgeError::BadRequest(m) => bad(format!("`{binding}`: {m}")),
                other => other,
            })?;
            values.insert(binding.clone(), value);
        }
        let parts = exec::execute(
            &query,
            &entry.members,
            &values,
            &options,
            entry.plugin.as_ref(),
        )?;
        let reply: Reply = Arc::new(
            parts
                .iter()
                .map(|(name, value)| Ok((name.clone(), encode_part(value)?)))
                .collect::<Result<_, BridgeError>>()?,
        );
        if let (Some(cache), Some(key)) = (&self.cache, key) {
            cache.insert(key, reply.clone());
        }
        Ok(reply)
    }
}

/// The cache key: the box, the canonical query with parts named by hash, and the options.
fn cache_key(
    box_name: &str,
    text: &str,
    hashes: &HashMap<String, String>,
    options: &Options,
) -> String {
    let mut text = text.to_owned();
    for (part, hex) in hashes {
        text = text.replace(&format!("\"part:{part}\""), &format!("\"sha256:{hex}\""));
    }
    let canonical = flatppl_syntax::parse(&text)
        .map(|m| flatppl_syntax::print(&m))
        .unwrap_or(text);
    sha256_hex(format!("{box_name}\0{canonical}\0{}", options.canonical()).as_bytes())
}
