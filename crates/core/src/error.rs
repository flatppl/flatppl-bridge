use std::fmt;

/// A query failure, by HTTP status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BridgeError {
    /// 400: a malformed request.
    BadRequest(String),
    /// 404: objects missing from the store, as `sha256:<hex>`.
    Missing(Vec<String>),
    /// 422: an unsupported query.
    Unsupported(String),
    /// 500: a box failure.
    Failure(String),
}

impl BridgeError {
    pub fn status(&self) -> u16 {
        match self {
            BridgeError::BadRequest(_) => 400,
            BridgeError::Missing(_) => 404,
            BridgeError::Unsupported(_) => 422,
            BridgeError::Failure(_) => 500,
        }
    }
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::BadRequest(m) | BridgeError::Unsupported(m) | BridgeError::Failure(m) => {
                f.write_str(m)
            }
            BridgeError::Missing(hashes) => write!(f, "missing objects: {}", hashes.join(", ")),
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<flatppl_bridge_abi::Error> for BridgeError {
    fn from(error: flatppl_bridge_abi::Error) -> Self {
        match error {
            flatppl_bridge_abi::Error::Invalid(m) => BridgeError::BadRequest(m),
            flatppl_bridge_abi::Error::Unsupported(m) => BridgeError::Unsupported(m),
            flatppl_bridge_abi::Error::Failure(m) => BridgeError::Failure(m),
        }
    }
}

pub(crate) fn bad(message: impl Into<String>) -> BridgeError {
    BridgeError::BadRequest(message.into())
}

pub(crate) fn unsupported(message: impl Into<String>) -> BridgeError {
    BridgeError::Unsupported(message.into())
}
