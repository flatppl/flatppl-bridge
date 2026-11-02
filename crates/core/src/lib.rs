//! Core of the FlatPPL black-box server. See `docs/protocol.md`.

/// The protocol version, and the path prefix `/v1`.
pub const PROTOCOL_VERSION: u32 = 1;

/// Parse query text with the FlatPPL parser.
pub fn parse_query(text: &str) -> Result<flatppl_core::Module, String> {
    flatppl_syntax::parse(text).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_query() {
        let text = "bb = load_module(\"box:normal\")\nresult = builtin_logdensityof(bb.M, 1.0)\n";
        assert!(parse_query(text).is_ok());
    }
}
