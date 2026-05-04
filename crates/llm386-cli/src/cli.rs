//! Clap argument definitions for `llm386`.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "llm386",
    version,
    about = "LLM386 — context virtualization runtime"
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
    /// Create (or open) an LMDB store at the given path.
    Init {
        /// Path to the store directory.
        path: PathBuf,
    },

    /// Insert a block from a file (or `-` for stdin).
    Put {
        /// Path to the LMDB store.
        #[arg(long)]
        store: PathBuf,
        /// Session id (decimal, or `0x`-prefixed hex).
        #[arg(long, value_parser = parse_u128)]
        session: u128,
        /// Kind of block.
        #[arg(long, value_enum)]
        kind: KindArg,
        /// Priority in [0.0, 1.0]; higher is preferred during paging.
        #[arg(long, default_value_t = 0.0)]
        priority: f32,
        /// File to read; `-` for stdin.
        file: PathBuf,
    },

    /// List built-in model profiles.
    ListModels,

    /// Run the pager and print the resulting plan.
    Page {
        #[arg(long)]
        store: PathBuf,
        #[arg(long, value_parser = parse_u128)]
        session: u128,
        /// Built-in model profile name (see `list-models`).
        #[arg(long)]
        model: String,
        #[arg(long)]
        task: String,
    },

    /// Run page + pack and print the resulting prompt.
    Pack {
        #[arg(long)]
        store: PathBuf,
        #[arg(long, value_parser = parse_u128)]
        session: u128,
        #[arg(long)]
        model: String,
        #[arg(long)]
        task: String,
        /// Print only the rendered prompt (no header / manifest).
        #[arg(long)]
        prompt_only: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum KindArg {
    System,
    UserMessage,
    AssistantMessage,
    ToolResult,
    Summary,
    Fact,
    DocumentChunk,
    Plan,
    State,
    Trace,
}

impl From<KindArg> for llm386_core::BlockKind {
    fn from(k: KindArg) -> Self {
        match k {
            KindArg::System => Self::System,
            KindArg::UserMessage => Self::UserMessage,
            KindArg::AssistantMessage => Self::AssistantMessage,
            KindArg::ToolResult => Self::ToolResult,
            KindArg::Summary => Self::Summary,
            KindArg::Fact => Self::Fact,
            KindArg::DocumentChunk => Self::DocumentChunk,
            KindArg::Plan => Self::Plan,
            KindArg::State => Self::State,
            KindArg::Trace => Self::Trace,
        }
    }
}

fn parse_u128(s: &str) -> Result<u128, String> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u128::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        s.parse::<u128>().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_u128_decimal_and_hex() {
        assert_eq!(parse_u128("42").unwrap(), 42);
        assert_eq!(parse_u128("0xff").unwrap(), 255);
        assert_eq!(parse_u128("0XFF").unwrap(), 255);
        assert!(parse_u128("not-a-number").is_err());
    }
}
