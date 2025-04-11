uniffi::include_scaffolding!("strathweb-phi-engine");

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::engine::ConversationContext;
use crate::engine::ConversationMessage;
use crate::engine::InferenceOptions;
use crate::engine::InferenceOptionsBuilder;
use crate::engine::InferenceResult;
use crate::engine::PhiEngine;
use crate::engine::PhiEngineBuilder;
use crate::engine::PhiEventHandler;
use crate::engine::PhiModelProvider;
use crate::engine::Role;
use crate::engine::StatefulPhiEngine;
use crate::engine::TokenizerProvider;
use crate::engine::ChatFormat;

use once_cell::sync::Lazy;
use thiserror::Error;
use tracing::Level;
use tracing_subscriber::{filter::FilterFn, prelude::*};

pub mod engine;
pub mod text_generator;
pub mod token_stream;

static TRACING_INITIALIZED: Lazy<AtomicBool> = Lazy::new(|| AtomicBool::new(false));

pub fn enable_tracing() {
    if TRACING_INITIALIZED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        // mute the annoying warnings https://github.com/huggingface/tokenizers/issues/1366
        let filter = FilterFn::new(|metadata| {
            metadata.level() <= &Level::DEBUG
                && !metadata
                    .target()
                    .starts_with("tokenizers::tokenizer::serialization")
        });
        let layer = tracing_subscriber::fmt::layer()
            .with_level(true)
            .pretty()
            .with_filter(filter);

        tracing_subscriber::registry().with(layer).init();
    }
}

#[derive(Error, Debug)]
pub enum PhiError {
    #[error("LockingError with message: `{error_text}`")]
    LockingError { error_text: String },

    #[error("InitalizationError with message: `{error_text}`")]
    InitalizationError { error_text: String },

    #[error("InferenceError with message: `{error_text}`")]
    InferenceError { error_text: String },

    #[error("GPU is not supported on this architecture")]
    GpuNotSupported,
}

// candle does not support Metal on iOS yet
#[cfg(target_arch = "aarch64")]
#[cfg(target_os = "macos")]
const GPU_SUPPORTED: bool = true;

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
const GPU_SUPPORTED: bool = false;
