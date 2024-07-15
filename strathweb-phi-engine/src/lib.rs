uniffi::include_scaffolding!("strathweb-phi-engine");

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::engine::BoxedPhiEventHandler;
use crate::engine::PhiEngineBuilder;
use crate::engine::InferenceOptions;
use crate::engine::InferenceOptionsBuilder;
use crate::engine::InferenceResult;
use crate::engine::PhiEngine;
use crate::engine::PhiEventHandler;
use crate::engine::Role;
use crate::engine::HistoryEntry;
use crate::engine::TokenizerProvider;
use crate::engine::PhiModelProvider;

use once_cell::sync::Lazy;
use thiserror::Error;

pub mod engine;
pub mod token_stream;
pub mod text_generator;

static TRACING_INITIALIZED: Lazy<AtomicBool> = Lazy::new(|| AtomicBool::new(false));

pub fn enable_tracing() {
    if TRACING_INITIALIZED.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_level(true)
            .pretty()
            .init();
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

    #[error("History with message: `{error_text}`")]
    HistoryError { error_text: String },
}
