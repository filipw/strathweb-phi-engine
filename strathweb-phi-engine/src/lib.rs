uniffi::include_scaffolding!("strathweb-phi-engine");

use crate::engine::BoxedPhiEventHandler;
use crate::engine::EngineOptions;
use crate::engine::InferenceOptions;
use crate::engine::InferenceResult;
use crate::engine::PhiEngine;
use crate::engine::PhiEventHandler;
use crate::engine::Role;
use crate::engine::HistoryEntry;
use thiserror::Error;

pub mod engine;
pub mod token_stream;

#[derive(Error, Debug)]
pub enum PhiError {
    #[error("InitalizationError with message: `{error_text}`")]
    InitalizationError { error_text: String },

    #[error("InferenceError with message: `{error_text}`")]
    InferenceError { error_text: String },

    #[error("History with message: `{error_text}`")]
    HistoryError { error_text: String },
}
