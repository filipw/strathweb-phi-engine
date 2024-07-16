use anyhow::{Error as E, Result};
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Repo;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tracing::debug;

use crate::text_generator::TextGenerator;
use crate::PhiError;

#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub role: Role,
    pub text: String,
}

#[derive(Debug, Clone)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub token_count: u16,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: u16,
    pub seed: u64,
}

pub struct InferenceOptionsBuilder {
    inner: Mutex<InferenceOptions>,
}

impl InferenceOptionsBuilder {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(InferenceOptions {
                token_count: 128,
                temperature: 0.7,
                top_p: None,
                top_k: None,
                repeat_penalty: 1.0,
                repeat_last_n: 64,
                seed: 146628346,
            }),
        }
    }

    pub fn with_token_count(&self, token_count: u16) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.token_count = token_count;
        Ok(())
    }

    pub fn with_temperature(&self, temperature: f64) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.temperature = temperature;
        Ok(())
    }

    pub fn with_top_p(&self, top_p: f64) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.top_p = Some(top_p);
        Ok(())
    }

    pub fn with_top_k(&self, top_k: u64) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.top_k = Some(top_k);
        Ok(())
    }

    pub fn with_repeat_penalty(&self, repeat_penalty: f32) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.repeat_penalty = repeat_penalty;
        Ok(())
    }

    pub fn with_repeat_last_n(&self, repeat_last_n: u16) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.repeat_last_n = repeat_last_n;
        Ok(())
    }

    pub fn with_seed(&self, seed: u64) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.seed = seed;
        Ok(())
    }

    pub fn build(&self) -> Result<InferenceOptions, PhiError> {
        let inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        Ok(inner.clone())
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub token_count: u16,
    pub result_text: String,
    pub duration: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub cache_dir: String,
    pub model_provider: PhiModelProvider,
    pub tokenizer_provider: TokenizerProvider,
    pub use_flash_attention: bool,
    pub context_window: Option<u16>,
}

pub trait PhiEventHandler: Send + Sync {
    fn on_model_loaded(&self) -> Result<(), PhiError>;
    fn on_inference_token(&self, token: String) -> Result<(), PhiError>;
}

pub struct BoxedPhiEventHandler {
    pub handler: Box<dyn PhiEventHandler>,
}

impl BoxedPhiEventHandler {
    pub fn new(handler: Box<dyn PhiEventHandler>) -> Self {
        Self { handler }
    }
}

pub struct PhiEngineBuilder {
    inner: Mutex<PhiEngineBuilderInner>,
}

impl PhiEngineBuilder {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(PhiEngineBuilderInner::new()),
        }
    }

    pub fn with_context_window(&self, context_window: u16) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.context_window = Some(context_window);
        Ok(())
    }

    pub fn with_flash_attention(&self, use_flash_attention: bool) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.use_flash_attention = use_flash_attention;
        Ok(())
    }

    pub fn with_event_handler(
        &self,
        event_handler: Arc<BoxedPhiEventHandler>,
    ) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.event_handler = Some(event_handler);
        Ok(())
    }

    pub fn with_model_provider(&self, model_provider: PhiModelProvider) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.model_provider = model_provider;
        Ok(())
    }

    pub fn with_tokenizer_provider(
        &self,
        tokenizer_provider: TokenizerProvider,
    ) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.tokenizer_provider = tokenizer_provider;
        Ok(())
    }

    pub fn build(&self, cache_dir: String) -> Result<Arc<PhiEngine>, PhiError> {
        let inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        let engine_options = EngineOptions {
            cache_dir: cache_dir,
            model_provider: inner.model_provider.clone(),
            tokenizer_provider: inner.tokenizer_provider.clone(),
            use_flash_attention: inner.use_flash_attention,
            context_window: inner.context_window.clone(),
        };
        PhiEngine::new(engine_options, inner.event_handler.clone()).map(|engine| Arc::new(engine))
    }

    pub fn build_stateful(
        &self,
        cache_dir: String,
        system_instruction: Option<String>,
    ) -> Result<Arc<StatefulPhiEngine>, PhiError> {
        let inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        let engine_options = EngineOptions {
            cache_dir: cache_dir,
            model_provider: inner.model_provider.clone(),
            tokenizer_provider: inner.tokenizer_provider.clone(),
            use_flash_attention: inner.use_flash_attention,
            context_window: inner.context_window.clone(),
        };

        let conversation_state: ConversationState = ConversationState {
            system_instruction: system_instruction,
            messages: Vec::new(),
        };

        let engine = PhiEngine::new(engine_options, inner.event_handler.clone())?;
        Ok(Arc::new(StatefulPhiEngine {
            engine: engine,
            conversation_state: Mutex::new(conversation_state),
        }))
    }
}

struct PhiEngineBuilderInner {
    context_window: Option<u16>,
    tokenizer_provider: TokenizerProvider,
    model_provider: PhiModelProvider,
    use_flash_attention: bool,
    event_handler: Option<Arc<BoxedPhiEventHandler>>,
}

impl PhiEngineBuilderInner {
    fn new() -> Self {
        Self {
            context_window: None,
            tokenizer_provider: TokenizerProvider::HuggingFace {
                tokenizer_repo: "microsoft/Phi-3-mini-4k-instruct".to_string(),
                tokenizer_file_name: "tokenizer.json".to_string(),
            },
            model_provider: PhiModelProvider::HuggingFace {
                model_repo: "microsoft/Phi-3-mini-4k-instruct-gguf".to_string(),
                model_file_name: "Phi-3-mini-4k-instruct-q4.gguf".to_string(),
                model_revision: "main".to_string(),
            },
            event_handler: None,
            use_flash_attention: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PhiModelProvider {
    HuggingFace {
        model_repo: String,
        model_file_name: String,
        model_revision: String,
    },
    FileSystem {
        model_path: String,
    },
}

#[derive(Debug, Clone)]
pub enum TokenizerProvider {
    HuggingFace {
        tokenizer_repo: String,
        tokenizer_file_name: String,
    },
    FileSystem {
        tokenizer_path: String,
    },
}

#[derive(Debug, Clone)]
pub struct ConversationState {
    pub system_instruction: Option<String>,
    pub messages: Vec<ConversationMessage>,
}

pub struct StatefulPhiEngine {
    pub engine: PhiEngine,
    pub conversation_state: Mutex<ConversationState>,
}

impl StatefulPhiEngine {
    pub fn run_inference(
        &self,
        prompt_text: String,
        inference_options: &InferenceOptions,
    ) -> Result<InferenceResult, PhiError> {
        let mut conversation_state =
            self.conversation_state
                .lock()
                .map_err(|e| PhiError::LockingError {
                    error_text: e.to_string(),
                })?;
        let result = self.engine
            .run_inference(prompt_text, conversation_state.clone(), inference_options).map_err(|e| PhiError::InferenceError {
                error_text: e.to_string(),
            })?;

        conversation_state.messages.push(ConversationMessage {
            role: Role::Assistant,
            text: result.result_text.clone(),
        });
        Ok(result)
    }

    pub fn clear_messsages(&self) -> Result<(), PhiError> {
        let mut conversation_state =
            self.conversation_state
                .lock()
                .map_err(|e| PhiError::LockingError {
                    error_text: e.to_string(),
                })?;
        conversation_state.messages.clear();
        Ok(())
    }

    pub fn get_history(&self) -> Result<Vec<ConversationMessage>, PhiError> {
        let conversation_state =
            self.conversation_state
                .lock()
                .map_err(|e| PhiError::LockingError {
                    error_text: e.to_string(),
                })?;
        Ok(conversation_state.messages.clone())
    }
}

pub struct PhiEngine {
    pub model: Phi3,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub event_handler: Option<Arc<BoxedPhiEventHandler>>,
    pub context_window: u16,
}

impl PhiEngine {
    pub fn new(
        engine_options: EngineOptions,
        event_handler: Option<Arc<BoxedPhiEventHandler>>,
    ) -> Result<Self, PhiError> {
        let start = std::time::Instant::now();
        // candle does not support Metal on iOS yet
        // this also requires building with features = ["metal"]
        //let device = Device::new_metal(0).unwrap();
        let device = Device::Cpu;

        let model_path = match engine_options.model_provider {
            PhiModelProvider::HuggingFace {
                model_repo,
                model_file_name,
                model_revision,
            } => {
                let api_builder = ApiBuilder::new()
                    .with_cache_dir(PathBuf::from(engine_options.cache_dir.clone()));
                let api = api_builder
                    .build()
                    .map_err(|e| PhiError::InitalizationError {
                        error_text: e.to_string(),
                    })?;
                let repo = Repo::with_revision(
                    model_repo.to_string(),
                    hf_hub::RepoType::Model,
                    model_revision,
                );
                let api = api.repo(repo);
                let model_path = api.get(model_file_name.as_str()).map_err(|e| {
                    PhiError::InitalizationError {
                        error_text: e.to_string(),
                    }
                })?;
                debug!(" --> Downloaded model to {:?}...", model_path);
                model_path
            }
            PhiModelProvider::FileSystem { model_path } => model_path.into(),
        };

        let tokenizer_path = match engine_options.tokenizer_provider {
            TokenizerProvider::HuggingFace {
                tokenizer_repo,
                tokenizer_file_name,
            } => {
                let api_builder = ApiBuilder::new()
                    .with_cache_dir(PathBuf::from(engine_options.cache_dir.clone()));
                let api = api_builder
                    .build()
                    .map_err(|e| PhiError::InitalizationError {
                        error_text: e.to_string(),
                    })?;
                let repo = Repo::with_revision(
                    tokenizer_repo.to_string(),
                    hf_hub::RepoType::Model,
                    "main".to_string(),
                );
                let api = api.repo(repo);
                let tokenizer_path = api.get(tokenizer_file_name.as_str()).map_err(|e| {
                    PhiError::InitalizationError {
                        error_text: e.to_string(),
                    }
                })?;
                debug!(" --> Downloaded tokenizer to {:?}...", tokenizer_path);
                tokenizer_path
            }
            TokenizerProvider::FileSystem { tokenizer_path } => tokenizer_path.into(),
        };

        // defaults
        let context_window = engine_options.context_window.unwrap_or(3800);

        let mut file = File::open(&model_path).map_err(|e| PhiError::InitalizationError {
            error_text: e.to_string(),
        })?;
        let model_content =
            gguf_file::Content::read(&mut file).map_err(|e| PhiError::InitalizationError {
                error_text: e.to_string(),
            })?;
        let model = Phi3::from_gguf(
            engine_options.use_flash_attention,
            model_content,
            &mut file,
            &device,
        )
        .map_err(|e| PhiError::InitalizationError {
            error_text: e.to_string(),
        })?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| PhiError::InitalizationError {
                error_text: e.to_string(),
            })?;

        let event_handler_clone = event_handler.clone();

        debug!(" --> Loaded the model in {:?}", start.elapsed());
        if let Some(event_handler) = event_handler {
            event_handler
                .handler
                .on_model_loaded()
                .map_err(|e| PhiError::InitalizationError {
                    error_text: e.to_string(),
                })?;
        }

        Ok(Self {
            model: model,
            tokenizer: tokenizer,
            device: device,
            event_handler: event_handler_clone,
            context_window: context_window,
        })
    }

    pub fn run_inference(
        &self,
        prompt_text: String,
        conversation_state: ConversationState,
        inference_options: &InferenceOptions,
    ) -> Result<InferenceResult, PhiError> {
        let mut history = conversation_state.messages.clone();
        self.trim_history_to_token_limit(&mut history, self.context_window);
        history.push(ConversationMessage {
            role: Role::User,
            text: prompt_text.clone(),
        });

        let history_prompt = history
            .iter()
            .map(|entry| {
                let role = match entry.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };
                format!("\n<|{}|>{}<|end|>", role, entry.text)
            })
            .collect::<String>();

        let system_instruction = conversation_state.system_instruction.unwrap_or("You are a helpful assistant that answers user questions. Be short and direct in your answers.".to_string());

        // Phi-3 has no system prompt so we inject it as a user prompt
        let prompt_with_history = format!("<|user|>\nYour overall instructions are: {}<|end|>\n<|assistant|>Understood, I will adhere to these instructions<|end|>{}\n<|assistant|>\n", system_instruction, history_prompt);

        let mut pipeline = TextGenerator::new(
            &self.model,
            self.tokenizer.clone(),
            inference_options,
            &self.device,
            self.event_handler.clone(),
        );

        let response = pipeline
            .run(&prompt_with_history, inference_options.token_count)
            .map_err(|e: E| PhiError::InferenceError {
                error_text: e.to_string(),
            })?;

        Ok(response)
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer.encode(text, true).unwrap().get_ids().len()
    }

    fn trim_history_to_token_limit(
        &self,
        history: &mut Vec<ConversationMessage>,
        token_limit: u16,
    ) {
        let mut token_count = history
            .iter()
            .map(|entry| self.count_tokens(&entry.text))
            .sum::<usize>();

        while token_count > token_limit as usize {
            let front = history.remove(0);
            token_count -= self.count_tokens(&front.text);
        }
    }
}
