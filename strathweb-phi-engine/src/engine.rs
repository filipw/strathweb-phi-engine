use anyhow::{Error as E, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use candle_transformers::models::quantized_phi3::ModelWeights as QuantizedPhi3;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Repo;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tracing::debug;

use crate::text_generator::TextGenerator;
use crate::{PhiError, GPU_SUPPORTED};

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

#[derive(Clone, Debug)]
pub enum ChatFormat {
    ChatML,     // Phi-4 style with <|im_start|>, <|im_sep|>, <|im_end|>
    Llama2,  // Phi-3 style with <|system|>, <|end|>, etc.
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
    pub chat_format: ChatFormat,
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
                chat_format: ChatFormat::Llama2,
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

    pub fn with_chat_format(&self, chat_format: ChatFormat) -> Result<(), PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.chat_format = chat_format;
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
    pub use_gpu: bool,
    pub dtype: Option<String>,
}

pub trait PhiEventHandler: Send + Sync {
    fn on_model_loaded(&self) -> Result<(), PhiError>;
    fn on_inference_started(&self) -> Result<(), PhiError>;
    fn on_inference_ended(&self) -> Result<(), PhiError>;
    fn on_inference_token(&self, token: String) -> Result<(), PhiError>;
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
        event_handler: Arc<dyn PhiEventHandler>,
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

    pub fn try_use_gpu(&self) -> Result<bool, PhiError> {
        let mut inner = self.inner.lock().map_err(|e| PhiError::LockingError {
            error_text: e.to_string(),
        })?;
        inner.use_gpu = GPU_SUPPORTED;
        Ok(GPU_SUPPORTED)
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
            use_gpu: inner.use_gpu,
            dtype: Some("bf16".to_string()),
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
            use_gpu: inner.use_gpu,
            dtype: Some("bf16".to_string()),
        };

        let conversation_context = ConversationContext {
            system_instruction: system_instruction,
            messages: Vec::new(),
        };

        let engine = PhiEngine::new(engine_options, inner.event_handler.clone())?;
        Ok(Arc::new(StatefulPhiEngine {
            engine: engine,
            conversation_context: Mutex::new(conversation_context),
        }))
    }
}

struct PhiEngineBuilderInner {
    context_window: Option<u16>,
    tokenizer_provider: TokenizerProvider,
    model_provider: PhiModelProvider,
    use_flash_attention: bool,
    event_handler: Option<Arc<dyn PhiEventHandler>>,
    use_gpu: bool,
}

impl PhiEngineBuilderInner {
    fn new() -> Self {
        Self {
            context_window: None,
            tokenizer_provider: TokenizerProvider::HuggingFace {
                tokenizer_repo: "microsoft/Phi-3-mini-4k-instruct".to_string(),
                tokenizer_file_name: "tokenizer.json".to_string(),
            },
            model_provider: PhiModelProvider::HuggingFaceGguf {
                model_repo: "microsoft/Phi-3-mini-4k-instruct-gguf".to_string(),
                model_file_name: "Phi-3-mini-4k-instruct-q4.gguf".to_string(),
                model_revision: "main".to_string(),
            },
            use_gpu: false,
            event_handler: None,
            use_flash_attention: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PhiModelProvider {
    HuggingFace {
        model_repo: String,
        model_revision: String,
    },
    HuggingFaceGguf {
        model_repo: String,
        model_file_name: String,
        model_revision: String,
    },
    FileSystem {
        index_path: String,
        config_path: String,
    },
    FileSystemGguf {
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
pub struct ConversationContext {
    pub system_instruction: Option<String>,
    pub messages: Vec<ConversationMessage>,
}

pub struct StatefulPhiEngine {
    pub engine: PhiEngine,
    pub conversation_context: Mutex<ConversationContext>,
}

impl StatefulPhiEngine {
    pub fn run_inference(
        &self,
        prompt_text: &str,
        inference_options: &InferenceOptions,
    ) -> Result<InferenceResult, PhiError> {
        let mut conversation_context =
            self.conversation_context
                .lock()
                .map_err(|e| PhiError::LockingError {
                    error_text: e.to_string(),
                })?;
        let result = self
            .engine
            .run_inference(prompt_text, &conversation_context, inference_options)
            .map_err(|e| PhiError::InferenceError {
                error_text: e.to_string(),
            })?;

        conversation_context.messages.push(ConversationMessage {
            role: Role::Assistant,
            text: result.result_text.clone(),
        });
        debug!(" --> Inference result: {:?}", result);
        Ok(result)
    }

    pub fn clear_messsages(&self) -> Result<(), PhiError> {
        let mut conversation_context =
            self.conversation_context
                .lock()
                .map_err(|e| PhiError::LockingError {
                    error_text: e.to_string(),
                })?;
        conversation_context.messages.clear();
        Ok(())
    }

    pub fn get_history(&self) -> Result<Vec<ConversationMessage>, PhiError> {
        let conversation_context =
            self.conversation_context
                .lock()
                .map_err(|e| PhiError::LockingError {
                    error_text: e.to_string(),
                })?;
        Ok(conversation_context.messages.clone())
    }
}

#[derive(Clone)]
pub enum Model {
    Standard(Phi3),
    Quantized(QuantizedPhi3),
}

pub struct PhiEngine {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub event_handler: Option<Arc<dyn PhiEventHandler>>,
    pub context_window: u16,
}

impl PhiEngine {
    pub fn new(
        engine_options: EngineOptions,
        event_handler: Option<Arc<dyn PhiEventHandler>>,
    ) -> Result<Self, PhiError> {
        let start = std::time::Instant::now();
        // this also requires building with features = ["metal"]
        let device = if engine_options.use_gpu {
            Device::new_metal(0).map_err(|_| PhiError::GpuNotSupported)?
        } else {
            Device::Cpu
        };

        let (files, is_gguf, config) = match engine_options.model_provider {
            PhiModelProvider::HuggingFace {
                model_repo,
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
                let api_provider = ApiFileProvider { repo: api };
                let files = load_safetensors(&api_provider, "model.safetensors.index.json")?;

                debug!("Loaded model files: {:?}", files);

                let config = load_config(&api_provider, "config.json")?;
                debug!("Loaded model config: {:?}", config);
                (files, false, Some(config))
            }
            PhiModelProvider::HuggingFaceGguf {
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
                (vec![model_path], true, None)
            }
            PhiModelProvider::FileSystemGguf { model_path } => (vec![model_path.into()], true, None),
            PhiModelProvider::FileSystem { index_path, config_path } => {
                let index_path = std::path::PathBuf::from(&index_path);
                if !index_path.is_absolute() {
                    return Err(PhiError::InitalizationError {
                        error_text: format!("The index path must be absolute. Provided path: {:?}", index_path),
                    });
                }

                let base_dir = match index_path.parent() {
                    Some(parent) => parent.to_path_buf(),
                    None => {
                        return Err(PhiError::InitalizationError {
                            error_text: format!("The index path {:?} does not have a valid parent directory", index_path),
                        });
                    }
                };

                let index_path_str = match index_path.to_str() {
                    Some(path_str) => path_str,
                    None => {
                        return Err(PhiError::InitalizationError {
                            error_text: format!("The index path {:?} could not be converted to a valid string", index_path),
                        });
                    }
                };

                let fs_provider = FilesystemFileProvider::new(base_dir);
                let files = load_safetensors(&fs_provider, &index_path_str)?;
                debug!("Loaded model files: {:?}", files);

                let config = load_config(&fs_provider, &config_path)?;
                debug!("Loaded model config: {:?}", config);
                (files, false, Some(config))
            },
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

        let model = if is_gguf {
            // Load quantized model using gguf
            let mut file = File::open(&files[0]).map_err(|e| PhiError::InitalizationError {
                error_text: e.to_string(),
            })?;
            let model_content =
                gguf_file::Content::read(&mut file).map_err(|e| PhiError::InitalizationError {
                    error_text: e.to_string(),
                })?;
            let quantized_model =
                candle_transformers::models::quantized_phi3::ModelWeights::from_gguf(
                    engine_options.use_flash_attention,
                    model_content,
                    &mut file,
                    &device,
                )
                .map_err(|e| PhiError::InitalizationError {
                    error_text: e.to_string(),
                })?;
            Model::Quantized(quantized_model)
        } else {
            if let Some(config) = config {
                let dtype = match engine_options.dtype.as_deref() {
                    Some("f32") => DType::F32,
                    Some("bf16") => device.bf16_default_to_f32(),
                    _ => DType::F32,
                };
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&files, dtype, &device).map_err(|e| {
                        PhiError::InitalizationError {
                            error_text: format!("Error loading model: {:?}", e),
                        }
                    })?
                };
                let standard_model = candle_transformers::models::phi3::Model::new(&config, vb)
                    .map_err(|e| PhiError::InitalizationError {
                        error_text: e.to_string(),
                    })?;
                Model::Standard(standard_model)
            } else {
                return Err(PhiError::InitalizationError {
                    error_text: "Model config not found".to_string(),
                });
            }
        };

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| PhiError::InitalizationError {
                error_text: e.to_string(),
            })?;

        let event_handler_clone = event_handler.clone();

        debug!(" --> Loaded the model in {:?}", start.elapsed());
        if let Some(event_handler) = event_handler {
            event_handler
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
        prompt_text: &str,
        conversation_context: &ConversationContext,
        inference_options: &InferenceOptions,
    ) -> Result<InferenceResult, PhiError> {
        let mut history = conversation_context.messages.clone();
        self.trim_history_to_token_limit(&mut history, self.context_window);
        history.push(ConversationMessage {
            role: Role::User,
            text: prompt_text.into(),
        });
    
        let history_prompt = match inference_options.chat_format {
            ChatFormat::Llama2 => history
                .iter()
                .map(|entry| {
                    let role = match entry.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    };
                    format!("\n<|{}|>{}<|end|>", role, entry.text)
                })
                .collect::<String>(),
            ChatFormat::ChatML => history
                .iter()
                .map(|entry| {
                    let role = match entry.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    };
                    format!("\n<|im_start|>{}<|im_sep|>{}<|im_end|>", role, entry.text)
                })
                .collect::<String>(),
        };
    
        let prompt_with_history = match inference_options.chat_format {
            ChatFormat::Llama2 => {
                if let Some(system_instruction) = conversation_context.system_instruction.clone() {
                    format!("<|system|>{}<|end|>{}\n<|assistant|>\n", 
                        system_instruction, history_prompt)
                } else {
                    format!("{}\n<|assistant|>\n", history_prompt)
                }
            }
            ChatFormat::ChatML => {
                if let Some(system_instruction) = conversation_context.system_instruction.clone() {
                    format!(
                        "<|im_start|>system<|im_sep|>{}<|im_end|>{}\n<|im_start|>assistant<|im_sep|>\n",
                        system_instruction, history_prompt
                    )
                } else {
                    format!("{}\n<|im_start|>assistant<|im_sep|>\n", history_prompt)
                }
            }
        };
    
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

// see https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs#L125C5-L149C2
fn load_safetensors(
    provider: &dyn FileProvider,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>, PhiError> {
    let json_file = provider.get(json_file)?;
    let json_file = std::fs::File::open(json_file).map_err(|e| PhiError::InitalizationError {
        error_text: e.to_string(),
    })?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(|e| PhiError::InitalizationError {
            error_text: e.to_string(),
        })?;
    let weight_map = match json.get("weight_map") {
        None => Err(PhiError::InitalizationError {
            error_text: "weight map not found in json file".to_string(),
        }),
        Some(serde_json::Value::Object(map)) => Ok(map),
        Some(_) => Err(PhiError::InitalizationError {
            error_text: "weight map is not an object".to_string(),
        }),
    }?;
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| provider.get(v))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(safetensors_files)
}

fn load_config(provider: &dyn FileProvider, config_file: &str) -> Result<Phi3Config, PhiError> {
    let config_filename = provider.get(config_file)?;
    let config_content = std::fs::read_to_string(config_filename).map_err(|e| {
        PhiError::InitalizationError {
            error_text: e.to_string(),
        }
    })?;
    let config: Phi3Config = serde_json::from_str(&config_content).map_err(|e| {
        PhiError::InitalizationError {
            error_text: e.to_string(),
        }
    })?;
    Ok(config)
}

trait FileProvider {
    fn get(&self, file_path: &str) -> Result<std::path::PathBuf, PhiError>;
}

struct ApiFileProvider {
    repo: hf_hub::api::sync::ApiRepo,
}

impl FileProvider for ApiFileProvider {
    fn get(&self, file_path: &str) -> Result<std::path::PathBuf, PhiError> {
        self.repo.get(file_path).map_err(|e| PhiError::InitalizationError {
            error_text: e.to_string(),
        })
    }
}

struct FilesystemFileProvider {
    base_dir: std::path::PathBuf,
}

impl FilesystemFileProvider {
    fn new(base_dir: std::path::PathBuf) -> Self {
        FilesystemFileProvider { base_dir }
    }
}
impl FileProvider for FilesystemFileProvider {
    fn get(&self, file_path: &str) -> Result<std::path::PathBuf, PhiError> {
        let path = std::path::PathBuf::from(file_path);
        let full_path = if path.is_absolute() {
            path
        } else {
            self.base_dir.join(path)
        };
        
        if full_path.exists() {
            Ok(full_path)
        } else {
            Err(PhiError::InitalizationError {
                error_text: format!("File not found: {}", full_path.display()),
            })
        }
    }
}