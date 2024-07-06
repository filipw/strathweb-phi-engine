uniffi::include_scaffolding!("strathweb-phi-engine");

use anyhow::{Error as E, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Repo;
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokenizers::Tokenizer;

struct PhiEngine {
    pub model: Phi3,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub history: Mutex<VecDeque<String>>,
    pub system_instruction: String,
    pub event_handler: Arc<BoxedPhiEventHandler>,
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
    pub model_repo: Option<String>,
    pub tokenizer_repo: Option<String>,
    pub model_file_name: Option<String>,
    pub model_revision: Option<String>,
    pub use_flash_attention: bool,
    pub system_instruction: Option<String>,
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

impl PhiEngine {
    pub fn new(
        engine_options: EngineOptions,
        event_handler: Arc<BoxedPhiEventHandler>,
    ) -> Result<Self, PhiError> {
        let start = std::time::Instant::now();
        // candle does not support Metal on iOS yet
        // this also requires building with features = ["metal"]
        //let device = Device::new_metal(0).unwrap();
        let device = Device::Cpu;

        // defaults
        let tokenizer_repo = engine_options
            .tokenizer_repo
            .unwrap_or("microsoft/Phi-3-mini-4k-instruct".to_string());
        let model_repo = engine_options
            .model_repo
            .unwrap_or("microsoft/Phi-3-mini-4k-instruct-gguf".to_string());
        let model_file_name = engine_options
            .model_file_name
            .unwrap_or("Phi-3-mini-4k-instruct-q4.gguf".to_string());
        let model_revision = engine_options
            .model_revision
            .unwrap_or("main".to_string());
        let system_instruction = engine_options.system_instruction.unwrap_or("You are a helpful assistant that answers user questions. Be short and direct in your answers.".to_string());

        let api_builder =
            ApiBuilder::new().with_cache_dir(PathBuf::from(engine_options.cache_dir.clone()));
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
        let model_path =
            api.get(model_file_name.as_str())
                .map_err(|e| PhiError::InitalizationError {
                    error_text: e.to_string(),
                })?;
        print!("Downloaded model to {:?}...", model_path);

        let api_builder =
            ApiBuilder::new().with_cache_dir(PathBuf::from(engine_options.cache_dir.clone()));
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
        let tokenizer_path =
            api.get("tokenizer.json")
                .map_err(|e| PhiError::InitalizationError {
                    error_text: e.to_string(),
                })?;
        print!("Downloaded tokenizer to {:?}...", tokenizer_path);

        let mut file = File::open(&model_path).map_err(|e| PhiError::InitalizationError {
            error_text: e.to_string(),
        })?;
        let model_content =
            gguf_file::Content::read(&mut file).map_err(|e| PhiError::InitalizationError {
                error_text: e.to_string(),
            })?;
        let model = Phi3::from_gguf(engine_options.use_flash_attention, model_content, &mut file, &device).map_err(|e| {
            PhiError::InitalizationError {
                error_text: e.to_string(),
            }
        })?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| PhiError::InitalizationError {
                error_text: e.to_string(),
            })?;

        println!("Loaded the model in {:?}", start.elapsed());
        event_handler.handler
            .on_model_loaded()
            .map_err(|e| PhiError::InitalizationError {
                error_text: e.to_string(),
            })?;

        Ok(Self {
            model: model,
            tokenizer: tokenizer,
            device: device,
            history: Mutex::new(VecDeque::with_capacity(6)),
            system_instruction: system_instruction,
            event_handler: event_handler,
        })
    }

    pub fn run_inference(
        &self,
        prompt_text: String,
        inference_options: &InferenceOptions,
    ) -> Result<InferenceResult, PhiError> {
        let mut history = self.history.lock().map_err(|e| PhiError::HistoryError {
            error_text: e.to_string(),
        })?;

        // todo: this is a hack to keep the history length short so that we don't overflow the token limit
        // under normal circumstances we should count the tokens
        if history.len() == 10 {
            history.pop_front();
            history.pop_front();
        }

        history.push_back(prompt_text.clone());

        let history_prompt = history
            .iter()
            .enumerate()
            .map(|(i, text)| {
                if i % 2 == 0 {
                    format!("\n<|user|>\n{}<|end|>", text)
                } else {
                    format!("\n<|assistant|>\n{}<|end|>", text)
                }
            })
            .collect::<String>();

        // Phi-3 has no system prompt so we inject it as a user prompt
        let prompt_with_history = format!("<|user|>\nYour overall instructions are: {}<|end|>\n<|assistant|>Understood, I will adhere to these instructions<|end|>{}\n<|assistant|>\n", self.system_instruction, history_prompt);

        let mut pipeline = TextGeneration::new(
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
        history.push_back(response.result_text.clone());
        Ok(response)
    }

    pub fn clear_history(&self) -> Result<(), PhiError> {
        let mut history = self.history.lock().map_err(|e| PhiError::HistoryError {
            error_text: e.to_string(),
        })?;
        history.clear();
        Ok(())
    }
}

struct TextGeneration {
    model: Phi3,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    inference_options: InferenceOptions,
    event_handler: Arc<BoxedPhiEventHandler>,
}

impl TextGeneration {
    fn new(
        model: &Phi3,
        tokenizer: Tokenizer,
        inference_options: &InferenceOptions,
        device: &Device,
        event_handler: Arc<BoxedPhiEventHandler>,
    ) -> Self {
        let logits_processor = {
            let temperature = inference_options.temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (inference_options.top_k, inference_options.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k: usize::try_from(k).unwrap_or(0), temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k: usize::try_from(k).unwrap_or(0), p, temperature },
                }
            };
            LogitsProcessor::from_sampling(inference_options.seed, sampling)
        };
        Self {
            model: model.clone(),
            tokenizer,
            logits_processor,
            inference_options: inference_options.clone(),
            device: device.clone(),
            event_handler: event_handler,
        }
    }

    // inference code adapted from https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized/main.rs
    fn run(&mut self, prompt: &str, sample_len: u16) -> Result<InferenceResult> {
        println!("{}", prompt);

        let mut tos = TokenOutputStream::new(self.tokenizer.clone());
        let tokens = tos.tokenizer().encode(prompt, true).map_err(E::msg)?;
        let tokens = tokens.get_ids();

        let mut all_tokens = vec![];
        let mut next_token = {
            let mut next_token = 0;
            for (pos, token) in tokens.iter().enumerate() {
                let input = Tensor::new(&[*token], &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, pos)?;
                let logits = logits.squeeze(0)?;
                next_token = self.logits_processor.sample(&logits)?;
            }
            next_token
        };

        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            self.event_handler.handler
                .on_inference_token(t)
                .map_err(|e| PhiError::InferenceError {
                    error_text: e.to_string(),
                })?;
        }

        let binding = self.tokenizer.get_vocab(true);
        let endoftext_token = binding
            .get("<|endoftext|>")
            .ok_or_else(|| anyhow::Error::msg("No <|endoftext|> found"))?;
        let end_token = binding
            .get("<|end|>")
            .ok_or_else(|| anyhow::Error::msg("No <|end|> found"))?;
        let assistant_token = binding
            .get("<|assistant|>")
            .ok_or_else(|| anyhow::Error::msg("No <|assistant|> found"))?;

        let start_post_prompt = std::time::Instant::now();
        let mut sampled = 0;
        let to_sample = sample_len.saturating_sub(1) as usize;
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if self.inference_options.repeat_penalty == 1.0 {
                logits
            } else {
                let start_at = all_tokens
                    .len()
                    .saturating_sub(self.inference_options.repeat_last_n.into());
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.inference_options.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };

            next_token = self.logits_processor.sample(&logits)?;
            all_tokens.push(next_token);

            if &next_token == endoftext_token
                || &next_token == end_token
                || &next_token == assistant_token
            {
                println!("Breaking due to eos: ${:?}$", next_token);
                std::io::stdout().flush()?;
                break;
            }

            if let Some(t) = tos.next_token(next_token)? {
                self.event_handler.handler
                    .on_inference_token(t)
                    .map_err(|e| PhiError::InferenceError {
                        error_text: e.to_string(),
                    })?;
            }
            sampled += 1;
        }

        let dt = start_post_prompt.elapsed();
        let inference_result = InferenceResult {
            token_count: sampled,
            result_text: tos.decode_all().map_err(E::msg)?,
            duration: dt.as_secs_f64(),
            tokens_per_second: sampled as f64 / dt.as_secs_f64(),
        };
        Ok(inference_result)
    }
}

pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => anyhow::bail!("cannot decode: {err}"),
        }
    }

    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

#[derive(Error, Debug)]
pub enum PhiError {
    #[error("InitalizationError with message: `{error_text}`")]
    InitalizationError { error_text: String },

    #[error("InferenceError with message: `{error_text}`")]
    InferenceError { error_text: String },

    #[error("History with message: `{error_text}`")]
    HistoryError { error_text: String },
}
