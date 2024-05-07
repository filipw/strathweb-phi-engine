uniffi::include_scaffolding!("strathweb-phi-engine");

use std::collections::VecDeque;
use std::io::{stdin, stdout, Write};
use candle_core::op::Op;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::quantized_llama::ModelWeights as QPhi3;
use candle_transformers::models::quantized_phi::ModelWeights;
use hf_hub::RepoType;
use hf_hub::{api::sync::Api, Repo};
use serde_json::de;
use tokenizers::Tokenizer;
use anyhow::{Error as E, Result};
use std::sync::Mutex;


struct PhiEngine {
    pub model: QPhi3,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub history: Mutex<VecDeque<String>>,
    pub system_instruction: String
}

#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub token_count: u16,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: u16,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub model_repo: Option<String>,
    pub tokenizer_repo: Option<String>,
    pub model_file_name: Option<String>,
    pub model_revision: Option<String>,
    pub system_instruction: Option<String>
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            token_count: 250,
            temperature: Some(0.0),
            top_p: Some(1.0),
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 299792458,
        }
    }
}

impl PhiEngine {
    pub fn new(engine_options: EngineOptions) -> Self {
        let start = std::time::Instant::now();
        //let device = Device::new_metal(0).unwrap();
        // candle does not support Metal on iOS
        let device = Device::Cpu;

        // defaults
        let tokenizer_repo = engine_options.tokenizer_repo.unwrap_or("microsoft/Phi-3-mini-4k-instruct".to_string());
        let model_repo = engine_options.model_repo.unwrap_or("microsoft/Phi-3-mini-4k-instruct-gguf".to_string());
        let model_file_name = engine_options.model_file_name.unwrap_or("Phi-3-mini-4k-instruct-q4.gguf".to_string());
        let model_revision = engine_options.model_revision.unwrap_or("main".to_string());
        let system_instruction = engine_options.system_instruction.unwrap_or("You are a helpful assistant that answers user questions. Be short and direct in your answers.".to_string());

        let api = hf_hub::api::sync::Api::new().unwrap();
        let api = api.model(model_repo.to_string());
        let model_path = api.get(model_file_name.as_str()).unwrap();

        let api = hf_hub::api::sync::Api::new().unwrap();
        let api = api.model(tokenizer_repo.to_string());
        let tokenizer_path = api.get("tokenizer.json").unwrap();
    
        let mut file = std::fs::File::open(&model_path).unwrap();
            let model = gguf_file::Content::read(&mut file).unwrap();
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensor_infos.iter() {
                let elem_count = tensor.shape.elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
            }
        let mut model = QPhi3::from_gguf(model, &mut file, &device).unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        println!("Loaded the model in {:?}", start.elapsed());

        Self {
            model: model,
            tokenizer: tokenizer,
            device: device, 
            history: Mutex::new(VecDeque::with_capacity(6)),
            system_instruction: system_instruction
        }
    }

    pub fn run_inference(&self, prompt_text: String, inference_options: &InferenceOptions) -> String {
        let mut history = self.history.lock().unwrap();
        if history.len() == 6 {
            history.pop_front();
            history.pop_front();
        }

        history.push_back(prompt_text.clone());

        // todo: Phi-3 has not system prompt
        let history_prompt = history
            .iter()
            .enumerate()
            .map(|(i, text)| if i % 2 == 0 { format!("\n<|user|>\n{}<|end|>", text) } else { format!("\n<|assistant|>\n{}<|end|>", text) })
            .collect::<String>();

        let prompt_with_history = format!("<|system|>\n{}<|end|>\n{}\n<|assistant|>\n", self.system_instruction, history_prompt);

        let mut pipeline = TextGeneration::new(
            &self.model,
            self.tokenizer.clone(),
            inference_options,
            &self.device,
        );

        let response = pipeline.run(&prompt_with_history, inference_options.token_count).unwrap();
        history.push_back(response.clone());
        response
    }

    pub fn clear_history(&self) {
        let mut history = self.history.lock().unwrap();
        history.clear();
    }
}

struct TextGeneration {
    model: QPhi3,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    inference_options: InferenceOptions,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &QPhi3,
        tokenizer: Tokenizer,
        inference_options: &InferenceOptions,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(inference_options.seed, inference_options.temperature, inference_options.top_p);
        Self {
            model: model.clone(),
            tokenizer,
            logits_processor,
            inference_options: inference_options.clone(),
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: u16) -> Result<String> {
        println!("{}", prompt);

        let mut tos = TokenOutputStream::new(self.tokenizer.clone());
        let tokens = tos
            .tokenizer()
            .encode(prompt, true).unwrap();
        let tokens = tokens.get_ids();

        let mut all_tokens = vec![];
        let mut logits_processor = LogitsProcessor::new(
            self.inference_options.seed,
            self.inference_options.temperature,
            self.inference_options.top_p,
        );

        let start_prompt_processing = std::time::Instant::now();
        let mut next_token = {
            let mut next_token = 0;
            for (pos, token) in tokens.iter().enumerate() {
                let input = Tensor::new(&[*token], &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, pos).unwrap();
                let logits = logits.squeeze(0)?;
                next_token = logits_processor.sample(&logits)?;
            }
            next_token
        };

        let prompt_dt = start_prompt_processing.elapsed();
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        let eos_token = *self
        .tokenizer
        .get_vocab(true)
        .get("<|end|>")
        .unwrap();

        let start_post_prompt = std::time::Instant::now();
        let mut sampled = 0;
        let to_sample = sample_len.saturating_sub(1) as usize;
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() + index).unwrap();
            let logits = logits.squeeze(0)?;
            let logits = if self.inference_options.repeat_penalty == 1.0 {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(self.inference_options.repeat_last_n.into());
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.inference_options.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };

            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if next_token == eos_token {
                break;
            }

            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            sampled += 1;
        }

        let dt = start_post_prompt.elapsed();
        println!(
            "\n\n{:4} prompt tokens processed in {:.2} seconds ({:.2} token/s)",
            tokens.len(),
            prompt_dt.as_secs_f64(),
            tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled} tokens generated in {:.2} seconds ({:.2} token/s)",
            dt.as_secs_f64(),
            sampled as f64 / dt.as_secs_f64(),
        );

        Ok(tos.decode_all().map_err(E::msg)?)
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

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_ascii() {
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