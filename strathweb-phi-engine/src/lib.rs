uniffi::include_scaffolding!("strathweb-phi-engine");

use std::collections::VecDeque;
use std::io::{stdout, Write};
use std::sync::Mutex;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use hf_hub::RepoType;
use hf_hub::{api::sync::Api, Repo};
use tokenizers::Tokenizer;
use anyhow::{Error as E, Result};

struct PhiEngine {
    pub model: Phi,
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
    pub model_name: Option<String>,
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
        
        // defaults
        let model_name = engine_options.model_name.unwrap_or("rhysjones/phi-2-orange-v2".to_string());
        let model_revision = engine_options.model_revision.unwrap_or("main".to_string());
        let system_instruction = engine_options.system_instruction.unwrap_or("You are a helpful assistant that answers user questions. Be short and direct in your answers.".to_string());

        let repo = Api::new().unwrap().repo(Repo::with_revision(
            model_name, RepoType::Model, model_revision,
        ));
        let tokenizer_filename = repo.get("tokenizer.json").unwrap();
    
        let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json").unwrap();
    
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg).unwrap();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &Device::Cpu).unwrap() };
        let config_filename = repo.get("config.json").unwrap();
        let config = std::fs::read_to_string(config_filename).unwrap();
        let config: PhiConfig = serde_json::from_str(&config).unwrap();
        let model = Phi::new(&config, vb).unwrap();

        println!("Loaded the model in {:?}", start.elapsed());


        Self {
            model: model,
            tokenizer: tokenizer,
            device: Device::Cpu, // candle does not support Metal on iOS
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

        let history_prompt = history
            .iter()
            .enumerate()
            .map(|(i, text)| if i % 2 == 0 { format!("<|im_start|>user\n{}<|im_end|>", text) } else { format!("<|im_start|>assistant\n{}<|im_end|>", text) })
            .collect::<String>();

        let prompt_with_history = format!("<|im_start|>system\n{}<|im_end|>\n{}<|im_start|>assistant\n", self.system_instruction, history_prompt);

        let mut pipeline = TextGeneration::new(
            &self.model,
            self.tokenizer.clone(),
            inference_options,
            false,
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
    model: Phi,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    inference_options: InferenceOptions,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &Phi,
        tokenizer: Tokenizer,
        inference_options: &InferenceOptions,
        verbose_prompt: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(inference_options.seed, inference_options.temperature, inference_options.top_p);
        Self {
            model: model.clone(),
            tokenizer,
            logits_processor,
            inference_options: inference_options.clone(),
            verbose_prompt,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: u16) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(true).get("<|im_end|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the im_end token"),
        };
        stdout().flush()?;

        let mut response = String::new();
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.inference_options.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.inference_options.repeat_last_n.into());
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.inference_options.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            response.push_str(&token);
            print!("{token}");
            std::io::stdout().flush()?;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        Ok(response)
    }
}

fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file)?;
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(E::from))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(safetensors_files)
}