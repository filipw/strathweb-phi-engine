use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;
use std::io::Write;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::engine::{BoxedPhiEventHandler, InferenceOptions, InferenceResult};
use crate::token_stream::TokenOutputStream;
use crate::PhiError;

pub(crate) struct TextGenerator {
    model: Phi3,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    inference_options: InferenceOptions,
    event_handler: Option<Arc<BoxedPhiEventHandler>>,
}

impl TextGenerator {
    pub fn new(
        model: &Phi3,
        tokenizer: Tokenizer,
        inference_options: &InferenceOptions,
        device: &Device,
        event_handler: Option<Arc<BoxedPhiEventHandler>>,
    ) -> Self {
        let logits_processor = {
            let temperature = inference_options.temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (inference_options.top_k, inference_options.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK {
                        k: usize::try_from(k).unwrap_or(0),
                        temperature,
                    },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP {
                        k: usize::try_from(k).unwrap_or(0),
                        p,
                        temperature,
                    },
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
    pub fn run(&mut self, prompt: &str, sample_len: u16) -> Result<InferenceResult> {
        //println!("{}", prompt);

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
            if let Some(event_handler) = &self.event_handler {
                event_handler
                    .handler
                    .on_inference_token(t)
                    .map_err(|e| PhiError::InferenceError {
                        error_text: e.to_string(),
                    })?;
            }
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
                println!("\n\nBreaking due to end token: ${:?}$", next_token);
                std::io::stdout().flush()?;
                break;
            }

            if let Some(t) = tos.next_token(next_token)? {
                if let Some(event_handler) = &self.event_handler {
                    event_handler
                        .handler
                        .on_inference_token(t)
                        .map_err(|e| PhiError::InferenceError {
                            error_text: e.to_string(),
                        })?;
                }
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
