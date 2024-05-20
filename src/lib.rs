use candle_transformers::models::qwen2::{Config, Model};

use candle_core::{utils, DType, Device, Tensor};
use candle_nn::VarBuilder;
use thiserror::Error;
use tokenizers::{
    utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy},
    Tokenizer,
};

#[derive(Error, Debug)]
pub enum EmbdQwenError {
    #[error("Configuration error, {0}")]
    InvalidConfig(String),
    #[error("Tokenizer error, {0}")]
    TokenizerError(String),
    #[error("Infer error, {0}")]
    InferError(String),
    #[error("Prompt error, {0}")]
    PromptError(String),
}

// gte-Qwen1.5-7B-instruct use EOS token as padding token
const EOS_TOKEN: &str = "<|endoftext|>";
const EOS_TOKEN_ID: u32 = 151643;
const INSTRUCT: &str =
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ";

#[derive(Debug)]
struct ConfigFiles {
    pub config: std::path::PathBuf,
    pub tokenizer: std::path::PathBuf,
    pub weights: Vec<std::path::PathBuf>,
}

// Loading the model from a local directory.
fn load_from_local(local_path: &str) -> Result<ConfigFiles, EmbdQwenError> {
    let local_path = std::path::PathBuf::from(local_path);
    let weight_path = local_path.join("model.safetensors.index.json");
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(weight_path)
            .map_err(|e| EmbdQwenError::InvalidConfig(e.to_string()))?,
    )
    .map_err(|e| EmbdQwenError::InvalidConfig(e.to_string()))?;
    let weight_map = match json.get("weight_map") {
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("`weight map` is not a map"),
        None => panic!("`weight map` not found"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        safetensors_files.insert(
            value
                .as_str()
                .expect("Weight files should be parsed as strings"),
        );
    }
    let safetensors_paths = safetensors_files
        .iter()
        .map(|v| local_path.join(v))
        .collect::<Vec<_>>();
    Ok(ConfigFiles {
        config: local_path.join("config.json"),
        tokenizer: local_path.join("tokenizer.json"),
        weights: safetensors_paths,
    })
}

/// Returns the best device to use for inference.
pub fn auto_device() -> Device {
    if utils::cuda_is_available() {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    }
}

pub struct EmbdQwenModel {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbdQwenModel {
    pub fn new(local_path: &str) -> Result<Self, EmbdQwenError> {
        let config_files = load_from_local(local_path)?;

        // Inputs will be padded to the longest sequence in the batch.
        let padding = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Left,
            pad_to_multiple_of: None,
            pad_id: EOS_TOKEN_ID,
            pad_type_id: 0,
            pad_token: String::from(EOS_TOKEN),
        };

        // Tokenizer setup
        let mut tokenizer = Tokenizer::from_file(config_files.tokenizer)
            .map_err(|e| EmbdQwenError::TokenizerError(e.to_string()))?;
        tokenizer.with_padding(Some(padding));

        // Model initialization
        let device = auto_device();
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(config_files.config)
                .map_err(|e| EmbdQwenError::InvalidConfig(e.to_string()))?,
        )
        .map_err(|e| EmbdQwenError::InvalidConfig(e.to_string()))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&config_files.weights, dtype, &device)
                .map_err(|e| EmbdQwenError::InvalidConfig(e.to_string()))?
        };
        let model =
            Model::new(&config, vb).map_err(|e| EmbdQwenError::InvalidConfig(e.to_string()))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn prepare(&self, texts: &str, as_query: bool) -> String {
        match as_query {
            true => format!("{INSTRUCT}{texts}{EOS_TOKEN}"),
            false => format!("{texts}{EOS_TOKEN}"),
        }
    }

    pub fn score(&self, t1: &Tensor, t2: &Tensor) -> Result<Vec<Vec<f32>>, EmbdQwenError> {
        t1.matmul(
            &t2.t()
                .map_err(|e| EmbdQwenError::InferError(e.to_string()))?,
        )
        .map_err(|e| EmbdQwenError::InferError(e.to_string()))?
        .to_vec2::<f32>()
        .map_err(|e| EmbdQwenError::InferError(e.to_string()))
    }

    pub fn embedding(&mut self, documents: Vec<&str>) -> Result<Tensor, EmbdQwenError> {
        let encoded = self
            .tokenizer
            .encode_batch(documents, true)
            .map_err(|e| EmbdQwenError::TokenizerError(e.to_string()))?;
        let tokens: Vec<&[u32]> = encoded.iter().map(|x| x.get_ids()).collect();
        let tokens = Tensor::new(tokens, &self.device)
            .map_err(|e| EmbdQwenError::TokenizerError(e.to_string()))?;
        let mask: Vec<&[u32]> = encoded.iter().map(|x| x.get_attention_mask()).collect();
        let mask = Tensor::new(mask, &self.device)
            .map_err(|e| EmbdQwenError::TokenizerError(e.to_string()))?;

        // Inference
        self.model.clear_kv_cache();
        let logits = self
            .model
            .forward(&tokens, 0, Some(&mask))
            .map_err(|e| EmbdQwenError::InferError(e.to_string()))?;

        // Extract the last hidden states as embeddings since inputs are padded left.
        let (_, seq_len, _) = logits.dims3().unwrap();
        let embd = logits
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| EmbdQwenError::InferError(e.to_string()))?
            .squeeze(1)
            .map_err(|e| EmbdQwenError::InferError(e.to_string()))?
            .to_dtype(DType::F32)
            .map_err(|e| EmbdQwenError::InferError(e.to_string()))?;

        // Embeddings should be normalized.
        embd.broadcast_div(
            &embd
                .sqr()
                .map_err(|e| EmbdQwenError::InferError(e.to_string()))?
                .sum_keepdim(1)
                .map_err(|e| EmbdQwenError::InferError(e.to_string()))?
                .sqrt()
                .map_err(|e| EmbdQwenError::InferError(e.to_string()))?,
        )
        .map_err(|e| EmbdQwenError::InferError(e.to_string()))
    }
}
