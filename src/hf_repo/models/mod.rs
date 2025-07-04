use std::collections::HashMap;

pub struct ModelDetails {
    pub weights_filename: String,
    pub tokenizer_filename: Option<String>,
}

pub fn get() -> HashMap<String, ModelDetails> {
    let default_weights_filename = "model.safetensors".to_string();
    let default_tokenizer_filename = Some("tokenizer.json".to_string());

    HashMap::from([
        (
            "mistralai/Mistral-7B-v0.1".to_string(),
            ModelDetails {
                weights_filename: default_weights_filename.clone(),
                tokenizer_filename: default_tokenizer_filename.clone(),
            },
        ),
        (
            "meta-llama/Llama-2-7b-chat-hf".to_string(),
            ModelDetails {
                weights_filename: default_weights_filename.clone(),
                tokenizer_filename: None,
            },
        ),
        (
            "meta-llama/Llama-2-13b-chat-hf".to_string(),
            ModelDetails {
                weights_filename: default_weights_filename.clone(),
                tokenizer_filename: None,
            },
        ),
        (
            "meta-llama/Llama-2-70b-chat-hf".to_string(),
            ModelDetails {
                weights_filename: default_weights_filename.clone(),
                tokenizer_filename: None,
            },
        ),
    ])
}
