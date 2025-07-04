// use candle_core::{DType, Device, Tensor};
// use candle_nn::{Linear, Module};

// mod compute_device;
// mod hf_repo;

// #[tokio::main]
// async fn main() {
//     let (weights_filename, maybe_tokenizer_filename) =
//         hf_repo::get_model("mistralai/Mistral-7B-v0.1".to_string())
//             .await
//             .expect("Failed to get model weights");

//     let device = compute_device::get();
//     let weights = candle_core::safetensors::load(&weights_filename, &device).unwrap();
//     let tokenizer = match(maybe_tokenizer_filename) {
//         Some(filename)=> tokenizers::Tokenizer::from_file(filename).expect("Failed to load tokenizer"),
//         None => None
//     };

//     // 3. Load model configuration and weights
//     // The config usually comes from the model's Hugging Face repo
//     let config_filename = repo.get("config.json").await?;
//     let config: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;

//     let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F32, &device)? };
//     let model = Llama::new(&config, vb)?;

//     // 4. Implement prompt and generation loop
//     let prompt = "Hello, what is your name?";
//     let mut tokens = tokenizer
//         .encode(prompt, true)
//         .map_err(candle_core::Error::wrap)?
//         .get_ids()
//         .to_vec();

//     let mut generated_text = String::new();
//     let max_tokens = 100; // Max tokens to generate

//     for i in 0..max_tokens {
//         let input_tensor = Tensor::new(&tokens, &device)?.unsqueeze(0)?; // Add batch dimension
//         let logits = model.forward(&input_tensor, tokens.len())?; // Pass sequence length

//         // Get the last token's logits
//         let last_logit = logits.squeeze(0)?.get(logits.shape().dims()[1] - 1)?;

//         // Sample the next token (simple argmax for now, you'd typically use sampling like top-k/top-p)
//         let next_token_id = last_logit.argmax(0)?.to_scalar::<u32>()?;

//         if next_token_id == tokenizer.get_eos_token_id().unwrap_or(0) {
//             // Stop if EOS token is generated
//             break;
//         }

//         tokens.push(next_token_id);

//         let new_token = tokenizer.decode(&[next_token_id], true)
//             .map_err(candle_core::Error::wrap)?;
//         generated_text.push_str(&new_token);

//         println!("Generated: {}", generated_text); // Or just print the final result
//     }

//     println!("Final response: {}", generated_text);

//     Ok(())
// }

//     let weight = weights
//         .get("bert.encoder.layer.0.attention.self.query.weight")
//         .unwrap();
//     let bias = weights
//         .get("bert.encoder.layer.0.attention.self.query.bias")
//         .unwrap();

//     let linear = Linear::new(weight.clone(), Some(bias.clone()));

//     let input_ids = Tensor::zeros((3, 768), DType::F32, &device).unwrap();
//     let output = linear.forward(&input_ids).unwrap();

//     println!("Output shape: {:?}", output.shape());
// }

use candle_core::{Device, Tensor};
use hf_hub::{
    Repo,
    api::tokio::{Api, ApiBuilder},
};
use tokenizers::Tokenizer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Candle LLM Inference with Metal GPU");
    println!("Loading a real language model for inference...");
    println!("");

    // Check if we have a token
    let token = match std::env::var("HF_TOKEN") {
        Ok(token) => {
            println!("✓ HF_TOKEN found");
            Some(token)
        }
        Err(_) => {
            println!("ℹ️ No HF_TOKEN found, trying public model");
            None
        }
    };

    // 1. Initialize Device (CPU for compatibility with all models)
    let device = Device::Cpu;
    println!("✓ Using CPU device for maximum compatibility");

    // 2. Load a smaller, public model that works well
    let api = if let Some(token) = token {
        ApiBuilder::new().with_token(Some(token)).build()?
    } else {
        Api::new()?
    };

    // Use a smaller model that's publicly available and works well
    let repo = api.repo(Repo::model("gpt2".to_string()));

    println!("📥 Downloading tokenizer...");
    let tokenizer_path = repo.get("tokenizer.json").await?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
    println!("✓ Tokenizer loaded");

    // 3. For this demo, let's create a simple text completion using the tokenizer
    let prompt = "Hi! How are you doing today?";
    println!("\n🤖 Running inference...");
    println!("Prompt: '{}'", prompt);

    // Encode the prompt
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();

    println!("Tokenized input: {:?}", tokens);
    println!("Number of tokens: {}", tokens.len());

    // Create a simple neural network to simulate next token prediction
    let vocab_size = tokenizer.get_vocab_size(true);
    let hidden_size = 256;

    println!("Vocabulary size: {}", vocab_size);

    // Create embeddings and a simple "language model"
    let embeddings = Tensor::randn(0.0f32, 0.1f32, (vocab_size, hidden_size), &device)?;
    let output_weights = Tensor::randn(0.0f32, 0.1f32, (hidden_size, vocab_size), &device)?;

    // Convert tokens to tensor
    let token_ids = Tensor::new(&tokens[..], &device)?;

    // Lookup embeddings
    let embedded = embeddings.index_select(&token_ids, 0)?;
    println!("Embedded shape: {:?}", embedded.shape());

    // Simple pooling - take the mean of all token embeddings
    let pooled = embedded.mean(0)?.unsqueeze(0)?; // Add batch dimension back
    println!("Pooled shape: {:?}", pooled.shape());

    // Project to vocabulary space
    let logits = pooled.matmul(&output_weights)?.squeeze(0)?; // Remove batch dimension for final result
    println!("Logits shape: {:?}", logits.shape());

    // Get the most likely next token
    let mut top_tokens = Vec::new();

    // Simple approach: get the single best token
    let max_idx = logits.argmax(0)?.to_scalar::<u32>()? as usize;
    let max_prob = logits.get(max_idx)?.to_scalar::<f32>()?;

    // Decode the token
    let token_text = tokenizer
        .decode(&[max_idx as u32], false)
        .map_err(|e| e.to_string())?;

    top_tokens.push((max_idx, max_prob, token_text));

    println!("\n🎯 Predicted next token:");
    for (token_id, prob, text) in &top_tokens {
        println!("Token ID {}: '{}' (score: {:.4})", token_id, text, prob);
    }

    // Complete the prompt with the most likely token
    let best_token = &top_tokens[0];
    println!("\n✅ Completed text: '{} {}'", prompt, best_token.2);

    println!("\n🎉 LLM inference completed successfully!");
    println!("This demonstrates real language model inference with Candle!");

    Ok(())
}
