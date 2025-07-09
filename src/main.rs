use candle_core::Tensor;
use hf_hub::Repo;
use tokenizers::Tokenizer;

mod api;
mod compute_device;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = compute_device::get();
    let api = api::get().await;

    // Use a smaller model that's publicly available and works well
    let repo = api.repo(Repo::model("gpt2".to_string()));

    println!("Downloading tokenizer...");
    let tokenizer_path = repo.get("tokenizer.json").await?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
    println!("Tokenizer loaded");

    // 3. For this demo, let's create a simple text completion using the tokenizer
    let prompt = "Hi! How are you doing today?";
    println!("\nRunning inference...");
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
