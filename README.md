# Mistral 7B Inference with Candle

This project demonstrates how to run inference with the Mistral 7B model using the Candle framework on macOS with Metal acceleration.

## Prerequisites

1. **macOS with Metal support** (Apple Silicon or Intel Mac with Metal-compatible GPU)
2. **Rust toolchain** installed
3. **HuggingFace account and token**

## Setup

### 1. Get a HuggingFace Token

1. Create an account at [HuggingFace](https://huggingface.co/)
2. Go to [Mistral-7B-v0.1 model page](https://huggingface.co/mistralai/Mistral-7B-v0.1)
3. Accept the terms and conditions for the model
4. Generate a token from your [HuggingFace settings](https://huggingface.co/settings/tokens)

### 2. Set Environment Variable

```bash
export HF_TOKEN=your_huggingface_token_here
```

### 3. Build and Run

```bash
# Build the project
cargo build --release

# Run the inference
cargo run --release
```

## What This Does

The program will:

1. **Check for HF_TOKEN** - Ensures you have authentication set up
2. **Initialize Metal device** - Uses Apple's Metal for GPU acceleration
3. **Download tokenizer** - Downloads the Mistral tokenizer from HuggingFace
4. **Download config** - Downloads the model configuration
5. **Download model weights** - Downloads the model weights (this is a large file, ~13GB)
6. **Load model** - Loads the model into GPU memory
7. **Run inference** - Runs a single forward pass with the prompt "The capital of France is "
8. **Display results** - Shows the next predicted token

## Expected Output

```
This example requires a HuggingFace token to access the Mistral model.
To set up authentication:
1. Create an account at https://huggingface.co/
2. Accept the terms for the Mistral-7B-v0.1 model
3. Set the HF_TOKEN environment variable with your token
Example: export HF_TOKEN=your_token_here

✓ HF_TOKEN found
✓ Metal device initialized
📥 Downloading tokenizer...
✓ Tokenizer loaded
📥 Downloading config...
✓ Config loaded
📥 Downloading model weights (this may take a while)...
🧠 Loading model into memory...
✓ Model loaded successfully!

🤖 Running inference...
Prompt: 'The capital of France is '
Next token: 'Paris'
Complete: 'The capital of France is Paris'

🎉 Inference completed successfully!
```

## Performance Notes

-   **First run**: Downloads ~13GB of model weights (this will be cached locally)
-   **Memory requirements**: Requires ~13GB+ of system RAM
-   **GPU acceleration**: Uses Metal for faster inference on Apple hardware

## Customization

To use a different prompt, modify the `prompt` variable in `src/main.rs`:

```rust
let prompt = "Your custom prompt here: ";
```

## Troubleshooting

### Authentication Issues

-   Make sure you've accepted the Mistral model terms on HuggingFace
-   Verify your token has the correct permissions
-   Check that the HF_TOKEN environment variable is set correctly

### Memory Issues

-   Ensure you have enough RAM (13GB+ recommended)
-   Close other memory-intensive applications

### Metal Issues

-   Verify your Mac supports Metal
-   Try CPU-only mode by changing `Device::new_metal(0)?` to `Device::Cpu`

## Dependencies

-   `candle-core` - Core tensor operations with Metal support
-   `candle-nn` - Neural network layers
-   `candle-transformers` - Pre-built transformer models
-   `hf-hub` - HuggingFace Hub integration
-   `tokenizers` - Tokenization support
-   `tokio` - Async runtime
-   `serde_json` - JSON parsing

## License

This project is for educational purposes. Please respect the licenses of the underlying models and libraries.
