use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use hf_hub::api::tokio::Api;

#[tokio::main]
async fn main() {
    let api = Api::new().unwrap();
    let repo = api.model("bert-base-uncased".to_string());

    let weights_filename = repo
        .get("model.safetensors")
        .await
        .expect("Failed to download weights");
    let device = Device::new_metal(0).expect("Failed to create Metal device");
    let weights = candle_core::safetensors::load(weights_filename, &device).unwrap();

    let weight = weights
        .get("bert.encoder.layer.0.attention.self.query.weight")
        .unwrap();
    let bias = weights
        .get("bert.encoder.layer.0.attention.self.query.bias")
        .unwrap();

    let linear = Linear::new(weight.clone(), Some(bias.clone()));

    let input_ids = Tensor::zeros((3, 768), DType::F32, &device).unwrap();
    let output = linear.forward(&input_ids).unwrap();

    println!("Output shape: {:?}", output.shape());
}
