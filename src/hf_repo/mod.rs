use std::path::PathBuf;

use hf_hub::api::tokio::{Api, ApiError};

mod models;

pub async fn get_model(model_id: String) -> Result<(PathBuf, Option<PathBuf>), ApiError> {
    let available_models = models::get();
    let model_details = available_models
        .get(&model_id)
        .expect("Model was not found in the model map");

    let api = Api::new().unwrap();
    let repo = api.model(model_id);

    let weights_path = repo.get(&model_details.weights_filename).await?;
    let tokenizer_path = match model_details.tokenizer_filename {
        Some(filename) => Some(repo.get(filename).await?),
        None => None,
    };

    Ok((weights_path, tokenizer_path))
}
