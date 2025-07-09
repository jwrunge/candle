use hf_hub::api::tokio::{Api, ApiBuilder};

pub async fn get() -> Api {
    let token = match std::env::var("HF_TOKEN") {
        Ok(token) => {
            println!("HF_TOKEN found");
            Some(token)
        }
        Err(_) => {
            println!("No HF_TOKEN found; some models may not be available");
            None
        }
    };

    if let Some(token) = token {
        ApiBuilder::new()
            .with_token(Some(token))
            .build()
            .expect("Failed to create API with token")
    } else {
        Api::new().expect("Failed to create API without token")
    }
}
