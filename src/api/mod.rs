use hf_hub::api::tokio::{Api, ApiBuilder};

pub struct ModelInfo {
    model_id: String,
    name: String,
    created_at: String,
    downloads: usize,
    likes: usize,
    private: bool,
    license: Option<String>,
    tags: Vec<String>,
    library_name: Option<String>,
    pipeline_tag: Option<String>,
}

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

fn model_json_to_model_info(model: &serde_json::Value) -> ModelInfo {
    ModelInfo {
        model_id: model
            .get("id")
            .and_then(|id| id.as_str())
            .unwrap_or("")
            .to_string(),
        name: model
            .get("name")
            .and_then(|name| name.as_str())
            .unwrap_or("")
            .to_string(),
        created_at: model
            .get("createdAt")
            .and_then(|date| date.as_str())
            .unwrap_or("")
            .to_string(),
        downloads: model.get("downloads").and_then(|d| d.as_u64()).unwrap_or(0) as usize,
        likes: model.get("likes").and_then(|l| l.as_u64()).unwrap_or(0) as usize,
        private: model
            .get("private")
            .and_then(|p| p.as_bool())
            .unwrap_or(false),
        license: model
            .get("license")
            .and_then(|l| l.as_str())
            .map(|s| s.to_string()),
        tags: model
            .get("tags")
            .and_then(|t| t.as_array())
            .map_or_else(Vec::new, |arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            }),
        library_name: model
            .get("libraryName")
            .and_then(|l| l.as_str())
            .map(String::from),
        pipeline_tag: model
            .get("pipeline_tag")
            .and_then(|p| p.as_str())
            .map(String::from),
    }
}

pub trait ApiExt {
    async fn search_models(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error>>;
    async fn list_popular_models(&self) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error>>;
    async fn print_search(&self, pretext: &str, models: Vec<ModelInfo>, print_result_count: bool) {
        println!("{}", pretext);
        if print_result_count {
            println!("Found {} models:", models.len());
        }
        for (i, model) in models.iter().enumerate() {
            println!(
                "  {}. {};\t\t\t{} likes, {} downloads",
                i + 1,
                model.model_id,
                model.likes,
                model.downloads
            );
        }
    }
}

impl ApiExt for Api {
    async fn search_models(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error>> {
        // Use reqwest to call the Hugging Face API directly
        let client = reqwest::Client::new();
        let limit = limit.unwrap_or(10);
        let url = format!(
            "https://huggingface.co/api/models?search={}&limit={}&filter=text-generation",
            urlencoding::encode(query),
            limit
        );
        let response = client.get(&url).send().await?;
        let models: serde_json::Value = response.json().await?;
        let mut model_info = Vec::new();

        if let Some(models_array) = models.as_array() {
            for model in models_array {
                model_info.push(model_json_to_model_info(model));
            }
        }

        Ok(model_info)
    }

    async fn list_popular_models(&self) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let url = "https://huggingface.co/api/models?filter=text-generation&sort=downloads&direction=-1&limit=20";
        let response = client.get(url).send().await?;
        let models: serde_json::Value = response.json().await?;
        let mut model_info = Vec::new();

        if let Some(models_array) = models.as_array() {
            for model in models_array {
                model_info.push(model_json_to_model_info(model));
            }
        }

        Ok(model_info)
    }
}
