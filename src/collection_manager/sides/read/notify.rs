use core::fmt;
use std::time::Duration;

use anyhow::{Context, Result};
use duration_str::deserialize_duration;
use http::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::types::CollectionId;

#[derive(Debug, Clone, Deserialize)]
pub struct NotifierConfig {
    pub url: Url,
    pub authorization_token: Option<String>,
    #[serde(deserialize_with = "deserialize_duration")]
    pub timeout: Duration,
    #[serde(default = "default_retry_count")]
    pub retry_count: u8,
}

pub struct Notifier {
    url: Url,
    client: reqwest::Client,
    retry_count: u8,
}

impl fmt::Debug for Notifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Notifier")
            .field("url", &self.url)
            .field("retry_count", &self.retry_count)
            .field("client", &"...")
            .finish()
    }
}

impl Notifier {
    pub fn try_new(config: &NotifierConfig) -> Result<Self> {
        let mut authorization_token = None;
        if let Some(token) = &config.authorization_token {
            authorization_token =
                Some(HeaderValue::from_str(token).context("Failed to parse authorization token")?);
        }

        let mut headers: HeaderMap = [
            (
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("my-app"),
            ),
            (
                HeaderName::from_static("content-type"),
                HeaderValue::from_static("application/json"),
            ),
        ]
        .into_iter()
        .collect();
        if let Some(authorization) = authorization_token {
            headers.insert(HeaderName::from_static("authorization"), authorization);
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            url: config.url.clone(),
            client,
            retry_count: config.retry_count,
        })
    }

    pub async fn notify_collection_substitution(
        &self,
        target_collection: CollectionId,
        source_collection: CollectionId,
        reference: Option<String>,
    ) -> Result<()> {
        self.notify(Notification::CollectionSubstituted {
            target_collection,
            source_collection,
            reference,
        })
        .await
    }

    async fn notify(&self, notification: Notification) -> Result<()> {
        for _ in 0..self.retry_count {
            let response = self
                .client
                .post(self.url.clone())
                .json(&notification)
                .send()
                .await
                .context("Failed to send HTTP request")?;

            match response.error_for_status() {
                Ok(_) => {
                    info!("Notification sent successfully");
                    return Ok(());
                }
                Err(err) => {
                    warn!(error = ?err, "Failed to send notification to {}", self.url);
                }
            }
        }
        anyhow::bail!(
            "Failed to send notification after {} attempts",
            self.retry_count
        )
    }
}

fn default_retry_count() -> u8 {
    3
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
enum Notification {
    CollectionSubstituted {
        target_collection: CollectionId,
        source_collection: CollectionId,
        reference: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_serialization() {
        let notification = Notification::CollectionSubstituted {
            target_collection: CollectionId::try_new("target_collection").unwrap(),
            source_collection: CollectionId::try_new("source_collection").unwrap(),
            reference: Some("reference".to_string()),
        };
        let json = serde_json::to_string(&notification).unwrap();
        println!("Serialized JSON: {}", json);
    }
}
