use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use base64::Engine;
use duration_string::DurationString;
use jwt_authorizer::{AuthError, Authorizer, JwtAuthorizer, Refresh, RefreshStrategy, Validation};
use reqwest::Url;
use serde::{de::DeserializeOwned, Deserialize};
use thiserror::Error;

use crate::types::StackString;

#[derive(Error, Debug)]
pub enum JwtError {
    #[error("Auth error: {0:?}")]
    Generic(#[from] AuthError),
    #[error("Invalid issuer. Wanted: {wanted:?}")]
    InvalidIssuer {
        wanted: Option<Vec<StackString<128>>>,
    },
    #[error("Invalid audience. Wanted: {wanted:?}")]
    InvalidAudience { wanted: Vec<StackString<128>> },
    #[error("This instance is not configured to handle JWT")]
    NotConfigured,
    #[error("The JWT is expired")]
    ExpiredToken,
    #[error("Missing required claim {0}")]
    MissingRequiredClaim(String),
    #[error("No provider found for issuer: {issuer}. Configured providers: {providers:?}")]
    NoProviderForIssuer {
        issuer: String,
        providers: Vec<String>,
    },
    #[error("Unable to decode JWT claims to extract issuer: {reason}")]
    UnableToDecodeIssuer { reason: String },
    #[error("All providers failed for issuer {issuer}. Errors: {errors:?}")]
    AllProvidersFailed { issuer: String, errors: Vec<String> },
}

/// Configuration for a single JWKS provider
#[derive(Deserialize, Clone, Debug)]
pub struct JwksProviderConfig {
    /// Human-readable name for logging and debugging
    pub name: String,
    /// URL of the JWKS endpoint
    pub jwks_url: Url,
    /// List of valid issuers for this provider (required, non-empty)
    pub issuers: Vec<StackString<128>>,
    /// List of valid audiences for this provider
    pub audiences: Vec<StackString<128>>,
    /// Optional refresh interval for JWKS keys
    pub refresh_interval: Option<DurationString>,
}

/// Top-level JWT configuration containing multiple providers
#[derive(Deserialize, Clone, Debug)]
pub struct JwtConfig {
    /// List of JWKS providers to use for JWT validation
    pub providers: Vec<JwksProviderConfig>,
}

/// Internal representation of a built JWKS provider with its authorizer.
/// This struct is public due to enum visibility requirements but should be
/// considered an implementation detail.
pub struct JwksProvider<Claims: Clone + Send + Sync + DeserializeOwned> {
    /// Human-readable name for logging
    name: String,
    /// The JWT authorizer instance
    auth: Arc<Authorizer<Claims>>,
    /// Valid issuers for this provider
    issuers: Vec<StackString<128>>,
    /// Valid audiences for this provider
    audiences: Vec<StackString<128>>,
}

/// Minimal claims structure for extracting the issuer without full validation
#[derive(Deserialize)]
struct MinimalClaims {
    iss: String,
}

/// Extracts the issuer claim from a JWT without performing signature validation.
///
/// JWTs have three parts separated by dots: header.payload.signature
/// We decode only the payload (second part) using base64url decoding.
fn extract_issuer_from_token(token: &str) -> Result<String, JwtError> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(JwtError::UnableToDecodeIssuer {
            reason: "JWT must have exactly 3 parts separated by dots".to_string(),
        });
    }

    // Decode the payload (second part) using base64url
    let payload_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|e| JwtError::UnableToDecodeIssuer {
            reason: format!("Failed to decode base64 payload: {e}"),
        })?;

    let minimal_claims: MinimalClaims =
        serde_json::from_slice(&payload_bytes).map_err(|e| JwtError::UnableToDecodeIssuer {
            reason: format!("Failed to parse claims JSON: {e}"),
        })?;

    Ok(minimal_claims.iss)
}

#[derive(Clone)]
pub enum JwtManager<Claims: Clone + Send + Sync + DeserializeOwned> {
    Enabled {
        /// Map from issuer string to list of providers that accept that issuer
        /// Multiple providers can share the same issuer
        providers_by_issuer: Arc<HashMap<String, Vec<Arc<JwksProvider<Claims>>>>>,
        /// All provider names for error messages
        provider_names: Vec<String>,
    },
    Disabled,
}

impl<Claims: Clone + Send + Sync + DeserializeOwned> JwtManager<Claims> {
    pub async fn new(config: Option<JwtConfig>) -> Result<Self> {
        let Some(config) = config else {
            return Ok(Self::Disabled);
        };

        if config.providers.is_empty() {
            return Ok(Self::Disabled);
        }

        let mut built_providers: Vec<Arc<JwksProvider<Claims>>> = Vec::new();
        let mut provider_names: Vec<String> = Vec::new();

        for provider_config in config.providers {
            if provider_config.issuers.is_empty() {
                anyhow::bail!(
                    "Provider '{}' must have at least one issuer configured",
                    provider_config.name
                );
            }

            tracing::info!(
                "Initializing JWKS provider '{}' with URL: {}",
                provider_config.name,
                provider_config.jwks_url
            );

            let validation = Validation::new()
                .aud(&provider_config.audiences)
                .iss(&provider_config.issuers)
                .exp(true);

            let mut refresh = Refresh {
                strategy: RefreshStrategy::Interval,
                ..Default::default()
            };
            if let Some(r) = provider_config.refresh_interval {
                refresh.refresh_interval = *r;
            }

            let auth = JwtAuthorizer::from_jwks_url(provider_config.jwks_url.as_str())
                .validation(validation)
                .refresh(refresh)
                .build()
                .await
                .with_context(|| {
                    format!(
                        "Cannot build jwt authorizer for provider '{}'",
                        provider_config.name
                    )
                })?;

            provider_names.push(provider_config.name.clone());
            built_providers.push(Arc::new(JwksProvider {
                name: provider_config.name,
                auth: Arc::new(auth),
                issuers: provider_config.issuers,
                audiences: provider_config.audiences,
            }));
        }

        let mut providers_by_issuer: HashMap<String, Vec<Arc<JwksProvider<Claims>>>> =
            HashMap::new();
        for provider in &built_providers {
            for issuer in &provider.issuers {
                providers_by_issuer
                    .entry(issuer.to_string())
                    .or_default()
                    .push(Arc::clone(provider));
            }
        }

        tracing::info!(
            "JWT manager initialized with {} providers, {} unique issuers",
            built_providers.len(),
            providers_by_issuer.len()
        );

        Ok(Self::Enabled {
            providers_by_issuer: Arc::new(providers_by_issuer),
            provider_names,
        })
    }

    pub async fn check(&self, token: &str) -> Result<Claims, JwtError> {
        let Self::Enabled {
            providers_by_issuer,
            provider_names,
        } = self
        else {
            return Err(JwtError::NotConfigured);
        };

        let issuer = extract_issuer_from_token(token)?;
        tracing::debug!("Extracted issuer from JWT: {}", issuer);

        let matching_providers = providers_by_issuer.get(&issuer).ok_or_else(|| {
            tracing::warn!(
                "No provider found for issuer '{}'. Configured providers: {:?}",
                issuer,
                provider_names
            );
            JwtError::NoProviderForIssuer {
                issuer: issuer.clone(),
                providers: provider_names.clone(),
            }
        })?;

        tracing::debug!(
            "Found {} provider(s) for issuer '{}': {:?}",
            matching_providers.len(),
            issuer,
            matching_providers
                .iter()
                .map(|p| &p.name)
                .collect::<Vec<_>>()
        );

        let mut errors: Vec<String> = Vec::new();
        for provider in matching_providers {
            tracing::debug!("Trying provider '{}' for token validation", provider.name);

            match provider.auth.check_auth(token).await {
                Ok(output) => {
                    tracing::debug!("Provider '{}' successfully validated token", provider.name);
                    return Ok(output.claims);
                }
                Err(e) => {
                    let error_msg = Self::format_auth_error(&e, provider);
                    tracing::debug!(
                        "Provider '{}' failed validation: {}",
                        provider.name,
                        error_msg
                    );
                    errors.push(format!("{}: {}", provider.name, error_msg));
                }
            }
        }

        Err(JwtError::AllProvidersFailed { issuer, errors })
    }

    fn format_auth_error(e: &AuthError, provider: &JwksProvider<Claims>) -> String {
        if let AuthError::InvalidToken(jwt_err) = e {
            use jsonwebtoken::errors::ErrorKind;
            match jwt_err.kind() {
                ErrorKind::ExpiredSignature => "token expired".to_string(),
                ErrorKind::InvalidIssuer => {
                    format!("invalid issuer (expected: {:?})", provider.issuers)
                }
                ErrorKind::InvalidAudience => {
                    format!("invalid audience (expected: {:?})", provider.audiences)
                }
                ErrorKind::MissingRequiredClaim(c) => {
                    format!("missing required claim: {c}")
                }
                _ => format!("{e:?}"),
            }
        } else {
            format!("{e:?}")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{SocketAddr, TcpListener};

    use axum::{Json, Router};
    use chrono::Utc;
    use jsonwebtoken::{encode, EncodingKey, Header};
    use serde::Serialize;
    use serde_json::json;

    use crate::types::{ClaimLimits, CollectionId, DashboardClaims};

    use super::*;

    async fn jwks_handler() -> Json<serde_json::Value> {
        Json(json!({
            "keys": [
                {
                    "kty": "oct",
                    "kid": "my-symmetric-key-id",
                    "k": "Zm9v",  // "foo" in base64url-encoded: it is the super secret key
                    "alg": "HS256",
                    "use": "sig"
                }
            ]
        }))
    }

    async fn generate_jwt(claims: Json<(u64, DashboardClaims)>) -> Json<serde_json::Value> {
        #[derive(Serialize)]
        struct MyClaims {
            #[serde(flatten)]
            inner: DashboardClaims,
            exp: u64,
        }

        let c = MyClaims {
            inner: claims.0 .1,
            exp: claims.0 .0,
        };

        let token = encode(
            &Header::default(),
            &c,
            &EncodingKey::from_secret("foo".as_ref()),
        )
        .unwrap();

        Json(serde_json::Value::String(token))
    }

    async fn start_http_issuer_server() -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let router = Router::new()
            .route(
                "/api/.well-known/jwks.json",
                axum::routing::get(jwks_handler),
            )
            .route("/jwt/generate", axum::routing::post(generate_jwt));

        tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
            axum::serve(listener, router).await.unwrap();
        });

        addr
    }

    #[tokio::test]
    async fn test_jwt() {
        let address = start_http_issuer_server().await;
        let port = address.port();

        let jwt_manager = JwtManager::<DashboardClaims>::new(Some(JwtConfig {
            providers: vec![JwksProviderConfig {
                name: "test-provider".to_string(),
                jwks_url: format!("http://localhost:{port}/api/.well-known/jwks.json")
                    .parse()
                    .unwrap(),
                refresh_interval: None,
                issuers: vec![StackString::try_new("https://the-dashboard").unwrap()],
                audiences: vec![StackString::try_new("http://the-orama-core").unwrap()],
            }],
        }))
        .await
        .unwrap();

        // Ok
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                Utc::now().timestamp() as u64,
                DashboardClaims {
                    aud: StackString::try_new("http://the-orama-core").unwrap(),
                    iss: StackString::try_new("https://the-dashboard").unwrap(),
                    scope: StackString::try_new("write").unwrap(),
                    limits: ClaimLimits { max_doc_count: 1 },
                    sub: CollectionId::try_new("coll-id").unwrap(),
                },
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let output = jwt_manager.check(&jwt).await;
        assert!(output.is_ok());

        // Bad aud - token has wrong audience, provider validation will fail
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                Utc::now().timestamp() as u64,
                DashboardClaims {
                    aud: StackString::try_new("BAD").unwrap(),
                    iss: StackString::try_new("https://the-dashboard").unwrap(),
                    scope: StackString::try_new("write").unwrap(),
                    limits: ClaimLimits { max_doc_count: 1 },
                    sub: CollectionId::try_new("coll-id").unwrap(),
                },
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let output = jwt_manager.check(&jwt).await.unwrap_err();
        // All providers failed because the audience doesn't match
        assert!(matches!(output, JwtError::AllProvidersFailed { .. }));

        // Bad iss - no provider configured for this issuer
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                Utc::now().timestamp() as u64,
                DashboardClaims {
                    aud: StackString::try_new("http://the-orama-core").unwrap(),
                    iss: StackString::try_new("BAD").unwrap(),
                    scope: StackString::try_new("write").unwrap(),
                    limits: ClaimLimits { max_doc_count: 1 },
                    sub: CollectionId::try_new("coll-id").unwrap(),
                },
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let output = jwt_manager.check(&jwt).await.unwrap_err();
        // No provider found for the issuer "BAD"
        assert!(matches!(output, JwtError::NoProviderForIssuer { .. }));

        // expired
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                0,
                DashboardClaims {
                    aud: StackString::try_new("http://the-orama-core").unwrap(),
                    iss: StackString::try_new("https://the-dashboard").unwrap(),
                    scope: StackString::try_new("write").unwrap(),
                    limits: ClaimLimits { max_doc_count: 1 },
                    sub: CollectionId::try_new("coll-id").unwrap(),
                },
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let output = jwt_manager.check(&jwt).await.unwrap_err();
        // All providers failed because the token expired
        assert!(matches!(output, JwtError::AllProvidersFailed { .. }));
    }

    #[tokio::test]
    async fn test_multi_provider_routing() {
        let address = start_http_issuer_server().await;
        let port = address.port();

        // Create two providers with different issuers
        let jwt_manager = JwtManager::<DashboardClaims>::new(Some(JwtConfig {
            providers: vec![
                JwksProviderConfig {
                    name: "provider-one".to_string(),
                    jwks_url: format!("http://localhost:{port}/api/.well-known/jwks.json")
                        .parse()
                        .unwrap(),
                    refresh_interval: None,
                    issuers: vec![StackString::try_new("https://issuer-one").unwrap()],
                    audiences: vec![StackString::try_new("http://the-orama-core").unwrap()],
                },
                JwksProviderConfig {
                    name: "provider-two".to_string(),
                    jwks_url: format!("http://localhost:{port}/api/.well-known/jwks.json")
                        .parse()
                        .unwrap(),
                    refresh_interval: None,
                    issuers: vec![StackString::try_new("https://issuer-two").unwrap()],
                    audiences: vec![StackString::try_new("http://the-orama-core").unwrap()],
                },
            ],
        }))
        .await
        .unwrap();

        // Token with issuer-one should be routed to provider-one
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                Utc::now().timestamp() as u64,
                DashboardClaims {
                    aud: StackString::try_new("http://the-orama-core").unwrap(),
                    iss: StackString::try_new("https://issuer-one").unwrap(),
                    scope: StackString::try_new("write").unwrap(),
                    limits: ClaimLimits { max_doc_count: 1 },
                    sub: CollectionId::try_new("coll-id").unwrap(),
                },
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let output = jwt_manager.check(&jwt).await;
        assert!(output.is_ok());

        // Token with issuer-two should be routed to provider-two
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                Utc::now().timestamp() as u64,
                DashboardClaims {
                    aud: StackString::try_new("http://the-orama-core").unwrap(),
                    iss: StackString::try_new("https://issuer-two").unwrap(),
                    scope: StackString::try_new("write").unwrap(),
                    limits: ClaimLimits { max_doc_count: 1 },
                    sub: CollectionId::try_new("coll-id").unwrap(),
                },
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let output = jwt_manager.check(&jwt).await;
        assert!(output.is_ok());

        // Token with unknown issuer should fail
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                Utc::now().timestamp() as u64,
                DashboardClaims {
                    aud: StackString::try_new("http://the-orama-core").unwrap(),
                    iss: StackString::try_new("https://unknown-issuer").unwrap(),
                    scope: StackString::try_new("write").unwrap(),
                    limits: ClaimLimits { max_doc_count: 1 },
                    sub: CollectionId::try_new("coll-id").unwrap(),
                },
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let output = jwt_manager.check(&jwt).await.unwrap_err();
        assert!(matches!(output, JwtError::NoProviderForIssuer { .. }));
    }

    #[tokio::test]
    async fn test_jwt_with_invalid_signature_rejected() {
        let address = start_http_issuer_server().await;
        let port = address.port();

        let jwt_manager = JwtManager::<DashboardClaims>::new(Some(JwtConfig {
            providers: vec![JwksProviderConfig {
                name: "test-provider".to_string(),
                jwks_url: format!("http://localhost:{port}/api/.well-known/jwks.json")
                    .parse()
                    .unwrap(),
                refresh_interval: None,
                issuers: vec![StackString::try_new("https://the-dashboard").unwrap()],
                audiences: vec![StackString::try_new("http://the-orama-core").unwrap()],
            }],
        }))
        .await
        .unwrap();

        // Create JWT signed with WRONG key ("wrong_secret" instead of "foo")
        // The JWKS server has key "foo", so this signature won't validate
        #[derive(Serialize)]
        struct MyClaims {
            #[serde(flatten)]
            inner: DashboardClaims,
            exp: u64,
        }

        let claims = MyClaims {
            inner: DashboardClaims {
                aud: StackString::try_new("http://the-orama-core").unwrap(),
                iss: StackString::try_new("https://the-dashboard").unwrap(),
                scope: StackString::try_new("write").unwrap(),
                limits: ClaimLimits { max_doc_count: 1 },
                sub: CollectionId::try_new("coll-id").unwrap(),
            },
            exp: Utc::now().timestamp() as u64 + 3600, // Valid expiration (1 hour from now)
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(b"wrong_secret"), // Wrong key!
        )
        .unwrap();

        let result = jwt_manager.check(&token).await;
        assert!(
            matches!(result, Err(JwtError::AllProvidersFailed { .. })),
            "Expected AllProvidersFailed error for invalid signature, got: {result:?}"
        );
    }
}
