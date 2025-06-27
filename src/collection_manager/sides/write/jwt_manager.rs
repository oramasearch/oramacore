use std::sync::Arc;

use anyhow::{Context, Result};
use duration_string::DurationString;
use jwt_authorizer::{AuthError, Authorizer, JwtAuthorizer, Refresh, RefreshStrategy, Validation};
use reqwest::Url;
use serde::Deserialize;
use thiserror::Error;

use crate::types::{Claims, StackString};

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
}

#[derive(Deserialize, Clone, Debug)]
pub struct JwtConfig {
    jwks_url: Url,
    refresh_interval: Option<DurationString>,
    issuers: Option<Vec<StackString<128>>>,

    // Unable to use Option<..> due to a bug
    // https://github.com/cduvray/jwt-authorizer/pull/48
    audiences: Vec<StackString<128>>,
}

#[derive(Clone)]
pub enum JwtManager {
    Enabled {
        auth: Arc<Authorizer<Claims>>,
        audiences: Vec<StackString<128>>,
        issuers: Option<Vec<StackString<128>>>,
    },
    Disabled,
}

impl JwtManager {
    pub async fn new(config: Option<JwtConfig>) -> Result<Self> {
        let Some(config) = config else {
            return Ok(Self::Disabled);
        };

        let validation = Validation::new().aud(&config.audiences).exp(true);
        let validation = if let Some(issuers) = config.issuers.as_ref() {
            validation.iss(issuers)
        } else {
            validation
        };

        let mut refresh = Refresh {
            strategy: RefreshStrategy::Interval,
            ..Default::default()
        };
        if let Some(r) = config.refresh_interval {
            refresh.refresh_interval = *r;
        }

        let auth = JwtAuthorizer::from_jwks_url(config.jwks_url.as_str())
            .validation(validation)
            .refresh(refresh)
            .build()
            .await
            .context("Cannot build jwt authorizer")?;

        Ok(Self::Enabled {
            auth: Arc::new(auth),
            audiences: config.audiences,
            issuers: config.issuers,
        })
    }

    pub async fn check(&self, token: &str) -> Result<Claims, JwtError> {
        let Self::Enabled {
            auth,
            audiences,
            issuers,
        } = self
        else {
            return Err(JwtError::NotConfigured);
        };

        let output = auth.check_auth(token).await;

        let output = match output {
            Ok(output) => output,
            Err(e) => {
                println!("AuthError {:?}", e);

                use jsonwebtoken::errors::ErrorKind;

                if let AuthError::InvalidToken(e) = &e {
                    match e.kind() {
                        ErrorKind::ExpiredSignature => return Err(JwtError::ExpiredToken),
                        ErrorKind::InvalidIssuer => {
                            return Err(JwtError::InvalidIssuer {
                                wanted: issuers.clone(),
                            })
                        }
                        ErrorKind::InvalidAudience => {
                            return Err(JwtError::InvalidAudience {
                                wanted: audiences.clone(),
                            })
                        }
                        ErrorKind::MissingRequiredClaim(c) => {
                            return Err(JwtError::MissingRequiredClaim(c.clone()))
                        }
                        _ => {}
                    }
                }
                return Err(JwtError::Generic(e));
            }
        };

        let claims = output.claims;

        Ok(claims)
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

    use crate::types::{ClaimLimits, CollectionId};

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

    async fn generate_jwt(claims: Json<(u64, Claims)>) -> Json<serde_json::Value> {
        #[derive(Serialize)]
        struct MyClaims {
            #[serde(flatten)]
            inner: Claims,
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

        let jwt_manager = JwtManager::new(Some(JwtConfig {
            jwks_url: format!("http://localhost:{port}/api/.well-known/jwks.json")
                .parse()
                .unwrap(),
            refresh_interval: None,
            issuers: Some(Vec::from_iter([StackString::try_new(
                "https://the-dashboard",
            )
            .unwrap()])),
            audiences: Vec::from_iter([StackString::try_new("http://the-orama-core").unwrap()]),
        }))
        .await
        .unwrap();

        // Ok
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                Utc::now().timestamp() as u64,
                Claims {
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

        // Bad aud
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                Utc::now().timestamp() as u64,
                Claims {
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
        assert!(matches!(output, JwtError::InvalidAudience { .. }));

        // Bad iss
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                Utc::now().timestamp() as u64,
                Claims {
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
        assert!(matches!(output, JwtError::InvalidIssuer { .. }));

        // expired
        let jwt: String = reqwest::Client::new()
            .post(format!("http://localhost:{port}/jwt/generate"))
            .json(&(
                0,
                Claims {
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
        assert!(matches!(output, JwtError::ExpiredToken));
    }
}
