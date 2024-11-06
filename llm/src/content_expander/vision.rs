use crate::content_expander::prompts::{get_prompt, Prompts};
use anyhow::Context;
use linkify::{LinkFinder, LinkKind};
use mistralrs::{IsqType, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder};
use url::Url;

const VISION_MODEL_ID: &str = "microsoft/Phi-3.5-vision-instruct";

struct UrlParser {
    domains_allow_list: Vec<String>,
    domains_deny_list: Vec<String>,
}

#[derive(Default)]
struct UrlParserConfig {
    pub domains_allow_list: Vec<String>,
    pub domains_deny_list: Vec<String>,
}

impl UrlParser {
    fn try_new(params: UrlParserConfig) -> Result<Self, &'static str> {
        if params
            .domains_deny_list
            .iter()
            .any(|domain| params.domains_allow_list.contains(domain))
        {
            return Err("Domain allow and deny lists contain common values");
        }

        Ok(UrlParser {
            domains_allow_list: params.domains_allow_list,
            domains_deny_list: params.domains_deny_list,
        })
    }

    fn get_all_links(&self, text: &str) -> Vec<String> {
        let mut finder = LinkFinder::new();
        finder.kinds(&[LinkKind::Url]);
        finder
            .links(text)
            .filter_map(|link| {
                let url = Url::parse(link.as_str());
                let host = url.map(|uri| uri.host().map(|host| host.to_string()));
                match host {
                    Ok(Some(host)) => {
                        // By default, we allow all domains. If we insert at least one domain in the allow list, then we're
                        // restricting the valid domains to those present in the allow list only.
                        let is_allowed = self.domains_allow_list.is_empty()
                            || self.domains_allow_list.contains(&host);

                        // The allow list may be empty, but the deny list may have elements, so we check there too.
                        let is_denied = self.domains_deny_list.contains(&host);

                        // Case: domain found in the deny list.
                        if is_denied {
                            return None;
                        }

                        // Cases: empty allow list or valid domain found in the allow list.
                        if is_allowed {
                            return Some(link.as_str().to_string()); // Convert to owned String
                        }

                        // Case: allow list has some elements but the current domain is not in that list.
                        None
                    }

                    // Case: not a valid URL
                    _ => None,
                }
            })
            .collect()
    }
}

async fn fetch_image(url: String) -> anyhow::Result<Option<Vec<u8>>> {
    if !is_image(&url).await? {
        return Ok(None);
    }

    let client = reqwest::Client::new();

    let http_resp = client
        .get(url)
        .send()
        .await
        .context("Failed to send request")?;

    let bytes = http_resp
        .bytes()
        .await
        .context("Failed to get response bytes")?;

    Ok(Some(bytes.to_vec()))
}

async fn is_image(url: &str) -> anyhow::Result<bool> {
    let client = reqwest::Client::new();
    let res = client
        .head(url)
        .send()
        .await
        .context("Failed to send HEAD request")?;

    let content_type = res
        .headers()
        .get("Content-Type")
        .context("No Content-Type header")?
        .to_str()
        .context("Invalid Content-Type header")?;

    Ok(content_type.starts_with("image"))
}

pub async fn describe_images(
    text: String,
    domain: Prompts,
) -> Result<Vec<(String, String)>, anyhow::Error> {
    let model = VisionModelBuilder::new(VISION_MODEL_ID, VisionLoaderType::Phi3V)
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await?;

    let image_links = UrlParser::try_new(UrlParserConfig {
        domains_allow_list: vec![],
        domains_deny_list: vec![],
    })
    .unwrap()
    .get_all_links(&text);

    let domain_ref = &domain;
    let futures: Vec<_> = image_links
        .iter()
        .map(|link| async {
            match fetch_image(link.to_string()).await {
                Ok(Some(bytes)) => match image::load_from_memory(&bytes) {
                    Ok(image) => {
                        let messages = VisionMessages::new().add_phiv_image_message(
                            TextMessageRole::User,
                            get_prompt(domain_ref.clone()),
                            image,
                        );
                        match model.send_chat_request(messages).await {
                            Ok(response) => response
                                .choices
                                .first()
                                .and_then(|choice| choice.message.content.clone())
                                .map(|content| (link.to_string(), content)),
                            Err(_) => None,
                        }
                    }
                    Err(_) => None,
                },
                _ => None,
            }
        })
        .collect();

    let results: Vec<(String, String)> = futures_util::future::join_all(futures)
        .await
        .into_iter()
        .flatten()
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_all_links() {
        let text = r"
            Hello, world. This is an email and shouldn't be included: hello@example.com\
            This is a valid link https://github.com/oramasearch/orama. It should be included.\
            Now this is not a valid link www.foo.bar.baz.com as it misses the protocol\
            Let's also try with paths and query strings: https://myimage.com/v1/foo/bar.png?foo=hello&world=true
        ";

        let llama_vision = UrlParser::try_new(UrlParserConfig::default()).unwrap();

        let links = llama_vision.get_all_links(text);

        assert_eq!(links.len(), 2);
        assert_eq!(links[0], "https://github.com/oramasearch/orama");
        assert_eq!(
            links[1],
            "https://myimage.com/v1/foo/bar.png?foo=hello&world=true"
        );
    }

    #[test]
    fn test_allow_list() {
        let text = "Here's a series of links: https://github.com/oramasearch/orama, https://orama.com, https://twitter.com/oramasearch.";
        let llama_vision = UrlParser::try_new(UrlParserConfig {
            domains_allow_list: vec!["github.com".to_string(), "orama.com".to_string()],
            ..UrlParserConfig::default()
        })
        .unwrap();
        let links = llama_vision.get_all_links(text);

        assert_eq!(links.len(), 2);
        assert_eq!(links[0], "https://github.com/oramasearch/orama");
        assert_eq!(links[1], "https://orama.com");
    }

    #[test]
    fn test_deny_list() {
        let text = "Here's a series of links: https://github.com/oramasearch/orama, https://orama.com, https://twitter.com/oramasearch.";
        let llama_vision = UrlParser::try_new(UrlParserConfig {
            domains_deny_list: vec!["twitter.com".to_string()],
            ..UrlParserConfig::default()
        })
        .unwrap();
        let links = llama_vision.get_all_links(text);

        assert_eq!(links.len(), 2);
        assert_eq!(links[0], "https://github.com/oramasearch/orama");
        assert_eq!(links[1], "https://orama.com");
    }
}
