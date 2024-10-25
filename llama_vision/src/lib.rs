extern crate core;

use linkify::{LinkFinder, LinkKind};
use std::fs::File;
use std::io::Write;
use tempdir::TempDir;
use url::Url;

#[derive(Default)]
struct LlamaVisionConfig {
    pub tmp_image_bucket: Option<String>,
    pub domains_allow_list: Vec<String>,
    pub domains_deny_list: Vec<String>,
}

struct LlamaVision {
    tmp_image_bucket: String,
    domains_allow_list: Vec<String>,
    domains_deny_list: Vec<String>,
}

impl LlamaVision {
    pub fn try_new(params: LlamaVisionConfig) -> Result<Self, &'static str> {
        if params
            .domains_deny_list
            .iter()
            .any(|domain| params.domains_allow_list.contains(domain))
        {
            return Err("Domain allow and deny lists contain common values");
        }

        let mut tmp_bucket_path = params.tmp_image_bucket.unwrap_or_else(|| {
            TempDir::new(".llama_vision")
                .expect("Unable to generate tmp directory for llama_vision")
                .path()
                .to_string_lossy()
                .to_string()
        });

        Ok(LlamaVision {
            domains_allow_list: params.domains_allow_list,
            domains_deny_list: params.domains_deny_list,
            tmp_image_bucket: tmp_bucket_path,
        })
    }

    fn get_all_links(&self, text: &str) -> Vec<String> {
        let mut finder = LinkFinder::new();
        finder.kinds(&[LinkKind::Url]);
        finder
            .links(text)
            .filter_map(|link| {
                let url = Url::parse(link.as_str());
                match url.unwrap().host() {
                    Some(host) => {
                        let host_as_str = host.to_string();

                        // By default, we allow all domains. If we insert at least one domain in the allow list, then we're
                        // restricting the valid domains to those present in the allow list only.
                        let is_allowed = self.domains_allow_list.is_empty()
                            || self.domains_allow_list.contains(&host_as_str);

                        // The allow list may be empty, but the deny list may have elements, so we check there too.
                        let is_denied = self.domains_deny_list.contains(&host_as_str);

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
                    None => None,
                }
            })
            .collect()
    }

    async fn fetch_image(&self, url: &str) -> Result<(), std::io::Error> {
        let tmp_file_path = format!("{}/{}", self.tmp_image_bucket, url);
        let mut file = File::create(tmp_file_path)?; // Use ? instead of unwrap()
        let response = reqwest::get(url)
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let image_bytes = response
            .bytes()
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        file.write_all(&image_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_all_links() {
        let text = "\
            Hello, world. This is an email and shouldn't be included: hello@example.com\
            This is a valid link https://github.com/oramasearch/orama. It should be included.\
            Now this is not a valid link www.foo.bar.baz.com as it misses the protocol\
            Let's also try with paths and query strings: https://myimage.com/v1/foo/bar.png?foo=hello&world=true
        ";

        let llama_vision = LlamaVision::try_new(LlamaVisionConfig::default()).unwrap();

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
        let llama_vision = LlamaVision::try_new(LlamaVisionConfig {
            domains_allow_list: vec!["github.com".to_string(), "orama.com".to_string()],
            ..LlamaVisionConfig::default()
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
        let llama_vision = LlamaVision::try_new(LlamaVisionConfig {
            domains_deny_list: vec!["twitter.com".to_string()],
            ..LlamaVisionConfig::default()
        })
        .unwrap();
        let links = llama_vision.get_all_links(text);

        assert_eq!(links.len(), 2);
        assert_eq!(links[0], "https://github.com/oramasearch/orama");
        assert_eq!(links[1], "https://orama.com");
    }
}
