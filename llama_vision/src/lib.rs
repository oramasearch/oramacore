use linkify::{LinkFinder, LinkKind};
use mistralrs::{IsqType, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder};
use url::Url;
mod fetcher;
mod prompts;

const MODEL_ID: &str = "microsoft/Phi-3.5-vision-instruct";

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

pub async fn describe_images(text: String) -> Result<Vec<(String, String)>, anyhow::Error> {
    let model = VisionModelBuilder::new(MODEL_ID, VisionLoaderType::Phi3V)
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

    let futures: Vec<_> = image_links
        .iter()
        .map(|link| async {
            match fetcher::fetch_image(link.to_string()).await {
                Ok(Some(bytes)) => match image::load_from_memory(&bytes) {
                    Ok(image) => {
                        let messages = VisionMessages::new().add_phiv_image_message(
                            TextMessageRole::User,
                            prompts::Prompts::get_prompt(prompts::Prompts::ECommerce),
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
        .filter_map(|x| x)
        .collect();

    Ok(results)
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let example_json: &str = "[\
        {\
            \"title\": \"Volcom flame hoodie\",\
            \"price\": 98.99,\
            \"image\": \"https://img01.ztat.net/article/spp-media-p1/32ab3d77233b4e92a8939cfae29694e4/c31db6a545cf4e548a9e4f5f38ac9c2a.jpg?imwidth=520\"
        },\
        {\
            \"title\": \"Solar Guitars A2.7 Canibalismo+\",\
            \"price\": 899.99,\
            \"image\": \"https://thumbs.static-thomann.de/thumb/padthumb600x600/pics/bdb/_54/544771/19211328_800.jpg\"
        },\
        {\
            \"title\": \"Apple AirPods 4 \",\
            \"price\": 187.99,\
            \"image\": \"https://m.media-amazon.com/images/I/61DvMw16ITL.__AC_SY445_SX342_QL70_ML2_.jpg\"
        },\
    ]";

    let results = describe_images(example_json.to_string()).await?;
    dbg!(results);

    Ok(())
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
