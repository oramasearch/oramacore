#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use anyhow::Result;

    use serde_json::{json, Value};
    use tokio::time::sleep;

    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_deno2() -> Result<()> {
        enum JsRuntimeMessage {
            AddFunction {
                name: String,
                function_code: String,
            },
            Job {
                name: String,
                input: String,
                response: tokio::sync::oneshot::Sender<Value>,
            },
            Terminate,
        }

        async fn foo(receiver: mpsc::Receiver<JsRuntimeMessage>) {
            let mut runtime = JsRuntime::new(RuntimeOptions::default());

            for message in receiver {
                match message {
                    JsRuntimeMessage::AddFunction {
                        name,
                        function_code,
                    } => {
                        // runtime.load_main_es_module_from_code(specifier)
                        let output = runtime
                            .execute_script(
                                "script.js",
                                format!(
                                    r#"
const {} = {};
"#,
                                    name, function_code
                                ),
                            )
                            .unwrap();
                    }
                    JsRuntimeMessage::Terminate => {
                        break;
                    }
                    JsRuntimeMessage::Job {
                        name,
                        input,
                        response,
                    } => {
                        let output = runtime
                            .execute_script(
                                "script.js",
                                format!(
                                    r#"
JSON.stringify({name}({input}))
"#
                                ),
                            )
                            .unwrap();
                        let scope = &mut runtime.handle_scope();
                        let local = output.open(scope);
                        if let Some(js_string) = local.to_string(scope) {
                            response
                                .send(
                                    serde_json::from_str(&js_string.to_rust_string_lossy(scope))
                                        .unwrap(),
                                )
                                .unwrap();
                        } else {
                            println!("Failed to convert JavaScript value to string");
                        }
                    }
                };
            }

            println!("Terminating runtime");

            drop(runtime);
        }

        let (sender, receiver) = mpsc::channel::<JsRuntimeMessage>();

        let handler = tokio::spawn(async move {
            foo(receiver).await;
        });

        sender.send(JsRuntimeMessage::AddFunction {
            name: "my_map".to_string(),
            function_code: r#"
function(input) {
        const doc = {
            ...input,
            price_description: input.price > 75 ? "expensive" : "not expensive"
        }

        return doc;
}
        "#
            .to_string(),
        })?;

        println!("sleeping for 1 second");
        sleep(Duration::from_secs(1)).await;
        println!("awake");

        async fn send_and_recv(
            sender: &mpsc::Sender<JsRuntimeMessage>,
            doc: Value,
        ) -> Result<(Value, Duration)> {
            let start = Instant::now();
            let output = {
                let (response_tx, response_rx) = tokio::sync::oneshot::channel();
                sender.send(JsRuntimeMessage::Job {
                    name: "my_map".to_string(),
                    input: serde_json::to_string(&doc).unwrap(),
                    response: response_tx,
                })?;
                response_rx.await?
            };
            let duration = start.elapsed();
            Ok((output, duration))
        }

        for i in 1..=10 {
            let d = send_and_recv(
                &sender,
                json!({
                    "title": "Wireless headphones".to_string(),
                    "description": "These are some beautiful wireless headphones".to_string(),
                    "favorite": true,
                    "price": 100,
                }),
            )
            .await?;
            println!("Call JavaScript result: {} (Duration: {:?})", i, d);
        }

        println!("sleeping for 1 second");
        sleep(Duration::from_secs(1)).await;
        println!("awake");

        sender.send(JsRuntimeMessage::Terminate)?;

        Ok(())
    }
}
