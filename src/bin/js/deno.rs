use anyhow::Error;
use oramacore::js::deno::{JavaScript, Operation};
use rand::Rng;
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize, Clone)]
struct Document {
    title: String,
    description: String,
    favorite: bool,
    price: usize,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let js = JavaScript::new();

    let mut timings = Vec::new();

    for i in 1..=10 {
        let input = Document {
            title: "Wireless headphones".to_string(),
            description: "These are some beautiful wireless headphones".to_string(),
            favorite: true,
            price: rand::thread_rng().gen_range(50..100),
        };

        let code = r#"
            function(input) {
                const doc = {
                    ...input,
                    price_description: input.price > 75 ? "expensive" : "not expensive"
                }

                return doc;
            }
        "#
        .to_string();

        let start = Instant::now();
        let result = js
            .eval(Operation::Anonymous, code.clone(), input.clone())
            .await?;
        let duration = start.elapsed();
        println!(
            "Call {}: JavaScript result: {} (Duration: {:?})",
            i, result, duration
        );
        timings.push(duration);
    }

    println!("\nTimings for all calls:");
    for (i, timing) in timings.iter().enumerate() {
        println!("Call {}: {:?}", i + 1, timing);
    }

    Ok(())
}
