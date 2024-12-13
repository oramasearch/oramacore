use anyhow::Error;
use rustorama::js::deno::JavaScript;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let js = JavaScript::new();

    let mut timings = Vec::new();

    for i in 1..=10 {
        let code = r#"
            function foo() {
                const x = 10;
                const y = 20;
                const result = x * y;
                console.log(`when x=${x} and y=${y}, x*y=${result}`);
            }

            foo()
        "#.to_string();

        let start = Instant::now();
        let result = js.eval(code).await?;
        let duration = start.elapsed();
        println!("Call {}: JavaScript result: {} (Duration: {:?})", i, result, duration);
        timings.push(duration);
    }

    println!("\nTimings for all calls:");
    for (i, timing) in timings.iter().enumerate() {
        println!("Call {}: {:?}", i + 1, timing);
    }

    Ok(())
}
