use anyhow::Error;
use deno_core::{JsRuntime, RuntimeOptions};
use std::sync::{mpsc};
use std::thread;
use tokio::sync::oneshot;

pub struct JavaScript {
    sender: mpsc::Sender<Job>,
}

struct Job {
    code: String,
    response: oneshot::Sender<Result<String, Error>>,
}

impl JavaScript {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();

        thread::spawn(move || {
            let mut runtime = JsRuntime::new(RuntimeOptions::default());

            for job in receiver {
                let result = runtime.execute_script("script.js", job.code)
                    .map(|_| "Execution succeeded".to_string());
                let _ = job.response.send(result);
            }
        });

        Self { sender }
    }

    pub async fn eval(&self, code: String) -> Result<String, Error> {
        let (response_tx, response_rx) = oneshot::channel();
        let job = Job {
            code,
            response: response_tx,
        };

        self.sender.send(job).map_err(|_| Error::msg("Runtime thread disconnected"))?;
        response_rx.await?
    }
}
