use web_server::WebServer;

fn main() -> std::io::Result<()> {

    let err = ::actix_web::rt::System::new().block_on(async move { 
        let web_server = WebServer::new();

        web_server.start().await
    });

    println!("{:?}", err);

    Ok(())
}