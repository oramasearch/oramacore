use anyhow;
use axum::{extract::{MatchedPath, State}, http::{Request, StatusCode}, response::IntoResponse, routing::{get, post}, Json, Router};
use axum_macros::debug_handler;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tower_http::trace::TraceLayer;
use tracing::{info, info_span, warn, Instrument};
use std::{collections::HashMap, fmt::Debug, net::{IpAddr, Ipv4Addr, SocketAddr}, sync::Arc, time::Duration};
use tokio::{net::{TcpListener, TcpStream}, sync::{mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender}, RwLock}, time::sleep};
use tokio_serde::{formats::SymmetricalBincode, SymmetricallyFramed};
use tokio_util::{
    codec::{FramedRead, FramedWrite, LengthDelimitedCodec},
    task::TaskTracker,
};
use types::CollectionId;

pub fn api_config<S: Clone + Send + Sync + 'static>() -> Router<S> {
    // build our application with a route
    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
}

#[derive(Debug, Serialize, Deserialize, Clone)]
enum WalOperation {
    CollectionCreated(CollectionId),
}

struct Collection {
    id: CollectionId,
}

struct WriterServer {
    collections: RwLock<HashMap<CollectionId, Collection>>,
    sender: UnboundedSender<WalOperation>,
}

impl Debug for WriterServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriterServer")
            // .field("collections", &self.collections)
            .finish()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CreateCollectionOptionDTO {
    collection_id: String,
}

#[tracing::instrument]
async fn create_collection(
    write_server: State<Arc<WriterServer>>,
    Json(json): Json<CreateCollectionOptionDTO>,
) -> Result<(StatusCode, impl IntoResponse), (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(json.collection_id);

    let mut collections = write_server.collections.write().await;

    if collections.contains_key(&collection_id) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "collection already exists" })),
        ));
    }
    collections.insert(collection_id.clone(), Collection { id: collection_id.clone() });
    drop(collections);

    write_server.sender.send(WalOperation::CollectionCreated(collection_id.clone()))
        .expect("Failed to send to WAL");

    Ok((StatusCode::CREATED, Json(json!({ "collection_id": collection_id }))))
}

fn write_api_config() -> Router<Arc<WriterServer>> {
    let router: Router<Arc<WriterServer>> = api_config();

    let collections_router: Router<Arc<WriterServer>> = Router::new()
        .route("/", post(create_collection));

    let router: Router<Arc<WriterServer>> = router.nest("/v0/collections", collections_router);

    router
        .layer(
        TraceLayer::new_for_http()
            .make_span_with(|request: &Request<_>| {
                // Log the matched route's path (with placeholders not filled in).
                // Use request.uri() or OriginalUri if you want the real path.
                let matched_path = request
                    .extensions()
                    .get::<MatchedPath>()
                    .map(MatchedPath::as_str);

                info_span!(
                    "http_request",
                    method = ?request.method(),
                    matched_path,
                    some_other_field = tracing::field::Empty,
                )
            })
        )
}

static INDEX_MESSAGE: &str = "hi! welcome to Orama";
async fn index() -> impl IntoResponse {
    INDEX_MESSAGE
}

static HEALTH_MESSAGE: &str = "up";
async fn health() -> impl IntoResponse {
    HEALTH_MESSAGE
}

#[tracing::instrument(name = "run_writer_webserver")]
async fn run_writer_webserver(sender: UnboundedSender<WalOperation>) -> anyhow::Result<()> {
    let ip_addr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
    let addr = SocketAddr::new(ip_addr, 8081);

    let writer_server = Arc::new(WriterServer {
        collections: RwLock::new(HashMap::new()),
        sender,
    });

    let router = write_api_config()
        .with_state(writer_server);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    info!("Started at http://{:?}", listener.local_addr().unwrap());

    let output = axum::serve(listener, router).await;

    match output {
        Ok(_) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum WalTCPServerRequest {
    FollowCollection(CollectionId),
}

#[tracing::instrument(name = "run_writer_wal_server")]
async fn run_writer_wal_server(mut receiver: UnboundedReceiver<WalOperation>) -> anyhow::Result<()> {
    let ip_addr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
    let addr = SocketAddr::new(ip_addr, 2012);

    let listener = TcpListener::bind(&addr).await?;
    println!("Listening on: {}", addr);

    let followers = Arc::new(RwLock::new(HashMap::<CollectionId, Vec<UnboundedSender<WalOperation>>>::new()));

    let r_followers = followers.clone();
    tokio::spawn(async move {
        while let Some(op) = receiver.recv().await {
            info!("Received: {:?}", op);
            match op.clone() {
                WalOperation::CollectionCreated(collection_id) => {
                    let followers = r_followers.clone();
                    info!("Collection created: {:?}", collection_id);
                    let mut followers = followers.write().await;
                    info!("Checking followers: {:?}", followers.len());
                    let followers_vec = match followers.get_mut(&collection_id) {
                        None => {
                            warn!("No followers for collection: {:?}", collection_id);
                            continue;
                        },
                        Some(followers) => followers,
                    };

                    info!("Sending to followers: {:?}", followers_vec.len());
                    for follower in followers_vec.iter_mut() {
                        follower.send(op.clone()).expect("Failed to send to follower");
                    }
                }
                
            }
        }
    });

    loop {
        let (mut socket, _) = listener.accept().await?;

        let followers = followers.clone();
        tokio::spawn(async move {
            let (socket_reader, socket_writer) = socket.split();

            let ld = FramedRead::new(socket_reader, LengthDelimitedCodec::new());
            let bincode = SymmetricalBincode::<WalTCPServerRequest>::default();
            let mut deserialized = SymmetricallyFramed::new(ld, bincode);
            
            let message = deserialized.next().await;

            let op = match message {
                None => {
                    // closed connection
                    return;
                }
                Some(Err(e)) => {
                    eprintln!("failed to read from socket; error = {:?}", e);
                    return;
                }
                Some(Ok(op)) => op,
            };

            match op {
                WalTCPServerRequest::FollowCollection(collection_id) => {

                    let (sender, mut rec) = unbounded_channel::<WalOperation>();

                    info!("Following collection: {:?}", collection_id);
                    let mut collection_followers = followers.write().await;
                    collection_followers.entry(collection_id)
                        .or_default()
                        .push(sender);
                    drop(collection_followers);

                    let rd = FramedWrite::new(socket_writer, LengthDelimitedCodec::new());
                    let bincode = SymmetricalBincode::<WalOperation>::default();
                    let mut serialized = SymmetricallyFramed::new(rd, bincode);

                    while let Some(op) = rec.recv().await {
                        serialized.send(op).await.expect("Failed to send to socket");
                    }

                },
                x => unimplemented!("Operation not supported: {x:?}"),
            }
        });
    }

    Ok(())
}

#[tracing::instrument(name = "run_reader_webserver")]
async fn run_reader_webserver() -> anyhow::Result<()> {
    let ip_addr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
    let addr = SocketAddr::new(ip_addr, 8080);

    let router = api_config();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    println!("Started at http://{:?}", listener.local_addr().unwrap());

    let output = axum::serve(listener, router).await;

    match output {
        Ok(_) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[tracing::instrument]
async fn run_reader_wal_server() -> anyhow::Result<()> {
    let ip_addr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
    let addr = SocketAddr::new(ip_addr, 2012);

    let mut stream = TcpStream::connect(addr).await;
    while let Err(e) = stream {
        warn!("Error connecting to WAL server: {:?}", e);
        stream = TcpStream::connect(addr).await;
        sleep(Duration::from_secs(1)).await;
    }
    let stream = stream.expect("Connected to WAL server");
    let (reader, writer) = stream.into_split();

    let rd = FramedWrite::new(writer, LengthDelimitedCodec::new());
    let bincode = SymmetricalBincode::<WalTCPServerRequest>::default();
    let mut serialized = SymmetricallyFramed::new(rd, bincode);

    let output = serialized.send(WalTCPServerRequest::FollowCollection(CollectionId("tommaso-example".to_string()))).await;
    match output {
        Ok(_) => println!("Sent"),
        Err(e) => panic!("Error in sending: {:?}", e),
    };

    let ld = FramedRead::new(reader, LengthDelimitedCodec::new());
    let bincode = SymmetricalBincode::<WalTCPServerRequest>::default();
    let mut deserialized = SymmetricallyFramed::new(ld, bincode);
    loop {
        let value = deserialized.next().await;

        info!("Received: {:?}", value);

        match value {
            None => {
                // closed connection
                break;
            }
            Some(Err(e)) => {
                eprintln!("failed to read from socket; error = {:?}", e);
                break;
            }
            Some(Ok(op)) => {
                println!("reader_server: Received: {:?}", op);
            }
        }
    }

    warn!("reader_server: Connection closed");

    Ok(())
}

async fn start() -> anyhow::Result<()> {
    // TODO: understand if `TaskTracker` is necessary
    let tracker = TaskTracker::new();

    let (internal_writer_wal_sender, internal_writer_wal_receiver) = unbounded_channel::<WalOperation>();

    tracker.spawn(run_writer_webserver(internal_writer_wal_sender)
        .instrument(tracing::info_span!("internal_writer_wal_sender")));
    tracker.spawn(run_writer_wal_server(internal_writer_wal_receiver)
        .instrument(tracing::info_span!("internal_writer_wal_receiver")));
    tracker.spawn(run_reader_webserver()
        .instrument(tracing::info_span!("run_reader_webserver")));
    tracker.spawn(run_reader_wal_server()
        .instrument(tracing::info_span!("run_reader_wal_server")));

    tracker.close();

    // Wait for everything to finish.
    tracker.wait().await;

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    warn!("Starting");

    start().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::start;

    #[tokio::test]
    async fn test_main() -> Result<()> {
        start().await
    }
}
