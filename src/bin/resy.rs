#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize S3 client
    let s3 = S3::new(
        "my-bucket".to_string(),
        "AKIA...".to_string(),   // access_key_id
        "secret...".to_string(), // secret_access_key
        "us-east-1".to_string(), // region
    );

    // Path to store state database
    let db_path = "./bucket_state.db";

    // Track changes and handle each one
    let stats = s3
        .stream_diff_and_update(db_path, |change| {
            match change {
                Change::Added(obj) => {
                    println!("➕ Added: {} ({} bytes)", obj.key, obj.size);
                }
                Change::Modified { old, new } => {
                    println!(
                        "🔄 Modified: {} ({} → {} bytes)",
                        new.key, old.size, new.size
                    );
                }
                Change::Deleted(obj) => {
                    println!("❌ Deleted: {} ({} bytes)", obj.key, obj.size);
                }
            }
            Ok(())
        })
        .await?;

    // Print summary
    println!("\nSummary:");
    println!("  Added: {}", stats.added);
    println!("  Modified: {}", stats.modified);
    println!("  Deleted: {}", stats.deleted);
    println!("  Unchanged: {}", stats.unchanged);

    Ok(())
}
