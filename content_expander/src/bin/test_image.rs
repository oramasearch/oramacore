use content_expander::prompts::Prompts;
use content_expander::vision::describe_images;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let example_text = r"
        Here's a guitar: https://fast-images.static-thomann.de/pics/bdb/_31/317515/18818768_800.jpg,
        And here's a hoodie: https://img01.ztat.net/article/spp-media-p1/276956282ea04324bab10b4f0d3e955a/ec5aaa2f9b2d4653a8a0154a1493c7af.jpg?imwidth=520
        Foo bar baz.
    ";

    let results = describe_images(example_text.to_string(), Prompts::VisionECommerce)
        .await
        .unwrap();
    dbg!(results);

    Ok(())
}
