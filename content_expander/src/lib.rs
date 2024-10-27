mod prompts;
mod vision;

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

    let results = vision::describe_images(example_json.to_string()).await?;
    dbg!(results);

    Ok(())
}
