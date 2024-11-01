use embeddings::product_quantization::*;
use ndarray::Array2;
use anyhow::Result;

fn main() -> Result<()> {
    let models = embeddings::load_models();

    let embeds = models.embed(
        embeddings::OramaModels::GTESmall,
        vec![
            "COMFORTABLE FIT WITH VENTILATION.The Nike Everyday Lightweight Footie Socks are made with mesh and a stretchy cotton blend for breathability and a snug, soft feel.BenefitsSweat-wicking fabric helps you stay dry and comfortable.Mesh panels help keep your feet cool.Reinforced toes add durability.Product Details71% cotton/26% nylon/2-3% spandex/0-1% metalized fiberMaterial percentages may vary. Check label for actual content.Machine washImportedShown: White/BlackStyle: SX4863-101".to_string(),
            "AN ESSENTIAL FOR PRACTICE OR GAME DAY.The Nike Performance Shorts feature a fitted design, while the lined gusset and sweat-wicking technology help keep you dry and comfortable during practice or play.BenefitsDri-FIT technology helps keep you dry and comfortable.3.5&quot; inseam lets you move freely.Lined gusset helps prevent irritation.Elastic waistband stretches for the perfect fit.Product DetailsTight fit for a body-hugging feelBody: 80% polyester/20% spandex. Gusset lining: 100% polyester.Machine washImportedShown: Team Scarlet/Team WhiteStyle: 108720-657".to_string(),
            "CASUAL COMFORT, SPORTY STYLE.Slide into comfort in the lightweight and sporty Nike Benassi JDI Slide. It features the Nike logo on the foot strap, which is lined in super soft fabric. The foam midsole brings that beach feeling to your feet and adds spring to your kicked-back style.Benefits1-piece, synthetic leather strap is lined with super soft, towel-like fabric.The foam midsole doubles as an outsole, adding lightweight cushioning.Flex grooves let you move comfortably.Shown: Black/WhiteStyle: 343880-090".to_string()
        ]
    )?;

    let lembeds = embeds.first().unwrap();

    println!("Embedding length: {}", lembeds.len());

    if lembeds.is_empty() {
        return Err(anyhow::anyhow!("Generated embeddings are empty"));
    }

    let embeddings = Array2::from_shape_vec(
        (lembeds.len(), embeddings::OramaModels::GTESmall.dimensions()),
        lembeds.clone(),
    )?;

    let num_subvectors = DEFAULT_SUBVECTORS;
    let num_centroids = DEFAULT_CENTROIDS;

    let codebooks = train_pq_codebooks(&embeddings, num_subvectors, num_centroids);

    let quantized_embeddings = quantize_embeddings(&embeddings, &codebooks);

    println!("Quantized Embeddings: {:?}", quantized_embeddings);
    println!("Codebooks:");
    for (i, codebook) in codebooks.iter().enumerate() {
        println!("Codebook {}: {:?}", i, codebook);
    }

    Ok(())
}
