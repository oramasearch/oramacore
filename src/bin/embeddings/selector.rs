use rustorama::embeddings::properties_selector::PropertiesSelector;
use tokio::time::Instant;

fn main() -> anyhow::Result<()> {
    let map_reduce_code = r#"
        function selectProperties(document) {
            return Object
                .keys(document)
                .map(prop => {
                    if (typeof document[prop] === 'string') return prop
                    else return null
                })
                .filter(Boolean)
        }
    "#
    .to_string();

    let document = r#"
      {
        "id": "10000722",
        "category": "Training & Gym,Dance",
        "sportTags": [ "Training & Gym", "Dance"],
        "genders": [ "WOMEN" ],
        "fullTitle": "Nike Everyday Lightweight Women's Training Footie Socks (3 Pairs)",
        "description": "COMFORTABLE FIT WITH VENTILATION.The Nike Everyday Lightweight Footie Socks are made with mesh and a stretchy cotton blend for breathability and a snug, soft feel.BenefitsSweat-wicking fabric helps you stay dry and comfortable.Mesh panels help keep your feet cool.Reinforced toes add durability.Product Details71% cotton/26% nylon/2-3% spandex/0-1% metalized fiberMaterial percentages may vary. Check label for actual content.Machine washImportedShown: White/BlackStyle: SX4863-101",
        "fullPrice": 14,
        "title": "Nike Everyday Lightweight",
        "review_titles": [ "Best No-show socks", "Needs improvement", "Great little find"]
      }
    "#.to_string();

    let evaluator = PropertiesSelector::try_new()?;
    let eval_start = Instant::now();
    let result = evaluator.eval(map_reduce_code, document)?;
    let elapsed = Instant::now().duration_since(eval_start);

    println!("Evaluated in {} microseconds", elapsed.as_micros());

    dbg!(result);

    Ok(())
}
