use anyhow::Result;
use rustorama::embeddings::pq;
use rustorama::embeddings::OramaModel;

#[tokio::main]
async fn main() -> Result<()> {
    let model = OramaModel::JinaV2BaseCode.try_new().await?;

    let vectors = model.embed( vec![
        "CASUAL COMFORT, SPORTY STYLE.Slide into comfort in the lightweight and sporty Nike Benassi JDI Slide. It features the Nike logo on the foot strap, which is lined in super soft fabric. The foam midsole brings that beach feeling to your feet and adds spring to your kicked-back style.Benefits1-piece, synthetic leather strap is lined with super soft, towel-like fabric.The foam midsole doubles as an outsole, adding lightweight cushioning.Flex grooves let you move comfortably.Shown: Black/WhiteStyle: 343880-090".to_string(),
        "CLASSIC SUPPORT AND COMFORT.The Nike Air Monarch IV gives you classic style with real leather and plenty of lightweight Nike Air cushioning to keep you moving in comfort.BenefitsLeather and synthetic leather team up for durability and classic comfort.An Air-Sole unit runs the length of your foot for cushioning, comfort and support.Rubber sole is durable and provides traction".to_string(),
        "STAY TRUE TO YOUR TEAM ALL DAY, EVERY DAY, GAME DAY.\nRep your favorite team and player anytime in the NFL Baltimore Ravens Game Jersey, inspired by what they&apos;re wearing on the field and designed for total comfort.\nTAILORED FIT\nThis jersey features a tailored fit designed for movement.\n\nLIGHT, SOFT FEEL\nScreen-print numbers provide a light and soft feel".to_string(),
        "STAY TRUE TO YOUR TEAM ALL DAY, EVERY DAY, GAME DAY.\nRep your favorite team and player anytime in the NFL Indianapolis Colts Game Jersey, inspired by what they&apos;re wearing on the field and designed for total comfort.\nTAILORED FIT\nThis jersey features a tailored fit designed for movement.\n\nCLEAN COMFORT\nThe no-tag neck label offers clean comfort.\n\nLIGHT, SOFT FEEL\nScreen-print numbers provide a light and soft feel.\n\nAdditional Details\n\n\nStrategic ventilation for breathability\nWoven jock tag at front lower left\nTPU shield at V-neck\n\n\n\nFabric: 100% recycled polyester\nMachine Wash\nImportedShown: Gym BlueStyle: 468955-442".to_string(),
        "A GAME-DAY ESSENTIAL.Featuring comfortable, absorbent fabric, the Nike Swoosh Wristbands stretch with you and keep your hands dry, so you can play your best even when the game heat up.Product DetailsWidth: 3&quot;Sold in pairsSwoosh design embroideryFabric: 72% cotton/12% nylon/11% polyester/4% rubber/1% spandexMachine washImportedShown: White/BlackStyle: NNN04-101".to_string(),
        "MATCH-READY COMFORT FOR YOUR FEET.The Nike Academy Socks are designed to keep you comfortable during play with soft, sweat-wicking fabric with arch support.BenefitsNike Dri-FIT technology moves sweat away from your skin for quicker evaporation, helping you stay dry and comfortable.Reinforced heel and toe add durability in high-wear areas.Snug band wraps around the arch for a supportive feel.Product DetailsLeft/right specific98% nylon/2% spandexMachine washImportedShown: Varsity Royal/WhiteStyle: SX4120-402".to_string()
    ], None)?;

    let new_vector = model.embed( vec![
        "COMFORTABLE COVERAGE FOR YOUR SHINS.Designed to take the impacts of the game, the Nike J Shin Guards are made with a tough composite shell and perforations for ventilated comfort.BenefitsAnatomical left/right construction contours for comfort.Perforations enhance ventilation.EVA foam provides soft cushioning.Product DetailsMaterials: 80% polyethylene/20% EVAImportedShown: Black/WhiteStyle: SP0040-009".to_string()
    ], None)?;

    let quantizer = pq::ProductQuantizer::try_new(vectors)?;
    let quantized = quantizer.quantize(new_vector);

    dbg!(quantized);

    let vector = model.embed(
        vec![r"
            import { create, insert, search } from '@orama/orama'
            
            const db = create({
                schema: {
                    title: 'string',
                    description: 'string'
                }
            })

            insert(db, {
                title: 'foo',
                description: 'bar'
            })
            
            search(db, {
                term: 'foo'
            })

        "
        .to_string()],
        Some(1),
    )?;

    dbg!(vector);

    Ok(())
}
