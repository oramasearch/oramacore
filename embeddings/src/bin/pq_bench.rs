use anyhow::Result;
use csv::ReaderBuilder;
use embeddings::pq;
use embeddings::OramaModels;
use fastembed::Embedding;
use serde::Deserialize;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct Record {
    artist: String,
    song: String,
    link: String,
    text: String,
}

const BATCH_SIZE: usize = 50;
const MAX_RECORDS: usize = 10_000;

fn main() -> Result<()> {
    let model = OramaModels::MultilingualE5Small.try_new()?;
    let mut data: Vec<Embedding> = vec![];

    let file_path = "src/bin/datasets/spotify_millsongdata.csv";
    let mut reader = ReaderBuilder::new().flexible(true).from_path(file_path)?;

    let total_records = reader.records().count();

    reader = ReaderBuilder::new().flexible(true).from_path(file_path)?;

    let mut batch_texts: Vec<String> = Vec::with_capacity(BATCH_SIZE);
    let mut current_record = 0;

    for result in reader.deserialize() {
        if current_record >= MAX_RECORDS {
            break;
        }

        let record: Record = result?;
        let concat_text = format!("{} - {} - {}", record.artist, record.song, record.text);
        batch_texts.push(concat_text);

        current_record += 1;

        if batch_texts.len() == BATCH_SIZE
            || current_record == total_records
            || current_record == MAX_RECORDS
        {
            let batch_embeddings = model.embed(batch_texts, Some(BATCH_SIZE))?;
            data.extend(batch_embeddings.into_iter());

            let percentage = (current_record as f64 / MAX_RECORDS as f64) * 100.0;
            println!(
                "generated embeddings up to record {} of {} ({:.2}%)",
                current_record, MAX_RECORDS, percentage
            );

            batch_texts = Vec::with_capacity(BATCH_SIZE);
        }
    }

    let start_time = Instant::now();
    let quantizer = pq::ProductQuantizer::try_new(data)?;
    let duration = start_time.elapsed();

    println!("Time taken to train ProductQuantizer: {:.2?}", duration);

    let new_embedding = model.embed(vec![CUSTOM_SONG_LYRICS.to_string()], Some(1))?;

    let quantization_time = Instant::now();
    let quantized_embeddings = quantizer.quantize(new_embedding);
    let quantization_duration = quantization_time.elapsed();

    println!(
        "Time taken to quantize a new embedding: {:.2?}",
        quantization_duration
    );
    println!("Quantized embedding:");
    dbg!(quantized_embeddings);

    Ok(())
}

const CUSTOM_SONG_LYRICS: &str = r"
Well, I walk upon the river like it's easier than land
Evil's in my pocket and your will is in my hand
Oh, your will is in my hand
And I'll throw it in the current that I stand upon so still
Love is all, from what I've heard, but my heart's learned to kill
Oh, mine has learned to kill
Oh, I said I could rise
From the harness of our goals
Here come the tears
But like always, I let them go
Just let them go
And now spikes will keep on falling from the heavens to the floor
The future was our skin and now we don't dream anymore
No, we don't dream anymore
Like a house made from spider webs and the clouds rolling in
I bet this mighty river's both my savior and my sin
Oh, my savior and my sin
Oh, I said I could rise
From the harness of our goals
Here come the tears
But like always, I let them go
Just let them go
Well, I walk upon the river like it's easier than land
Evil's in my pocket and your strength is in my hand
Strength is in my hand
And I'll throw you in the current that I stand upon so still
Love is all, from what I've heard, but my heart's learned to kill
Oh, mine has learned to kill
Oh, I said I could rise
From the harness of our goals
Here come the tears
But like always, I let them go
Just let them go
";
