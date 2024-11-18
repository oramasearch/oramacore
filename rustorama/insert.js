import { readLines } from "https://deno.land/std@0.208.0/io/mod.ts";

const INPUT_FILE = "./reviews.json";
const BATCH_SIZE = 2_000;
const API_URL = "http://127.0.0.1:8080/v0/collections/xyz/documents";

async function sendBatch(batch, currentCount, totalCount) {
    try {
        const response = await fetch(API_URL, {
            method: "PATCH",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(batch),
        });

        if (response.ok) {
            console.log(`Batch upload successful: ${currentCount}/${totalCount} uploaded`);
        } else {
            console.error(`Batch upload failed: HTTP ${response.status}`);
        }
    } catch (error) {
        console.error("Error during batch upload:", error);
    }
}

async function processJsonlFile() {
    const fileReader = await Deno.open(INPUT_FILE);
    let batch = [];
    let counter = 0;
    let totalLines = 0;

    for await (const _ of readLines(await Deno.open(INPUT_FILE))) {
        totalLines++;
    }

    for await (const line of readLines(fileReader)) {
        const json = JSON.parse(line.trim());

        batch.push({
            id: counter.toString(),
            reviewerName: json.reviewerName,
            reviewText: json.reviewText,
            summary: json.summary,
        });
        counter++;

        if (batch.length === BATCH_SIZE) {
            await sendBatch(batch, counter, totalLines);
            batch = [];
        }
    }

    if (batch.length > 0) {
        await sendBatch(batch, counter, totalLines);
    }

    fileReader.close();
    console.log("All data sent.");
}

await processJsonlFile();
