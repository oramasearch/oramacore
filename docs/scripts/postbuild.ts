import type { GeneralPurposeCrawlerResult } from '@orama/crawly'
import path from 'path';
import { fileURLToPath } from 'url';
import { readFileSync } from 'node:fs'
import { globby } from 'globby'
import { generalPurposeCrawler } from '@orama/crawly'
import { CloudManager } from '@oramacloud/client'
import 'dotenv/config'

const baseURL = process.env.DOCS_BASE_URL || "http://localhost:3000";
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const buildDir = path.join(__dirname, "../.next/server/app");
const docsDir = path.join(buildDir, "/docs");


const isGitHubAction = Boolean(process.env.GITHUB_ACTIONS)

if (isGitHubAction) {
  console.log("🚫 Skipping postbuild script in GitHub Action.");
  process.exit(0)
}

const unslugify = (slug: string) => slug
  .toLowerCase()
  .split(/[-_.\s]/)
  .map((w) => `${w.charAt(0).toUpperCase()}${w.substring(1)}`)
  .join(' ');

const getAllFiles = async () => {
  const [docs] = await Promise.all([
    globby(`${docsDir}/**/*.html`)
  ]);

  docs.forEach((file) => {
    console.log(`📄 File: ${file.replace(buildDir, '')}`);
    console.log("------------------------------------------");
  });

  return { docs };
}

const generateSearchDoc = (path: string) => {
  const content = readFileSync(path, 'utf-8');

  const localPath = path
    .replace(buildDir, '')
    .replace('.html', '');

  const fullPath = `${baseURL}${localPath}`;

  return generalPurposeCrawler(fullPath, content, {
    parseCodeBlocks: true
  }).map((doc) => ({
    // Overwrite the object for doc indexing
    ...doc,
    path: fullPath
  }));
}

const getAllParsedDocs = async () => {
  const { docs } = await getAllFiles();
  const parsedDocs = (await Promise.all(docs.map(generateSearchDoc)))
    .flat()
    .map((doc) => ({
      ...doc,
      category: 'OramaCore',
      section: unslugify(doc.section)
    }))

  return [...parsedDocs];
}

const updateOramaCloud = async (docs: GeneralPurposeCrawlerResult[]) => {
  if (process.env.ORAMA_CLOUD_INDEX_ID === undefined || process.env.ORAMA_CLOUD_PRIVATE_API_KEY === undefined) {
    console.warn("\n🚫 ORAMA_CLOUD_INDEX_ID and ORAMA_CLOUD_PRIVATE_API_KEY are not set, skipping Orama Cloud index update.\n")
    return
  }

  const oramaCloudManager = new CloudManager({
    api_key: process.env.ORAMA_CLOUD_PRIVATE_API_KEY
  });

  const index = oramaCloudManager.index(process.env.ORAMA_CLOUD_INDEX_ID);

  console.log(`🐶 Updating Orama Cloud with ${docs.length} documents...`);
  await index.snapshot(docs);

  console.log('🚀 Deploying Orama Cloud...');
  await index.deploy();
};


/**
 * Main function: 
 * - List all .html files in the build directory
 * - Generate search documents for each .html file
 * - Parse the content of each .html file
 * - Index the parsed documents
 * - Deploy the index to Orama Cloud
 */
(async () => {
  console.log("📁 Build directory: \t", buildDir)
  console.log("📁 Docs directory: \t", docsDir);

  console.log("\n🔍 Listing all .html build files and content...\n");

  const documents = await getAllParsedDocs();

  console.log(`✨ Generated ${documents.length} search documents.`);

  await updateOramaCloud(documents);

  console.log("✅ Done!");
})();
