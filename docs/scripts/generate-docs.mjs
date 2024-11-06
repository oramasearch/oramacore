import { generateFiles } from 'fumadocs-openapi';
import { rimrafSync } from 'rimraf';

rimrafSync('./content/docs/rest-api')

void generateFiles({
    input: ['./orama-api.yaml'],
    output: './content/docs/rest-api',
});