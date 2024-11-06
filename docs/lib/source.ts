import { docs, meta } from '@/.source';
import { createMDXSource } from 'fumadocs-mdx';
import { createOpenAPI } from 'fumadocs-openapi/server';
import { loader } from 'fumadocs-core/source';
import { attachFile } from 'fumadocs-openapi/server';

export const source = loader({
  baseUrl: '/docs',
  source: createMDXSource(docs, meta),
  pageTree: {
    attachFile
  }
});

export const openapi = createOpenAPI({});
