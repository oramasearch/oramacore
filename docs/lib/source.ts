import type { InferPageType } from 'fumadocs-core/source'

import { createElement } from 'react'
import { createMDXSource } from 'fumadocs-mdx'
import { loader } from 'fumadocs-core/source'
import { icons } from 'lucide-react'
import { docs, meta } from '@/.source'

export type Page = InferPageType<typeof source>

export const source = loader({
  baseUrl: '/docs',
  source: createMDXSource(docs, meta),
  icon(icon) {
    if (icon && icon in icons) {
      return createElement(icons[icon as keyof typeof icons])
    }
  }
})
