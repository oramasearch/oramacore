import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

function Logo() {
  return (
    <div className='flex items-center text-base'>
      <img src={`/logo/orama-logo.svg`} className='h-6 mr-2' alt='OramaSearch Inc. Logo' />
      OramaCore
    </div>
  )
}

/**
 * Shared layout configurations
 *
 * you can configure layouts individually from:
 * Home Layout: app/(home)/layout.tsx
 * Docs Layout: app/docs/layout.tsx
 */
export const baseOptions: BaseLayoutProps = {
  nav: {
    transparentMode: "always",
    title: (
      <Logo />
    )
  },
  links: [
    {
      text: 'Docs',
      url: '/docs',
      active: 'nested-url',
    }
  ],
};
