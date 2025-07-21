import { build, emptyDir } from 'jsr:@deno/dnt';
import denojson from './deno.json' with { type: 'json' };

const outDir = './.npm';

await emptyDir(outDir);

await build({
  entryPoints: ['./main.ts'],
  outDir,
  shims: {
    deno: 'dev',
  },
  compilerOptions: {
    lib: ['DOM'],
  },
  scriptModule: 'cjs',
  typeCheck: 'both',
  package: {
    name: denojson.name,
    version: denojson.version,
    description: 'JavaScript and TypeScript client for OramaCore',
    license: 'AGPL-3.0',
    author: {
      name: 'Michele Riva',
      email: 'michele@orama.com',
    },
    repository: {
      type: 'git',
      url: 'git+https://github.com/oramasearch/oramacore.git',
    },
    bugs: {
      url: 'https://github.com/oramasearch/oramacore/repo/issues',
    },
  },
  testPattern: '**/*_test.{ts,js}',
  rootTestDir: '.',
  test: false,
  postBuild() {
    Deno.copyFileSync('./LICENSE.md', `${outDir}/LICENSE.md`);
    Deno.copyFileSync('./README.md', `${outDir}/README.md`);
  },
});
