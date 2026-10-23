# Three.js

Three.js 0.186.1, MIT licensed. Upstream: https://github.com/mrdoob/three.js

The two ES modules are minified copies of `build/three.module.js` and
`build/three.core.js` from the npm package, generated with esbuild 0.28.2:

```sh
esbuild three.module.js three.core.js --minify --format=esm --outdir=OUTPUT
```

Keep the relative module imports and LICENSE together when updating.
Self-hosting keeps the landing page independent of third-party CDNs.
