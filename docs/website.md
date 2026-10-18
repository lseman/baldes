# Website and documentation publishing

[Documentation index](README.md) · [Build and validation](testing.md)

The website consists of the handwritten landing page in `landing/` and HTML
guides compiled from `docs/**/*.md`. Markdown files are the documentation source;
do not edit generated `landing/docs/` files.

## Compile and preview

The website build needs Python 3.10 or newer. It does not build the Rust solver.
Create a separate environment and install the pinned documentation dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r .github/scripts/docs-requirements.txt
.venv/bin/python .github/scripts/compile_docs.py
.venv/bin/python -m http.server 8000 --directory landing
```

Open `http://localhost:8000/` for the landing page and
`http://localhost:8000/docs/` for the guides. Use an HTTP server rather than
opening HTML files directly, so search can load its local index.

On Windows, use `.venv/Scripts/python.exe` instead of the `bin/python`
path. Re-run the compiler after changing Markdown, reader styles, or scripts.
No JavaScript bundler or external search service is required.

## What the compiler does

- Compiles `README.md` to `index.html` and other Markdown files to matching
  `.html` paths, including nested directories.
- Builds sidebar navigation from this documentation index's link order,
  followed by any additional guides.
- Renders tables, fenced code with syntax highlighting, heading anchors, an
  on-page contents list, and previous/next guide links.
- Rewrites local Markdown links to generated HTML, validates local links and
  heading fragments, and maps repository source links to GitHub.
- Copies linked local images and writes a section-level search index.
- Escapes embedded HTML in Markdown. The shared page templates supply the
  site's markup; guide authors use Markdown for content.
- Replaces only marked generated output after rendering succeeds, removing
  stale generated pages. A failed build leaves the previous output intact.

A new guide appears in the reader automatically. Add its link to
[the documentation index](README.md) to choose its navigation position.
`README.md` and `index.md` in the same directory would both produce
`index.html`; the compiler rejects that collision.

## Custom destinations

```sh
.venv/bin/python .github/scripts/compile_docs.py \
  --output /tmp/baldes-docs \
  --base-url https://example.org/baldes/ \
  --repository-url https://github.com/lseman/baldes
```

`--source` defaults to the repository's `docs/` directory. `--output` defaults
to `landing/docs/`. Source and output must be separate directories, neither
inside the other. Local repository links must resolve within the repository.
The output includes its own styles and scripts; the Home link expects the
landing page one level above the documentation root.

Internal links and assets are relative, so the website works both at `/` and
under a project prefix such as `/baldes/`. `--base-url` controls canonical
page URLs; it does not change internal navigation.

## Automated publishing

The [Pages workflow](../.github/workflows/landing.yaml) runs on pushes to
`main` that change the landing page, documentation, compiler, or its tests and
dependencies. It installs Python dependencies, runs compiler tests, compiles
the guides, and publishes `landing/` to the `gh-pages` branch. It also supports
manual dispatch.

Deployment replaces the published file tree so deleted guides do not remain
on the website. Generated output is ignored by Git; commit the Markdown,
compiler, templates/assets, and dependency file instead. Editing files locally
does not publish them until the workflow runs after a push.

## Reader accessibility

The page templates use headings, landmarks, and visible form labels to expose
structure (WCAG 1.3.1). Skip links bypass repeated navigation (2.4.1), and
interactive elements have visible keyboard focus (2.4.7). The capacity control
uses a native labeled range input, while search and copy actions announce
status changes (4.1.2 / 4.1.3). Reduced-motion preferences disable smooth
scrolling and decorative transitions.

## Validate changes

```sh
.venv/bin/python -m unittest discover -s .github/scripts -p 'test_compile_docs.py'
.venv/bin/python .github/scripts/compile_docs.py
```

The tests cover nested navigation, local/source/image links, duplicate anchors,
tables, code escaping/highlighting, search entries, destination collisions,
output protection, and rebuilds. For visual changes, also inspect a narrow
viewport, keyboard focus, search results, code-copy buttons, and the landing
page's capacity illustration. The illustration groups customers into routes;
it does not execute the Rust solver or claim optimal routes.
