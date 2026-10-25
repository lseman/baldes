"""Behavioral tests for Markdown compilation and generated-site integrity."""
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from compile_docs import build


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.ids = []
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        self.links.extend(attrs[key] for key in ("href", "src") if key in attrs)


class DocumentationBuild(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "docs"
        self.source.mkdir()
        self.output = self.root / "site" / "docs"
        (self.root / "src").mkdir()
        (self.root / "src" / "lib.rs").write_text("// example\n")
        (self.source / "nested").mkdir()
        (self.source / "README.md").write_text(
            "# Overview\n\n[Child](nested/README.md#intro)\n\n"
            "[Source](../src/lib.rs#L1)\n\n"
            "| Name | Value |\n| --- | --- |\n| x | 1 |\n\n"
            "```python\nprint('<safe>')\n```\n\n"
            "<script>alert('untrusted')</script>\n", encoding="utf-8")
        (self.source / "nested" / "README.md").write_text(
            "# Child guide\n\n[Back](../README.md)\n\n"
            "## Intro\n\nVehicle capacity and route bounds.\n\n"
            "## Intro\n\nRepeated heading.\n", encoding="utf-8")

    def compile(self):
        return build(self.source, self.output, self.root,
                     base_url="https://example.org/project/")

    def test_nested_navigation_source_links_and_every_local_target(self):
        self.assertEqual(self.compile(), 2)
        index = (self.output / "index.html").read_text()
        child = (self.output / "nested" / "index.html").read_text()
        self.assertIn('href="nested/index.html#intro"', index)
        self.assertIn('href="../index.html"', child)
        self.assertIn("https://github.com/lseman/baldes/blob/main/src/lib.rs#L1", index)
        self.assertIn("https://example.org/project/docs/nested/index.html", child)
        for path in self.output.rglob("*.html"):
            parsed = Links(path.read_text())
            self.assertEqual(len(parsed.ids), len(set(parsed.ids)))
            self.assertIn("main", parsed.tags)
            for url in parsed.links:
                parts = urlsplit(url)
                if parts.scheme or parts.netloc:
                    continue
                target = path.parent / unquote(parts.path) if parts.path else path
                # Home lives outside the generated documentation directory.
                if target.resolve() == self.output.parent.resolve():
                    continue
                self.assertTrue(target.exists(), (path, url))
                if parts.fragment:
                    self.assertIn(unquote(parts.fragment), Links(target.read_text()).ids)

    def test_tables_highlighting_escape_and_duplicate_anchors(self):
        self.compile()
        index = (self.output / "index.html").read_text()
        self.assertIn("<table>", index)
        self.assertIn('<span class="', index)
        self.assertIn("&lt;script&gt;", index)
        self.assertNotIn("<script>alert", index)
        self.assertIn('class="landing-dark docs-body"', index)
        self.assertIn('name="theme-color" content="#0b0e11"', index)
        syntax = (self.output / "_static" / "syntax.css").read_text()
        self.assertIn("#090c0f", syntax)
        child = (self.output / "nested" / "index.html").read_text()
        self.assertIn('id="intro-1"', child)

    def test_code_panels_keep_literal_text_and_normalize_rust_annotations(self):
        with (self.source / "README.md").open("a") as f:
            f.write('\n```rust,no_run\nlet label = "<route>";\n```\n'
                    '\n```unknown-language\n<script>unsafe</script>\n```\n')
        self.compile()
        index = (self.output / "index.html").read_text()
        self.assertIn('class="code-heading"><span>Python</span>', index)
        self.assertIn('class="language-rust"', index)
        self.assertIn('&lt;route&gt;', index)
        self.assertIn('&lt;script&gt;unsafe&lt;/script&gt;', index)
        self.assertNotIn('<script>unsafe</script>', index)

    def test_tables_have_keyboard_accessible_scroll_regions(self):
        self.compile()
        index = (self.output / "index.html").read_text()
        self.assertIn('class="table-scroll" role="region" aria-label="Data table" tabindex="0"', index)
        self.assertIn('</table></div>', index)

    def test_navigation_groups_follow_directories_and_keep_current_page(self):
        (self.source / "guides").mkdir()
        (self.source / "guides" / "install.md").write_text("# Install\n")
        self.compile()
        guide = (self.output / "guides" / "install.html").read_text()
        self.assertIn('class="nav-group-title">Getting started</p>', guide)
        self.assertIn('href="install.html" aria-current="page">Install</a>', guide)
        self.assertIn('href="../nested/index.html"', guide)
        self.assertIn('DOCS <span aria-hidden="true">/</span> Getting started', guide)

    def test_python_handbook_has_dedicated_navigation_and_search(self):
        import json
        handbook = self.source / "pybaldes"
        handbook.mkdir()
        (handbook / "README.md").write_text(
            "# PyBaldes documentation\n\n[Pricing](pricing.md#exactness)\n")
        (handbook / "pricing.md").write_text(
            "# Stateful pricing\n\n## Exactness\n\nConfigured pricing domain.\n")
        self.compile()
        page = (self.output / "pybaldes" / "index.html").read_text()
        self.assertIn('class="nav-group-title">PyBaldes · Python</p>', page)
        self.assertIn('href="pricing.html#exactness"', page)
        self.assertIn('href="index.html" aria-current="page">PyBaldes documentation</a>', page)
        entries = json.loads((self.output / "search-index.json").read_text())
        self.assertTrue(any(e["url"] == "pybaldes/pricing.html#exactness"
                            and "pricing domain" in e["text"] for e in entries))

    def test_bad_link_fails_without_changing_previous_build(self):
        self.compile()
        before = (self.output / "index.html").read_bytes()
        (self.source / "README.md").write_text("# Overview\n[Missing](missing.md)\n")
        with self.assertRaisesRegex(ValueError, "missing or out-of-repository"):
            self.compile()
        self.assertEqual((self.output / "index.html").read_bytes(), before)

    def test_missing_anchor_fails(self):
        (self.source / "README.md").write_text("# Overview\n[Wrong](nested/README.md#missing)\n")
        with self.assertRaisesRegex(ValueError, "missing heading"):
            self.compile()

    def test_rebuild_removes_stale_generated_pages(self):
        extra = self.source / "old.md"
        extra.write_text("# Retired guide\n")
        self.compile()
        self.assertTrue((self.output / "old.html").exists())
        extra.unlink()
        self.compile()
        self.assertFalse((self.output / "old.html").exists())

    def test_refuses_non_generated_output_or_overlapping_source(self):
        self.output.mkdir(parents=True)
        (self.output / "keep.txt").write_text("handwritten\n")
        with self.assertRaisesRegex(ValueError, "non-generated"):
            self.compile()
        self.assertEqual((self.output / "keep.txt").read_text(), "handwritten\n")
        for output in [self.source, self.source / "out", self.root]:
            with self.assertRaisesRegex(ValueError, "not nested"):
                build(self.source, output, self.root)

    def test_colliding_destinations_are_rejected(self):
        (self.source / "index.md").write_text("# Collision\n")
        with self.assertRaisesRegex(ValueError, "Duplicate HTML"):
            self.compile()

    def test_local_images_are_copied_and_external_links_preserved(self):
        (self.root / "diagram.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        with (self.source / "README.md").open("a") as f:
            f.write("\n![Diagram](../diagram.svg)\n\n[External](https://example.org/x?q=1#z)\n")
        self.compile()
        self.assertTrue((self.output / "_assets" / "diagram.svg").exists())
        index = (self.output / "index.html").read_text()
        self.assertIn('src="_assets/diagram.svg"', index)
        self.assertIn('href="https://example.org/x?q=1#z"', index)

    def test_search_index_covers_section_text_and_nested_urls(self):
        import json
        self.compile()
        entries = json.loads((self.output / "search-index.json").read_text())
        self.assertTrue(any(e["url"] == "nested/index.html#intro" and "capacity" in e["text"] for e in entries))
        self.assertTrue(any(e["url"] == "nested/index.html#intro-1" for e in entries))


if __name__ == "__main__":
    unittest.main()
