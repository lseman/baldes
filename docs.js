/* Offline-friendly reader: navigation works without JavaScript; search is local. */
(() => {
  const input = document.querySelector('#docs-search');
  const results = document.querySelector('#search-results');
  const status = document.querySelector('#search-status');
  const root = new URL(document.body.dataset.docsRoot, location.href);
  let index;
  let revision = 0;
  const load = async () => {
    if (!index) index = fetch(new URL('search-index.json', root)).then(response => {
      if (!response.ok) throw new Error('Search index unavailable');
      return response.json();
    }).catch(error => { index = null; throw error; });
    return index;
  };
  const search = async () => {
    const query = input.value.trim().toLowerCase();
    const current = ++revision;
    results.replaceChildren();
    if (!query) { results.hidden = true; status.textContent = ''; return; }
    results.hidden = false;
    status.textContent = 'Searching guides.';
    try {
      const entries = await load();
      if (current !== revision) return;
      const terms = query.split(/\s+/);
      const matches = entries.map(entry => {
        const title = `${entry.title} ${entry.heading}`.toLowerCase();
        const text = `${title} ${entry.text}`.toLowerCase();
        return {entry, score: terms.every(term => text.includes(term)) ? 1 + terms.filter(term => title.includes(term)).length * 3 : 0};
      }).filter(match => match.score).sort((a,b) => b.score - a.score).slice(0,8);
      const heading = document.createElement('h2');
      heading.textContent = matches.length ? 'Matching sections' : 'No matching sections';
      results.append(heading);
      const list = document.createElement('ul');
      matches.forEach(({entry}) => {
        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = new URL(entry.url, root).href;
        const page = document.createElement('small');
        page.textContent = entry.title;
        const title = document.createElement('strong');
        title.textContent = entry.heading;
        const excerpt = document.createElement('span');
        excerpt.className = 'search-excerpt';
        const at = Math.max(0, entry.text.toLowerCase().indexOf(terms[0]) - 35);
        excerpt.textContent = `${at ? '…' : ''}${entry.text.slice(at, at + 145)}${entry.text.length > at + 145 ? '…' : ''}`;
        link.append(page, title, excerpt);
        item.append(link);
        list.append(item);
      });
      results.append(list);
      if (!matches.length) {
        const hint = document.createElement('p');
        hint.textContent = 'Try a feature name such as branching, knapsack, or HiGHS.';
        results.append(hint);
      }
      status.textContent = matches.length ? `${matches.length} matching sections shown.` : 'No matching sections.';
    } catch {
      if (current !== revision) return;
      const message = document.createElement('p');
      message.textContent = 'Search is unavailable. Use the guide navigation to keep reading.';
      results.append(message);
      status.textContent = 'Search is unavailable.';
    }
  };
  input.addEventListener('input', search);
  input.form.addEventListener('submit', event => { event.preventDefault(); search(); });
  input.addEventListener('keydown', event => {
    if (event.key === 'Escape') { input.value = ''; search(); }
  });
  document.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault(); input.focus(); input.select();
    }
  });
  const initial = new URLSearchParams(location.search).get('q');
  if (initial) { input.value = initial; search(); }
})();
