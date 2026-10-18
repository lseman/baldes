/* Progressive enhancement: the page remains readable without JavaScript. */
(() => {
  const capacity = document.querySelector('#capacity');
  if (capacity) {
    const customers = [[180,85],[310,70],[430,90],[530,145],[545,260],[460,325],[335,345],[220,320],[125,245],[110,150]];
    const depot = [330,210];
    const update = () => {
      const size = Number(capacity.value);
      const paths = [];
      for (let i = 0; i < customers.length; i += size) {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        path.setAttribute('points', [depot, ...customers.slice(i, i + size), depot].map(p => p.join(',')).join(' '));
        path.setAttribute('class', `route-${paths.length}`);
        paths.push(path);
      }
      document.querySelector('#route-lines').replaceChildren(...paths);
      document.querySelector('#capacity-value').textContent = size;
      document.querySelector('#route-summary').textContent = `${paths.length} routes · 10 customers · unit demand`;
      document.querySelector('#map-description').textContent = `${paths.length} illustrative routes for a vehicle capacity of ${size}. Each customer has unit demand. Routes are grouped, not optimized by BALDES.`;
    };
    capacity.addEventListener('input', update);
    update();
  }
  const status = document.createElement('p');
  status.className = 'sr-only';
  status.setAttribute('role', 'status');
  document.body.append(status);
  document.querySelectorAll('pre').forEach(pre => {
    const code = pre.querySelector('code');
    if (!code) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-button';
    button.textContent = 'Copy';
    button.setAttribute('aria-label', 'Copy code');
    let reset;
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = 'Copied';
        status.textContent = 'Code copied to clipboard.';
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(code);
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'Code selected. Press Control+C or Command+C to copy.';
      }
      clearTimeout(reset);
      reset = setTimeout(() => { button.textContent = 'Copy'; }, 1800);
    });
    pre.append(button);
  });
})();
