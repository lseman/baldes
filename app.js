/* Progressive enhancement: the page remains readable without JavaScript. */
(() => {
  const capacity = document.querySelector('#capacity');
  if (capacity) {
    const customers = [[180,85],[310,70],[430,90],[530,145],[545,260],[460,325],[335,345],[220,320],[125,245],[110,150]];
    const depot = [330,210];
    const controls = document.querySelector('.route-controls');
    const demo = document.createElement('button');
    demo.type = 'button';
    demo.className = 'route-demo';
    controls.append(demo);
    const summary = document.querySelector('#route-summary');
    const nodes = [...document.querySelectorAll('#customer-nodes > g')];
    let playing = true;
    let visible = true;
    let timer;
    let animations = [];
    const syncDemo = () => {
      demo.textContent = playing ? 'Pause demo Ⅱ' : 'Play demo ▷';
      demo.setAttribute('aria-label', playing ? 'Pause routing demo' : 'Play routing demo');
      // Automatic rounds should not repeatedly announce changes.
      summary.setAttribute('aria-live', playing ? 'off' : 'polite');
    };
    const update = () => {
      clearTimeout(timer);
      animations.forEach(animation => animation.cancel());
      animations = [];
      const size = Number(capacity.value);
      const paths = [];
      let delay = 0;
      for (let i = 0; i < customers.length; i += size) {
        const points = [depot, ...customers.slice(i, i + size), depot];
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        path.setAttribute('points', points.map(p => p.join(',')).join(' '));
        path.setAttribute('class', `route-${paths.length}`);
        paths.push(path);
      }
      document.querySelector('#route-lines').replaceChildren(...paths);
      document.querySelector('#route-vehicles')?.remove();
      const vehicles = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      vehicles.id = 'route-vehicles';
      vehicles.setAttribute('aria-hidden', 'true');
      document.querySelector('#route-map').append(vehicles);
      paths.forEach((path, route) => {
        const length = path.getTotalLength();
        const duration = length * 3;
        const vehicle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        vehicle.setAttribute('r', '7');
        vehicle.setAttribute('class', `route-${route}`);
        vehicle.setAttribute('fill', 'var(--surface)');
        vehicle.setAttribute('stroke-width', '3');
        vehicle.style.opacity = '0';
        vehicles.append(vehicle);
        const startsAt = performance.now() + delay;
        let frame;
        const move = now => {
          const progress = (now - startsAt) / duration;
          if (progress >= 1) {
            vehicle.style.opacity = '0';
            return;
          }
          if (progress >= 0) {
            const point = path.getPointAtLength(length * progress);
            vehicle.setAttribute('cx', point.x);
            vehicle.setAttribute('cy', point.y);
            vehicle.style.opacity = '1';
          }
          frame = requestAnimationFrame(move);
        };
        frame = requestAnimationFrame(move);
        const stop = () => {
          cancelAnimationFrame(frame);
          vehicle.style.opacity = '0';
        };
        animations.push({ cancel:stop, finish:stop });
        animations.push(path.animate([
          { strokeDasharray:`${length} ${length}`, strokeDashoffset:length },
          { strokeDasharray:`${length} ${length}`, strokeDashoffset:0 }
        ], { duration, delay, fill:'backwards', easing:'linear' }));
        let distance = 0;
        let previous = depot;
        customers.slice(route * size, (route + 1) * size).forEach((point, index) => {
          distance += Math.hypot(point[0] - previous[0], point[1] - previous[1]);
          previous = point;
          animations.push(nodes[route * size + index].animate([
            { opacity:.3 }, { opacity:1 }
          ], { duration:250, delay:delay + distance * 3, fill:'backwards' }));
        });
        delay += duration + 180;
      });
      document.querySelector('#capacity-value').textContent = size;
      summary.textContent = `${paths.length} routes · 10 customers · unit demand`;
      document.querySelector('#map-description').textContent = `${paths.length} illustrative routes for a vehicle capacity of ${size}. Each customer has unit demand. Routes are grouped, not optimized by BALDES.`;
      if (playing && visible && !document.hidden) {
        timer = setTimeout(() => {
          capacity.value = size >= 6 ? 2 : size + 1;
          update();
        }, delay + 900);
      }
    };
    capacity.addEventListener('input', () => {
      playing = false;
      syncDemo();
      update();
    });
    demo.addEventListener('click', () => {
      playing = !playing;
      syncDemo();
      if (playing) update();
      else {
        clearTimeout(timer);
        animations.forEach(animation => animation.finish());
      }
    });
    const syncVisibility = () => {
      clearTimeout(timer);
      if (visible && !document.hidden && playing) update();
      else animations.forEach(animation => animation.finish());
    };
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver(([entry]) => {
        const next = entry.isIntersecting;
        if (next === visible) return;
        visible = next;
        syncVisibility();
      });
      observer.observe(document.querySelector('.route-card'));
    }
    document.addEventListener('visibilitychange', syncVisibility);
    syncDemo();
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
