/* Page shell: header, sidebar, outline, search, progress, theme. Runs on every page. */
(function () {
  const H = window.HUB, $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const page = document.body.dataset.topic || 'home';
  const store = {
    get(k, d) { try { return JSON.parse(localStorage.getItem('aihub:' + k)) ?? d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('aihub:' + k, JSON.stringify(v)); } catch (e) { /* storage blocked: progress just won't persist */ } },
  };

  // theme
  const setTheme = (t) => { document.documentElement.dataset.theme = t; store.set('theme', t); window.dispatchEvent(new Event('themechange')); };
  document.documentElement.dataset.theme = store.get('theme', 'dark');

  // progress
  const progress = store.get('progress', {});
  const live = H.topics.filter((t) => t.status === 'live');
  const pct = (t) => { const s = t.sections; const d = s.filter((x) => progress[t.id + ':' + x.id]).length; return Math.round((100 * d) / s.length); };
  const total = () => { const all = live.flatMap((t) => t.sections.map((s) => t.id + ':' + s.id)); return Math.round((100 * all.filter((k) => progress[k]).length) / all.length); };

  // header
  $('#hdr').innerHTML = `
    <button class="btn" id="menu-btn" aria-label="Menu">☰</button>
    <a class="brand" href="index.html">
      <svg viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#238636"/><circle cx="9" cy="10" r="2.5" fill="#fff"/><circle cx="9" cy="22" r="2.5" fill="#fff"/><circle cx="16" cy="16" r="2.5" fill="#fff"/><circle cx="23" cy="16" r="2.5" fill="#fff"/><path d="M9 10L16 16L9 22M16 16H23" stroke="#fff" stroke-width="1.4" fill="none"/></svg>
      <span><b>AI Learning Hub</b><small>Learn · Build · Explore</small></span></a>
    <div class="search"><input id="q" placeholder="Search topics, concepts, equations…" autocomplete="off"><kbd>Ctrl K</kbd><div class="results" id="res"></div></div>
    <div class="hdr-actions">
      <a class="btn hide-sm" href="${H.repo}" target="_blank" rel="noopener">GitHub</a>
      <span class="btn" id="prog" title="Overall progress"></span>
      <button class="btn" id="theme" aria-label="Toggle theme"></button>
    </div>`;
  const themeBtn = $('#theme'), icon = () => (themeBtn.textContent = document.documentElement.dataset.theme === 'dark' ? '🌙' : '☀️');
  icon(); themeBtn.onclick = () => { setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); icon(); };

  // sidebar
  const side = $('#side');
  const renderSide = () => {
    side.innerHTML = `<div class="side-title">AI learning paths</div>` + H.topics.map((t) => {
      const on = t.id === page, p = t.status === 'live' ? pct(t) : 0;
      const a = t.status === 'live'
        ? `<a class="topic ${on ? 'active' : ''}" href="${t.href}">` : `<div class="topic soon">`;
      return `${a}<span class="ico" style="background:${t.color}26;color:${t.color}">${t.icon}</span><span class="nm">${t.title}</span>${t.status === 'live' ? `<span class="pct">${p}%</span>` : '<span class="tag">soon</span>'}${t.status === 'live' ? '</a>' : '</div>'}`
        + (on ? `<div class="subnav">${t.sections.map((s) => `<a href="#${s.id}">${s.t}</a>`).join('')}</div>` : '');
    }).join('') + `<div class="journey"><b>Your learning journey</b><div class="bar"><i style="width:${total()}%"></i></div><span style="color:var(--muted)">${total()}% of live topics complete</span></div>`;
    $('#prog').textContent = `Progress ${total()}%`;
  };
  renderSide();
  $('#menu-btn').onclick = () => side.classList.toggle('open');
  side.addEventListener('click', (e) => e.target.closest('a') && side.classList.remove('open'));

  // lessons: mark complete
  $$('section.lesson[id]').forEach((sec) => {
    const key = page + ':' + sec.id, btn = $('.mark', sec);
    if (!btn) return;
    const paint = () => { btn.textContent = progress[key] ? '✓ Completed' : 'Mark complete'; btn.classList.toggle('on', !!progress[key]); };
    paint();
    btn.onclick = () => { progress[key] = !progress[key]; store.set('progress', progress); paint(); renderSide(); updateHero(); };
  });
  const updateHero = () => {
    const t = H.topics.find((x) => x.id === page); if (!t || !t.sections) return;
    const el = $('#hero-pct'); if (el) { el.textContent = pct(t) + '% complete'; $('#hero-bar').style.width = pct(t) + '%'; }
  };
  updateHero();

  // outline (right)
  const out = $('#outline');
  if (out) {
    const hs = $$('main section.lesson[id] h2, main h3[id]'); hs.forEach((h) => { h.dataset.t = h.id || h.closest('section').id; });
    const t = H.topics.find((x) => x.id === page), nx = t && H.topics.find((x) => x.id === t.next);
    out.innerHTML = `<h4>On this page</h4>` + hs.map((h) => `<a class="${h.tagName === 'H3' ? 'l3' : ''}" href="#${h.dataset.t}">${h.textContent}</a>`).join('')
      + (nx ? `<div class="next"><small>Next topic</small><br><a href="${nx.href}"><b>${nx.title}</b></a><br><small>${nx.desc}</small></div>` : '');
    const links = new Map($$('a', out).map((a) => [a.getAttribute('href').slice(1), a]));
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { links.forEach((a) => a.classList.remove('act')); links.get(e.target.dataset.t)?.classList.add('act'); } }), { rootMargin: '-80px 0px -70% 0px' });
    hs.forEach((h) => io.observe(h));
  }

  // search
  const idx = [];
  live.forEach((t) => { idx.push({ t: t.title, s: 'Topic', h: t.href, k: t.desc + ' ' + (t.tags || []).join(' ') }); t.sections.forEach((s) => idx.push({ t: s.t, s: t.title, h: t.href + '#' + s.id, k: s.k })); });
  H.topics.filter((t) => t.status !== 'live').forEach((t) => idx.push({ t: t.title, s: 'Coming soon', h: 'index.html#topics', k: t.desc }));
  const q = $('#q'), res = $('#res');
  q.addEventListener('input', () => {
    const w = q.value.toLowerCase().split(/\s+/).filter(Boolean);
    const hits = w.length ? idx.filter((i) => w.every((x) => (i.t + ' ' + i.s + ' ' + i.k).toLowerCase().includes(x))).slice(0, 8) : [];
    res.innerHTML = hits.map((i) => `<a href="${i.h}">${i.t} <small>· ${i.s}</small></a>`).join('') || (w.length ? '<a>No matches yet</a>' : '');
    res.classList.toggle('open', !!w.length);
  });
  q.addEventListener('keydown', (e) => { if (e.key === 'Enter') $('a[href]', res)?.click(); if (e.key === 'Escape') { q.blur(); res.classList.remove('open'); } });
  document.addEventListener('click', (e) => { if (!e.target.closest('.search')) res.classList.remove('open'); });
  document.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); q.focus(); q.select(); } });

  // topic cards on home
  const grid = $('#topic-grid');
  if (grid) grid.innerHTML = H.topics.map((t) => {
    const inner = `<span class="ico" style="background:${t.color}26;color:${t.color}">${t.icon}</span><h3>${t.title}</h3><p>${t.desc}</p><div style="margin-top:10px" class="chips">${t.status === 'live' ? (t.tags || []).slice(0, 3).map((x) => `<span class="chip">${x}</span>`).join('') + `<span class="badge live">live · ${pct(t)}%</span>` : '<span class="badge">coming soon</span>'}</div>`;
    return t.status === 'live' ? `<a class="card" href="${t.href}">${inner}</a>` : `<div class="card soon">${inner}</div>`;
  }).join('');

  // math + labs
  const ready = () => {
    $$('.eq,.tex').forEach((el) => { if (window.katex) { try { window.katex.render(el.textContent, el, { displayMode: el.classList.contains('eq'), throwOnError: false }); } catch (e) { /* leave raw TeX */ } } });
    $$('[data-lab]').forEach((el) => { try { AI.labs[el.dataset.lab](el); } catch (e) { el.textContent = 'Lab failed to load: ' + e.message; console.error(e); } });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready); else ready();
})();
