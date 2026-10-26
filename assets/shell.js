/* Page shell: header, sidebar, outline, search, progress, theme. Runs on every page. */
(async function () {
  const H = window.HUB, $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const page = document.body.dataset.topic || 'home';
  const modIdx = document.body.dataset.module;
  // Topics with a module curriculum name their generated data file (topic.curriculum -> window.<ID>_MODULES).
  // Load any that this page did not include, then flatten lessons into sections so progress/search work for every topic.
  const loadScript = (src) => new Promise((ok) => { if (document.querySelector(`script[src="${src}"]`)) return ok(); const s = document.createElement('script'); s.src = src; s.onload = s.onerror = ok; document.head.appendChild(s); });
  await Promise.all(H.topics.filter((t) => t.curriculum).map((t) => loadScript(t.curriculum)));
  H.topics.forEach((t) => {
    const mods = window[t.id.toUpperCase() + '_MODULES'];
    if (mods) { t.modules = mods; t.sections = mods.flatMap((m) => m.lessons.map((l) => ({ ...l, href: m.href }))); }
  });
  const curT = H.topics.find((t) => t.id === page);
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
  const pct = (t) => { const s = t.sections || []; if (!s.length) return 0; const d = s.filter((x) => progress[t.id + ':' + x.id]).length; return Math.round((100 * d) / s.length); };
  const total = () => { const all = live.flatMap((t) => (t.sections || []).map((s) => t.id + ':' + s.id)); return Math.round((100 * all.filter((k) => progress[k]).length) / all.length); };

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
  const subnav = (t) => {
    if (!t.modules) return `<div class="subnav">${(t.sections || []).map((s) => `<a href="#${s.id}">${s.t}</a>`).join('')}</div>`;
    return `<div class="subnav">${t.modules.map((m) => {
      const here = String(m.n) === modIdx, done = m.lessons.filter((l) => progress[t.id + ':' + l.id]).length;
      return `<a class="mod ${here ? 'here' : ''}" href="${m.href}">${m.n}. ${m.title} <small>${done}/${m.lessons.length}</small></a>`
        + (here ? m.lessons.map((l) => `<a class="les" href="#${l.id}">${l.t}</a>`).join('') : '');
    }).join('')}</div>`;
  };
  const renderSide = () => {
    side.innerHTML = `<div class="side-title">AI learning paths</div>` + H.topics.map((t) => {
      const on = t.id === page, p = t.status === 'live' ? pct(t) : 0;
      const a = t.status === 'live'
        ? `<a class="topic ${on ? 'active' : ''}" href="${t.href}">` : `<div class="topic soon">`;
      return `${a}<span class="ico" style="background:${t.color}26;color:${t.color}">${t.icon}</span><span class="nm">${t.title}</span>${t.status === 'live' ? `<span class="pct">${p}%</span>` : '<span class="tag">soon</span>'}${t.status === 'live' ? '</a>' : '</div>'}`
        + (on ? subnav(t) : '') + (t.children ? `<div class="subnav kids">${t.children.map((c) => `<span class="kid">${c.icon} ${c.title}</span>`).join('')}</div>` : '');
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
      + (nx && nx.href ? `<div class="next"><small>Next topic</small><br><a href="${nx.href}"><b>${nx.title}</b></a><br><small>${nx.desc}</small></div>` : '');
    const links = new Map($$('a', out).map((a) => [a.getAttribute('href').slice(1), a]));
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { links.forEach((a) => a.classList.remove('act')); links.get(e.target.dataset.t)?.classList.add('act'); } }), { rootMargin: '-80px 0px -70% 0px' });
    hs.forEach((h) => io.observe(h));
  }

  // previous / next module
  if (modIdx !== undefined && curT && curT.modules) {
    const i = +modIdx, ms = curT.modules, prev = ms[i - 1], next = ms[i + 1];
    const nav = document.createElement('nav'); nav.className = 'modnav';
    nav.innerHTML = (prev ? `<a class="card" href="${prev.href}"><small>← Previous module</small><h3>${prev.n}. ${prev.title}</h3></a>` : '<span></span>')
      + (next ? `<a class="card" href="${next.href}" style="text-align:right"><small>Next module →</small><h3>${next.n}. ${next.title}</h3></a>` : `<a class="card" href="${curT.href}" style="text-align:right"><small>Finished</small><h3>Back to the curriculum</h3></a>`);
    $('main').appendChild(nav);
  }

  // search
  const idx = [];
  live.forEach((t) => { idx.push({ t: t.title, s: 'Topic', h: t.href, k: t.desc + ' ' + (t.tags || []).join(' ') }); (t.sections || []).forEach((s) => idx.push({ t: s.t, s: t.title, h: (s.href || t.href) + '#' + s.id, k: s.k })); });
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

  // curriculum module cards (math hub)
  const ml = $('#mod-list');
  if (ml && curT && curT.modules) ml.innerHTML = curT.modules.map((m) => {
    const done = m.lessons.filter((l) => progress[curT.id + ':' + l.id]).length;
    return `<div class="modcard"><h3><a href="${m.href}">Module ${m.n} · ${m.title}</a></h3><small style="color:var(--muted)">${m.lessons.length} lessons · ${done} completed</small><ol>${m.lessons.map((l) => `<li><a href="${m.href}#${l.id}"><i>${progress[curT.id + ':' + l.id] ? '✓' : '○'}</i>${l.t}</a></li>`).join('')}</ol></div>`;
  }).join('');
  const pb = $('#pathbar');
  if (pb && curT && curT.modules) pb.innerHTML = curT.modules.map((m) => `<a href="${m.href}"><b>${m.n}</b>${m.title}</a>`).join('');

  // topic cards on home
  const grid = $('#topic-grid');
  if (grid) grid.innerHTML = H.topics.map((t) => {
    const kids = t.children ? `<div class="chips" style="margin-top:10px">${t.children.map((c) => `<span class="chip">${c.icon} ${c.title}</span>`).join('')}</div>` : '';
    const inner = `<span class="ico" style="background:${t.color}26;color:${t.color}">${t.icon}</span><h3>${t.title}</h3><p>${t.desc}</p>${kids}<div style="margin-top:10px" class="chips">${t.status === 'live' ? (t.tags || []).slice(0, 3).map((x) => `<span class="chip">${x}</span>`).join('') + `<span class="badge live">live · ${pct(t)}%</span>` : '<span class="badge">coming soon</span>'}</div>`;
    return t.status === 'live' ? `<a class="card" href="${t.href}">${inner}</a>` : `<div class="card soon">${inner}</div>`;
  }).join('');

  // math + labs
  const ready = () => {
    $$('.eq,.tex').forEach((el) => { if (window.katex) { try { window.katex.render(el.textContent, el, { displayMode: el.classList.contains('eq'), throwOnError: false }); } catch (e) { /* leave raw TeX */ } } });
    $$('[data-lab]').forEach((el) => { try { AI.labs[el.dataset.lab](el); } catch (e) { el.textContent = 'Lab failed to load: ' + e.message; console.error(e); } });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready); else ready();
})();
