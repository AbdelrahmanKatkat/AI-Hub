// Interaction smoke test: for each page, scroll every lab into view, click its buttons, move its sliders / selects,
// and record console errors and page errors per lab. Also checks horizontal overflow at 390px width.
// usage: node tools/smoke.js <dir containing node_modules/puppeteer-core> <repo-dir> <regex of pages to include> <out.json>
// setup: in an empty folder run  npm init -y && npm install puppeteer-core ;  Chrome path is the CHROME constant below.
const path = require('path');
const fs = require('fs');
const puppeteer = require(path.join(process.argv[2], 'node_modules/puppeteer-core'));
const repo = process.argv[3];
const include = new RegExp(process.argv[4] || '.*');
const out = process.argv[5];
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const pages = fs.readdirSync(repo).filter((f) => f.endsWith('.html') && include.test(f)).sort();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];

async function testPage(browser, file) {
  const page = await browser.newPage();
  const errs = [];
  let current = '(load)';
  page.on('console', (m) => { if (m.type() === 'error') errs.push({ lab: current, msg: m.text().slice(0, 200) }); });
  page.on('pageerror', (e) => errs.push({ lab: current, msg: 'PAGEERROR ' + String(e.message).slice(0, 200) }));
  page.on('dialog', (d) => d.accept().catch(() => {}));
  await page.setViewport({ width: 1280, height: 900 });
  const url = 'file:///' + repo.replace(/\\/g, '/') + '/' + file;
  let labs = [];
  try {
    await page.goto(url, { waitUntil: 'load', timeout: 45000 });
    await sleep(1500);
    labs = await page.$$eval('[data-lab]', (els) => els.map((e, i) => { e.dataset.smokeIdx = i; return e.dataset.lab; }));
    for (let i = 0; i < labs.length; i++) {
      current = labs[i];
      const h = await page.$(`[data-smoke-idx="${i}"]`);
      if (!h) continue;
      await h.evaluate((e) => e.scrollIntoView({ block: 'center' }));
      await sleep(350);
      await page.evaluate((idx) => {
        const root = document.querySelector(`[data-smoke-idx="${idx}"]`);
        if (!root) return;
        root.querySelectorAll('input[type=range]').forEach((r) => {
          const min = +r.min || 0, max = +r.max || 100;
          for (const v of [max, min, (min + max) / 2]) { r.value = v; r.dispatchEvent(new Event('input', { bubbles: true })); r.dispatchEvent(new Event('change', { bubbles: true })); }
        });
        root.querySelectorAll('select').forEach((s) => { for (let k = 0; k < s.options.length; k++) { s.selectedIndex = k; s.dispatchEvent(new Event('input', { bubbles: true })); s.dispatchEvent(new Event('change', { bubbles: true })); } });
        root.querySelectorAll('input[type=checkbox]').forEach((c) => { c.click(); c.click(); });
        root.querySelectorAll('button, .chip').forEach((b) => { if (!/export|download/i.test(b.textContent)) b.click(); });
        root.querySelectorAll('canvas').forEach((c) => { const r = c.getBoundingClientRect(); for (const t of ['pointerdown', 'pointermove', 'pointerup']) c.dispatchEvent(new PointerEvent(t, { bubbles: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2, pointerId: 1 })); });
      }, i);
      await sleep(500);
    }
    current = '(phone)';
    await page.bringToFront();   // background tabs throttle timers and resize handling when pages run in parallel
    await page.setViewport({ width: 390, height: 800 });
    await sleep(1800);   // formulas re-fit 150 ms after a resize; allow time under load
    const over = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
    let culprits = null;
    if (over.sw > over.iw + 2) {
      // find what widens the document by hiding parts one at a time, then drilling down
      culprits = await page.evaluate(() => {
        const sw = () => document.documentElement.scrollWidth, base = sw(), res = [];
        const label = (e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (typeof e.className === 'string' && e.className ? '.' + e.className.trim().split(/\s+/).slice(0, 2).join('.') : '');
        const drill = (el, depth) => { for (const c of [...el.children]) { const old = c.style.display; c.style.display = 'none'; const w = sw(); c.style.display = old; if (w < base) { res.push('  '.repeat(depth) + label(c)); if (depth < 8) drill(c, depth + 1); } } };
        drill(document.body, 0); return res.slice(0, 14);
      });
    }
    results.push({ file, labs: labs.length, errors: errs, overflow: over.sw > over.iw + 2 ? over : null, culprits });
  } catch (e) {
    results.push({ file, labs: labs.length, errors: errs.concat([{ lab: '(run)', msg: 'TEST ERROR ' + e.message.slice(0, 200) }]), overflow: null });
  }
  await page.close();
}

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--disable-gpu', '--no-sandbox', '--allow-file-access-from-files'] });
  const queue = pages.slice();
  const worker = async () => { while (queue.length) { const f = queue.shift(); await testPage(browser, f); process.stdout.write('.'); } };
  await Promise.all([worker(), worker(), worker(), worker()]);
  await browser.close();
  results.sort((a, b) => a.file.localeCompare(b.file));
  fs.writeFileSync(out, JSON.stringify(results, null, 1));
  const bad = results.filter((r) => r.errors.length || r.overflow);
  console.log(`\npages: ${results.length}, labs: ${results.reduce((s, r) => s + r.labs, 0)}, pages with problems: ${bad.length}`);
  for (const r of bad) {
    console.log(`${r.file} (${r.labs} labs)` + (r.overflow ? `  OVERFLOW@390 ${r.overflow.sw}>${r.overflow.iw}` : ''));
    if (r.culprits) r.culprits.forEach((c) => console.log('     ' + c));
    const seen = new Set();
    r.errors.forEach((e) => { const k = e.lab + e.msg; if (!seen.has(k) && seen.size < 4) { seen.add(k); console.log(`   [${e.lab}] ${e.msg}`); } });
  }
})();
