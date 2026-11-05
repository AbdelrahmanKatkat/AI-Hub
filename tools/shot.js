// usage: node shot.js <puppeteer-dir> <repo-dir> <page> <anchor-or-empty> <out.png> [width] [height]
const path = require('path');
const puppeteer = require(path.join(process.argv[2], 'node_modules/puppeteer-core'));
(async () => {
  const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--disable-gpu', '--no-sandbox', '--allow-file-access-from-files'] });
  const page = await browser.newPage();
  const errs = [];
  page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text().slice(0, 160)); });
  page.on('pageerror', (e) => errs.push('PAGEERROR ' + e.message.slice(0, 160)));
  await page.setViewport({ width: +(process.argv[7] || 1300), height: +(process.argv[8] || 900) });
  await page.goto('file:///' + process.argv[3].replace(/\\/g, '/') + '/' + process.argv[4], { waitUntil: 'load' });
  await new Promise((r) => setTimeout(r, 3500));
  const info = await page.evaluate((a) => {
    if (a) { const el = document.getElementById(a); if (el) el.scrollIntoView(); }
    return { textLen: document.body.innerText.length, katex: document.querySelectorAll('.katex').length, labs: document.querySelectorAll('[data-lab] canvas').length, bg: getComputedStyle(document.body).backgroundColor };
  }, process.argv[5] || '');
  await new Promise((r) => setTimeout(r, 1500));
  await page.screenshot({ path: process.argv[6] });
  console.log(JSON.stringify(info), 'errors:', errs.length, errs.slice(0, 3));
  await browser.close();
})();
