// Run from the repo root: node tools/check-links.js   (checks every local link, every #anchor, and duplicate lab names)
// Check every local link (and #anchor) in every HTML page, plus duplicate lab names across lab files.
const fs = require('fs');
const files = fs.readdirSync('.').filter((f) => f.endsWith('.html'));
const idsOf = {};
for (const f of files) { const h = fs.readFileSync(f, 'utf8'); idsOf[f] = new Set([...h.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1])); }
let bad = 0;
for (const f of files) {
  const h = fs.readFileSync(f, 'utf8');
  for (const m of h.matchAll(/href="([^"#]*\.html)?(#[^"]*)?"/g)) {
    const page = m[1], anchor = m[2] ? m[2].slice(1) : '';
    if (!page && !anchor) continue;
    const target = page || f;
    if (!idsOf[target]) { console.log(`${f}: link to missing page ${target}`); bad++; continue; }
    if (anchor && !idsOf[target].has(anchor) && !/^(top|main)$/.test(anchor)) { console.log(`${f}: ${target}#${anchor} - anchor not found`); bad++; }
  }
}
const seen = new Map();
for (const f of fs.readdirSync('assets').filter((x) => x.endsWith('.js'))) {
  const src = fs.readFileSync('assets/' + f, 'utf8');
  for (const m of src.matchAll(/AI\.labs(?:\.([A-Za-z0-9_]+)|\[['"]([^'"]+)['"]\])\s*=/g)) {
    const n = m[1] || m[2];
    if (seen.has(n) && seen.get(n) !== f) { console.log(`duplicate lab name "${n}" in ${seen.get(n)} and ${f}`); bad++; } else seen.set(n, f);
  }
}
console.log(`link/anchor/duplicate problems: ${bad}; lab definitions: ${seen.size}`);
