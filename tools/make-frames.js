// Generates the panel frames and the neon theme CSS and writes them into assets/style.css between the markers
//   /* @@FRAMES-START */ ... /* @@FRAMES-END */
// Frames are nine-slice SVG border images: chamfered (cut) corners, a diagonal gradient outline and a baked-in glow, so they
// work on scrolling panels and cost nothing while labs animate. One set per theme (dark, light, neon).
// Run from the repo root:   node tools/make-frames.js
const fs = require('fs');

// frame image: G = glow margin, cuts = [top-left, top-right, bottom-right, bottom-left] chamfer sizes, sw = outline width
function frame({ G, cuts, sw, fill, stops, glow, blur, glowW }) {
  const maxCut = Math.max(...cuts), slice = G + maxCut + 2, size = slice * 2 + 16, L = G, R = size - G, T = G, B = size - G;
  const [a, b, c, d] = cuts;
  const pts = [[L + a, T], [R - b, T], [R, T + b], [R, B - c], [R - c, B], [L + d, B], [L, B - d], [L, T + a]].map((p) => p.join(',')).join(' ');
  const gradient = `<linearGradient id='g' gradientUnits='userSpaceOnUse' x1='${L}' y1='${T}' x2='${R}' y2='${B}'>${stops.map(([o, col]) => `<stop offset='${o}' stop-color='${col}'/>`).join('')}</linearGradient>`;
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${size}' height='${size}' viewBox='0 0 ${size} ${size}'><defs>${gradient}<filter id='b' x='-20%' y='-20%' width='140%' height='140%'><feGaussianBlur stdDeviation='${blur}'/></filter></defs>`
    + `<polygon points='${pts}' fill='${fill}'/>`
    + (glow ? `<polygon points='${pts}' fill='none' stroke='url(#g)' stroke-width='${glowW}' opacity='${glow}' filter='url(#b)'/>` : '')
    + `<polygon points='${pts}' fill='none' stroke='url(#g)' stroke-width='${sw}' stroke-linejoin='miter'/></svg>`;
  return { url: `url("data:image/svg+xml,${encodeURIComponent(svg).replace(/'/g, '%27')}")`, slice, G };
}

const NEON = { fill: 'rgba(3,8,28,0.88)', cyan: '#00e5ff', mag: '#ff2bd6', yel: '#ffe600', ora: '#ffb000' };
const themes = {
  // dark and light keep the chamfered shape but use the old plain colours: the old --border grey, the old panel fills,
  // blue for the active item and for the question box. No gradient and no glow.
  dark: {
    panel: { G: 5, cuts: [14, 6, 14, 6], sw: 1.2, fill: '#0d1117', stops: [[0, '#30363d'], [1, '#30363d']], glow: 0, blur: 1, glowW: 2 },
    card: { G: 5, cuts: [14, 6, 14, 6], sw: 1.2, fill: '#161b22', stops: [[0, '#30363d'], [1, '#30363d']], glow: 0, blur: 1, glowW: 2 },
    hot: { G: 5, cuts: [14, 6, 14, 6], sw: 1.2, fill: '#161b22', stops: [[0, '#30363d'], [1, '#30363d']], glow: 0, blur: 1, glowW: 2 },
    act: { G: 4, cuts: [9, 9, 9, 9], sw: 1.2, fill: '#161b22', stops: [[0, '#2f81f7'], [1, '#2f81f7']], glow: 0, blur: 1, glowW: 2 },
    sm: { G: 2, cuts: [5, 3, 5, 3], sw: 1.1, fill: '#161b22', stops: [[0, '#30363d'], [1, '#30363d']], glow: 0, blur: 1, glowW: 2 },
    smq: { G: 2, cuts: [5, 3, 5, 3], sw: 1.1, fill: 'rgba(0,0,0,0)', stops: [[0, '#2f81f7'], [1, '#2f81f7']], glow: 0, blur: 1, glowW: 2 },
  },
  light: {
    panel: { G: 5, cuts: [14, 6, 14, 6], sw: 1.2, fill: '#ffffff', stops: [[0, '#d0d7de'], [1, '#d0d7de']], glow: 0, blur: 1, glowW: 2 },
    card: { G: 5, cuts: [14, 6, 14, 6], sw: 1.2, fill: '#f6f8fa', stops: [[0, '#d0d7de'], [1, '#d0d7de']], glow: 0, blur: 1, glowW: 2 },
    hot: { G: 5, cuts: [14, 6, 14, 6], sw: 1.2, fill: '#f6f8fa', stops: [[0, '#d0d7de'], [1, '#d0d7de']], glow: 0, blur: 1, glowW: 2 },
    act: { G: 4, cuts: [9, 9, 9, 9], sw: 1.2, fill: '#f6f8fa', stops: [[0, '#0969da'], [1, '#0969da']], glow: 0, blur: 1, glowW: 2 },
    sm: { G: 2, cuts: [5, 3, 5, 3], sw: 1.1, fill: '#f6f8fa', stops: [[0, '#d0d7de'], [1, '#d0d7de']], glow: 0, blur: 1, glowW: 2 },
    smq: { G: 2, cuts: [5, 3, 5, 3], sw: 1.1, fill: 'rgba(0,0,0,0)', stops: [[0, '#0969da'], [1, '#0969da']], glow: 0, blur: 1, glowW: 2 },
  },
  neon: {
    panel: { G: 11, cuts: [16, 8, 16, 8], sw: 2, fill: NEON.fill, stops: [[0, NEON.yel], [0.16, NEON.cyan], [0.46, NEON.cyan], [0.64, NEON.mag], [1, NEON.mag]], glow: 0.95, blur: 3.2, glowW: 5 },
    hot: { G: 11, cuts: [16, 8, 16, 8], sw: 2.2, fill: NEON.fill, stops: [[0, NEON.yel], [0.5, NEON.yel], [1, NEON.ora]], glow: 0.95, blur: 3.2, glowW: 5 },
    act: { G: 7, cuts: [11, 11, 11, 11], sw: 2, fill: 'rgba(3,8,28,0.92)', stops: [[0, NEON.yel], [1, NEON.yel]], glow: 1, blur: 2.6, glowW: 4.5 },
    sm: { G: 4, cuts: [6, 3, 6, 3], sw: 1.3, fill: 'rgba(3,8,28,0.62)', stops: [[0, NEON.cyan], [0.55, '#1a8fc0'], [1, NEON.mag]], glow: 0.7, blur: 1.6, glowW: 3 },
    card: { G: 11, cuts: [16, 8, 16, 8], sw: 2, fill: NEON.fill, stops: [[0, NEON.yel], [0.16, NEON.cyan], [0.46, NEON.cyan], [0.64, NEON.mag], [1, NEON.mag]], glow: 0.95, blur: 3.2, glowW: 5 },
    smq: { G: 4, cuts: [6, 3, 6, 3], sw: 1.3, fill: 'rgba(3,8,28,0.62)', stops: [[0, NEON.cyan], [0.55, '#1a8fc0'], [1, NEON.mag]], glow: 0.7, blur: 1.6, glowW: 3 },
  },
};

const sel = {
  panel: '#side,#outline',
  card: '.lab,.pipe,.modcard,.card,.journey,.notes-drawer',
  hot: '.hero',
  sm: '.takeaways,.eq,.try,.callout,.worked,.aihook,.trap',
};
const vars = (name) => Object.entries(themes[name]).map(([k, v]) => `--fr-${k}:${frame(v).url};`).join('\n  ');
const sl = (k) => { const v = themes.dark[k]; return frame(v); };
const S = { panel: sl('panel'), hot: sl('hot'), act: sl('act'), sm: sl('sm') };
// the slice sizes differ per theme (glow margin), so the CSS reads them from variables
const sizes = Object.fromEntries(Object.keys(themes).map((t) => [t, Object.fromEntries(Object.keys(themes[t]).map((k) => [k, frame(themes[t][k])]))]));
const sizeVars = (t) => Object.keys(sizes[t]).map((k) => `--fr-${k}-s:${sizes[t][k].slice};--fr-${k}-g:${sizes[t][k].G}px;`).join('');

const ICONS = {
  math: "<path d='M18 5H6l7 7-7 7h12'/>",
  ml: "<circle cx='6' cy='7' r='2.4'/><circle cx='18' cy='7' r='2.4'/><circle cx='12' cy='17.5' r='2.4'/><path d='M7.8 8.8l3 6.6M16.2 8.8l-3 6.6M8.4 7h7.2'/>",
  ds: "<path d='M5 20V11M12 20V4M19 20v-6M3 20h18'/>",
  cv: "<path d='M3.5 8h3.2l1.8-3h7l1.8 3h3.2v11h-17z'/><circle cx='12' cy='13.5' r='3.6'/>",
  nlp: "<path d='M4 5h16v11H9.5L4 20z'/><path d='M8.5 10.5h.01M12 10.5h.01M15.5 10.5h.01'/>",
  voice: "<path d='M4 10v4M8 6v12M12 3v18M16 7v10M20 10v4'/>",
  genai: "<path d='M11 3l1.9 5.6L18.5 10.5 12.9 12.4 11 18l-1.9-5.6L3.5 10.5l5.6-1.9z'/><path d='M19 16v5M16.5 18.5h5'/>",
  gov: "<path d='M12 4v16M7 20h10M4 8h16'/><path d='M4 8l-2.5 7a3.5 3.5 0 005 0zM20 8l-2.5 7a3.5 3.5 0 005 0z'/>",
  rl: "<rect x='3' y='8' width='18' height='10' rx='5'/><path d='M8 11v4M6 13h4'/><circle cx='16' cy='12.2' r='.9'/><circle cx='18.2' cy='14.6' r='.9'/>",
  domains: "<circle cx='12' cy='12' r='9'/><path d='M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18'/>",
  geoai: "<path d='M12 21s7-6 7-11a7 7 0 10-14 0c0 5 7 11 7 11z'/><circle cx='12' cy='10' r='2.6'/>",
  robotics: "<rect x='5' y='8' width='14' height='10' rx='2.5'/><path d='M12 8V4.5M9 13h.01M15 13h.01M8.5 18v2.5M15.5 18v2.5'/>",
};
const iconCss = Object.entries(ICONS).map(([id, d]) => `:root[data-theme="neon"] .topic[data-t="${id}"] .ico{--icn:url("data:image/svg+xml,${encodeURIComponent(`<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'>${d}</svg>`)}")}`).join('\n');

const css = `/* @@FRAMES-START (generated by tools/make-frames.js: do not edit by hand) */
:root{
  ${vars('dark')}
  ${sizeVars('dark')}
  --stage-fill:#0d1117;
}
:root[data-theme="light"]{
  ${vars('light')}
  ${sizeVars('light')}
  --stage-fill:#ffffff;
}
:root[data-theme="neon"]{
  --bg:#050914;--bg2:#0a1030;--bg3:#141a46;--border:#1b6a8f;--fg:#f2f8ff;--muted:#9db9d8;
  --blue:#00e5ff;--green:#35ffa5;--purple:#b266ff;--orange:#ff9a1f;--yellow:#ffe600;--red:#ff4d73;--pink:#ff2bd6;--teal:#00f0e0;
  --link:#35e0ff;--btn-green:#ffe600;--shadow:0 10px 30px rgba(0,0,0,.75);
  ${vars('neon')}
  ${sizeVars('neon')}
  --stage-fill:rgba(3,8,28,.93);
}

/* panel frames: chamfered corners, gradient outline, glow (images above; borders are drawn by the image, not by CSS) */
${sel.panel}{border:2px solid transparent;border-radius:0;background:none;border-image:var(--fr-panel) var(--fr-panel-s) fill / calc(var(--fr-panel-s) * 1px) / var(--fr-panel-g) stretch}
${sel.card}{border:2px solid transparent;border-radius:0;background:none;border-image:var(--fr-card) var(--fr-card-s) fill / calc(var(--fr-card-s) * 1px) / var(--fr-card-g) stretch}
${sel.hot}{border:2px solid transparent;border-radius:0;background:none;border-image:var(--fr-hot) var(--fr-hot-s) fill / calc(var(--fr-hot-s) * 1px) / var(--fr-hot-g) stretch;padding:20px 26px;margin-left:var(--fr-hot-g);margin-right:var(--fr-hot-g)}
${sel.sm}{border:1.5px solid transparent;border-radius:0;border-image:var(--fr-sm) var(--fr-sm-s) fill / calc(var(--fr-sm-s) * 1px) / var(--fr-sm-g) stretch;background-color:transparent}
.worked{box-shadow:inset 3px 0 0 var(--green)}.aihook{box-shadow:inset 3px 0 0 var(--purple)}.trap{box-shadow:inset 3px 0 0 var(--red)}.callout{box-shadow:inset 3px 0 0 var(--blue)}.callout.warn{box-shadow:inset 3px 0 0 var(--yellow)}
.question{border:1.5px solid transparent;border-radius:0;border-image:var(--fr-smq) var(--fr-smq-s) fill / calc(var(--fr-smq-s) * 1px) / var(--fr-smq-g) stretch;background-color:color-mix(in srgb,var(--blue) 7%,transparent)}
.topic.active{border:2px solid transparent;border-radius:0;background:none;border-image:var(--fr-act) var(--fr-act-s) fill / calc(var(--fr-act-s) * 1px) / var(--fr-act-g) stretch}
/* the sidebar and outline keep their mobile behaviour (overlay) below 861px */
/* sidebar and outline are floating framed panels on wide screens */
@media(min-width:861px){
  #side,#outline{top:72px;height:calc(100vh - 84px);margin:12px 0 12px 12px}
  #outline{margin:12px 12px 12px 0}
}
/* every lesson section is a framed panel; the page art shows in the gaps (inline styles on hub sections are overridden) */
.lesson{border-width:2px!important;border-style:solid!important;border-color:transparent!important;border-radius:0;background:none!important;border-image:var(--fr-panel) var(--fr-panel-s) fill / calc(var(--fr-panel-s) * 1px) / var(--fr-panel-g) stretch!important;padding:22px 28px!important;margin:26px var(--fr-panel-g)!important}
.lesson.lesson .lesson-head{margin-top:0}
/* pipeline stage cards: chamfered like every panel. Dark and light: the old grey outline with a coloured top edge. Neon: a coloured, glowing outline. */
.stage{position:relative;isolation:isolate;background:none;border:0;border-radius:0;padding:12px 14px;transition:filter .25s,transform .25s}
.stage::before{content:"";position:absolute;inset:0;z-index:-2;background:var(--stage-ring,var(--border));clip-path:polygon(11px 0,calc(100% - 5px) 0,100% 5px,100% calc(100% - 11px),calc(100% - 11px) 100%,5px 100%,0 calc(100% - 5px),0 11px)}
.stage::after{content:"";position:absolute;inset:2px;z-index:-1;background:var(--stage-fill);box-shadow:var(--stage-accent,none);clip-path:polygon(10px 0,calc(100% - 4px) 0,100% 4px,100% calc(100% - 10px),calc(100% - 10px) 100%,4px 100%,0 calc(100% - 4px),0 10px)}
:root:not([data-theme="neon"]) .stage{--stage-ring:var(--border);--stage-accent:inset 0 3px 0 var(--c)}
:root:not([data-theme="neon"]) .stage:hover::before,:root:not([data-theme="neon"]) .stage.pinned::before{background:var(--c)}
:root[data-theme="neon"] .stage{--stage-ring:var(--c);filter:drop-shadow(0 0 4px color-mix(in srgb,var(--c) 55%,transparent))}
:root[data-theme="neon"] .stage:hover{filter:drop-shadow(0 0 7px color-mix(in srgb,var(--c) 80%,transparent))}
.stage.active{box-shadow:none;transform:translateY(-2px);filter:drop-shadow(0 0 9px color-mix(in srgb,var(--c) 60%,transparent))}
:root[data-theme="neon"] .stage.active{filter:drop-shadow(0 0 11px var(--c))}
.stage.pinned{border-color:transparent}

/* header: the old plain line under the bar in dark and light; the gradient is neon only (below) */

/* ===== neon theme ===== */
:root[data-theme="neon"] body{background:#050914}
:root[data-theme="neon"] body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;background:linear-gradient(rgba(4,6,20,.30),rgba(4,6,20,.30)),url(neon-city.jpg) center/cover no-repeat}
:root[data-theme="neon"] .app,:root[data-theme="neon"] footer{position:relative;z-index:1}
:root[data-theme="neon"] #hdr{background:rgba(3,8,28,.9);border-bottom:2px solid transparent;border-image:linear-gradient(90deg,var(--yellow),var(--teal) 16%,var(--blue) 60%,var(--pink)) 1;box-shadow:0 8px 26px -10px rgba(0,229,255,.8)}
:root[data-theme="neon"] h1,:root[data-theme="neon"] h2{text-shadow:0 0 16px rgba(0,229,255,.35)}
:root[data-theme="neon"] .hero h1 .hl,:root[data-theme="neon"] .lesson-head h2 .hl{color:var(--yellow);font-style:italic;text-shadow:0 0 14px rgba(255,230,0,.5)}
:root[data-theme="neon"] .chip{background:rgba(3,8,28,.72);border-color:rgba(0,229,255,.6);box-shadow:0 0 8px rgba(0,229,255,.25)}
:root[data-theme="neon"] .btn{background:rgba(3,8,28,.8);border-color:rgba(0,229,255,.55)}
:root[data-theme="neon"] .btn:hover{border-color:var(--yellow);box-shadow:0 0 10px rgba(255,230,0,.45)}
:root[data-theme="neon"] .btn.primary{color:#10121f!important;border-color:transparent;background:var(--yellow);box-shadow:0 0 14px rgba(255,230,0,.55);clip-path:polygon(9px 0,calc(100% - 9px) 0,100% 50%,calc(100% - 9px) 100%,9px 100%,0 50%);padding-left:18px;padding-right:18px}
:root[data-theme="neon"] .btn.on{border-color:var(--yellow);color:var(--yellow)!important}
:root[data-theme="neon"] .topic .ico{box-shadow:0 0 9px currentColor}
:root[data-theme="neon"] .topic.active .nm{color:#fff;font-weight:600}
:root[data-theme="neon"] .bar i{background:linear-gradient(90deg,var(--teal),var(--pink));box-shadow:0 0 8px var(--pink)}
:root[data-theme="neon"] .lesson{border-top-color:rgba(0,229,255,.35)}
:root[data-theme="neon"] .tbl td,:root[data-theme="neon"] .tbl th{background:rgba(3,8,28,.7)}
:root[data-theme="neon"] .notes-fab{background:rgba(3,8,28,.9);border-color:var(--yellow);box-shadow:0 0 12px rgba(255,230,0,.5);color:var(--yellow)}

:root[data-theme="neon"] h1,:root[data-theme="neon"] h2,:root[data-theme="neon"] h3,:root[data-theme="neon"] .topic .nm,:root[data-theme="neon"] .btn,:root[data-theme="neon"] .chip,:root[data-theme="neon"] .brand b,:root[data-theme="neon"] .side-title,:root[data-theme="neon"] .lesson h3.sec{font-family:'Poppins',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
:root[data-theme="neon"] body::before{background:linear-gradient(rgba(4,6,20,.16),rgba(4,6,20,.16)),url(neon-city.jpg) center/cover no-repeat}
:root[data-theme="neon"] .topic.active .ico{background:var(--yellow)!important;color:#10121f!important;box-shadow:0 0 10px rgba(255,230,0,.6)}
:root[data-theme="neon"] .topic[data-t] .ico{font-size:0;border-radius:0;background:currentColor!important;-webkit-mask:var(--icn) center/64% no-repeat;mask:var(--icn) center/64% no-repeat;box-shadow:none}
:root[data-theme="neon"] .topic.active[data-t] .ico{background:var(--yellow)!important;color:#10121f}
:root[data-theme="neon"] .hero .big{position:relative;background:none!important;border:0;color:var(--pink);filter:drop-shadow(0 0 8px var(--pink))}
:root[data-theme="neon"] .hero .big::before{content:"";position:absolute;inset:-7px;pointer-events:none;background:
  linear-gradient(var(--yellow),var(--yellow)) 0 0/18px 3px no-repeat,linear-gradient(var(--yellow),var(--yellow)) 0 0/3px 18px no-repeat,
  linear-gradient(var(--yellow),var(--yellow)) 100% 0/18px 3px no-repeat,linear-gradient(var(--yellow),var(--yellow)) 100% 0/3px 18px no-repeat,
  linear-gradient(var(--yellow),var(--yellow)) 0 100%/18px 3px no-repeat,linear-gradient(var(--yellow),var(--yellow)) 0 100%/3px 18px no-repeat,
  linear-gradient(var(--yellow),var(--yellow)) 100% 100%/18px 3px no-repeat,linear-gradient(var(--yellow),var(--yellow)) 100% 100%/3px 18px no-repeat}
${iconCss}

/* phone: the framed hero should not eat the width */
@media(max-width:600px){.hero{padding:14px 16px;gap:12px;margin-left:0;margin-right:0}.hero .big{width:46px;height:46px;font-size:22px;border-radius:12px}.hero h1{font-size:22px}}
/* @@FRAMES-END */`;

let s = fs.readFileSync('assets/style.css', 'utf8');
const a = s.indexOf('/* @@FRAMES-START'), b = s.indexOf('/* @@FRAMES-END */');
if (a >= 0 && b > a) s = s.slice(0, a).replace(/\s+$/, '\n') + s.slice(b + '/* @@FRAMES-END */'.length);
fs.writeFileSync('assets/style.css', s.replace(/\s+$/, '\n') + '\n' + css + '\n');
console.log('frames written; style.css', (fs.statSync('assets/style.css').size / 1024).toFixed(1) + ' KB');
