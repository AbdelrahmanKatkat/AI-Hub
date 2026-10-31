/* GeoAI · Module 0 labs (geo-foundations.html).
 * Mounted by <div data-lab="geocoords | geoglobe | geocrs | georaster | geobands | geosensors">.
 * The maths is exposed as AI.geo (projections, UTM, shoelace, sphere area) so it can be tested in isolation. */
(function () {
  const AI = window.AI;
  const R = 6371008.8;                        // mean Earth radius in metres (sphere formulas)
  const D2R = Math.PI / 180;
  const M_DEG = (2 * Math.PI * R) / 360;      // metres per degree of latitude (about 111 195 m)
  const F = (v, d = 0) => (Number.isFinite(v) ? v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '–');
  const G = (AI.geo = {});
  G.R = R; G.M_DEG = M_DEG;

  /* ---------- projections. lon/lat in degrees in, plane units out (radius R = 1 unit for the sphere forms) ---------- */
  G.equirect = (lon, lat) => [lon * D2R, lat * D2R];
  G.mercator = (lon, lat) => {
    const p = AI.clamp(lat, -85, 85) * D2R;
    return [lon * D2R, Math.log(Math.tan(Math.PI / 4 + p / 2))];
  };
  G.mollweide = (lon, lat) => {
    const phi = lat * D2R, lam = lon * D2R;
    let th = phi;                              // Newton solve of 2θ + sin2θ = π sinφ
    for (let i = 0; i < 60; i++) {
      const f = 2 * th + Math.sin(2 * th) - Math.PI * Math.sin(phi), dd = 2 + 2 * Math.cos(2 * th);
      if (Math.abs(dd) < 1e-12) break;
      const s = f / dd;
      th -= s;
      if (Math.abs(s) < 1e-13) break;
    }
    return [(2 * Math.SQRT2 / Math.PI) * lam * Math.cos(th), Math.SQRT2 * Math.sin(th)];
  };
  /* orthographic view centred on (lon0, lat0); null when the point is on the far side */
  G.ortho = (lon, lat, lon0 = 0, lat0 = 0) => {
    const p = lat * D2R, p0 = lat0 * D2R, l = (lon - lon0) * D2R;
    if (Math.sin(p0) * Math.sin(p) + Math.cos(p0) * Math.cos(p) * Math.cos(l) < 0) return null;
    return [Math.cos(p) * Math.sin(l), Math.cos(p0) * Math.sin(p) - Math.sin(p0) * Math.cos(p) * Math.cos(l)];
  };
  /* shoelace area of a closed (or open, auto-closed) ring of [x, y] */
  G.shoelace = (pts) => {
    let s = 0;
    for (let i = 0; i < pts.length; i++) {
      const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % pts.length];
      s += x1 * y2 - x2 * y1;
    }
    return Math.abs(s) / 2;
  };
  /* boundary of a lon/lat box, densified: returns [lon, lat] pairs */
  G.boxRing = (l1, l2, p1, p2, n = 24) => {
    const ring = [];
    for (let k = 0; k < n; k++) ring.push([l1 + ((l2 - l1) * k) / n, p1]);
    for (let k = 0; k < n; k++) ring.push([l2, p1 + ((p2 - p1) * k) / n]);
    for (let k = 0; k < n; k++) ring.push([l2 - ((l2 - l1) * k) / n, p2]);
    for (let k = 0; k < n; k++) ring.push([l1, p2 - ((p2 - p1) * k) / n]);
    return ring;
  };
  /* exact area of a lon/lat box on a sphere of radius R, in m² */
  G.sphereArea = (l1, l2, p1, p2) => R * R * (l2 - l1) * D2R * (Math.sin(p2 * D2R) - Math.sin(p1 * D2R));

  /* ---------- UTM (Krüger/Karney series, WGS 84), k0 = 0.9996 ---------- */
  G.utmZone = (lon) => Math.floor((lon + 180) / 6) + 1;
  G.utm = (lon, lat, zone = G.utmZone(lon)) => {
    const a = 6378137, f = 1 / 298.257223563, k0 = 0.9996;
    const n = f / (2 - f), n2 = n * n, n3 = n2 * n, n4 = n3 * n;
    const A = ((a / (1 + n)) * (1 + n2 / 4 + n4 / 64));
    const beta = [                                // Krüger alpha coefficients (checked against PROJ to ~1 cm)
      n / 2 - (2 * n2) / 3 + (5 * n3) / 16 + (41 * n4) / 180,
      (13 * n2) / 48 - (3 * n3) / 5 + (557 * n4) / 1440,
      (61 * n3) / 240 - (103 * n4) / 140,
      (49561 * n4) / 161280,
    ];
    const e = Math.sqrt(f * (2 - f));
    const phi = lat * D2R, dl = lon * D2R - (6 * zone - 183) * D2R;
    const tau = Math.tan(phi);
    const sig = Math.sinh(e * Math.atanh((e * tau) / Math.sqrt(1 + tau * tau)));
    const tauP = tau * Math.sqrt(1 + sig * sig) - sig * Math.sqrt(1 + tau * tau);
    const xiP = Math.atan2(tauP, Math.cos(dl));
    const etaP = Math.asinh(Math.sin(dl) / Math.sqrt(tauP * tauP + Math.cos(dl) * Math.cos(dl)));
    let xi = xiP, eta = etaP;
    beta.forEach((b, j) => {
      const k = j + 1;
      xi += b * Math.sin(2 * k * xiP) * Math.cosh(2 * k * etaP);
      eta += b * Math.cos(2 * k * xiP) * Math.sinh(2 * k * etaP);
    });
    let E = 500000 + k0 * A * eta, N = k0 * A * xi;
    if (lat < 0) N += 10000000;
    return [E, N];
  };

  /* ---------- helpers for the rasters ---------- */
  const pip = (poly, x, y) => {                       // ray casting, odd crossings = inside
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  };
  const segDist = (px, py, [ax, ay], [bx, by]) => {
    const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
    const t = AI.clamp(L ? ((px - ax) * dx + (py - ay) * dy) / L : 0, 0, 1);
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  };
  const lineDist = (pts, x, y) => {
    let m = Infinity;
    for (let i = 0; i < pts.length - 1; i++) m = Math.min(m, segDist(x, y, pts[i], pts[i + 1]));
    return m;
  };
  const polyLen = (pts) => {
    let L = 0;
    for (let i = 0; i < pts.length - 1; i++) L += Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
    return L;
  };

  /* ==================================================================== */
  /* 1. geocoords: one lat/lon cell drawn to scale                          */
  /* ==================================================================== */
  AI.labs.geocoords = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, d.title || 'Degrees are not metres', d.sub || 'one latitude–longitude cell, drawn to scale',
      '<canvas style="height:260px"></canvas><div class="readout" id="o" style="margin-top:10px"></div>',
      `<label>Latitude φ <span class="v" id="lv"></span><input type="range" id="lat" min="-84" max="84" step="0.5" value="${+(d.lat || 52)}"></label>
       <label>Cell size in degrees (latitude and longitude)<input type="number" id="cell" step="any" min="0.000001" max="10" value="${+(d.cell || 1)}"></label>
       <div class="err" id="er"></div>
       <div class="caption">Dashed square: 10 m × 10 m, for scale. Every degree of latitude is about 111.2 km; a degree of longitude is 111.2 km × cos φ.</div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    let shown = +q('#lat').value, key = '';
    AI.loop(root, (dt) => {
      const target = +q('#lat').value;
      shown += (target - shown) * (1 - Math.exp(-dt * 5));
      if (Math.abs(target - shown) < 1e-3) shown = target;
      q('#lv').textContent = `${F(target, 1)}°`;
      const cell = +q('#cell').value;
      if (!(cell > 0)) { q('#er').textContent = 'Type a positive number of degrees.'; return; }
      q('#er').textContent = '';
      const k = [shown.toFixed(3), cell, c.w, c.h, document.documentElement.dataset.theme].join('|');
      if (k === key) return;
      key = k;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const hM = cell * M_DEG, wM = hM * Math.cos(shown * D2R);
      const s = Math.min((0.6 * H) / hM, (0.5 * W) / hM);     // pixels per metre
      const cx = W / 2, cy = H / 2;
      g.fillStyle = C.blue; g.globalAlpha = 0.18; g.fillRect(cx - (wM * s) / 2, cy - (hM * s) / 2, wM * s, hM * s); g.globalAlpha = 1;
      g.strokeStyle = C.orange; g.lineWidth = 2.5; g.strokeRect(cx - (wM * s) / 2, cy - (hM * s) / 2, wM * s, hM * s);
      g.setLineDash([5, 4]); g.strokeStyle = C.muted; g.lineWidth = 1.5; g.strokeRect(cx - 5 * s, cy - 5 * s, 10 * s, 10 * s); g.setLineDash([]);
      g.fillStyle = C.muted; g.font = '12px sans-serif'; g.textAlign = 'center';
      g.fillText(`${F(hM)} m north–south`, cx, cy + (hM * s) / 2 + 16);
      g.save(); g.translate(cx - (wM * s) / 2 - 12, cy); g.rotate(-Math.PI / 2); g.fillText(`${F(wM)} m east–west`, 0, 0); g.restore();
      const barM = AI.niceStep((0.3 * W) / s, 4), bx = 16, by = H - 16;
      g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(bx, by); g.lineTo(bx + barM * s, by); g.stroke();
      g.textAlign = 'left'; g.fillText(barM >= 1000 ? `${F(barM / 1000)} km` : `${F(barM)} m`, bx, by - 6);
      q('#o').innerHTML = `φ = <b class="o">${F(shown, 2)}°</b>   cell = <b class="b">${cell}°</b>  →  ${F(hM, 2)} m tall, ${F(wM, 2)} m wide\n`
        + `width ÷ height = cos φ = <b>${F(Math.cos(shown * D2R), 3)}</b>      area of one cell = ${F(hM * wM)} m²`;
    });
  };

  /* ==================================================================== */
  /* 2. geoglobe: the same box in four views of the Earth                   */
  /* ==================================================================== */
  AI.labs.geoglobe = function (root) {
    const d = root.dataset;
    const P = {
      eq: { name: 'Equirectangular (plate carrée)', bb: [-Math.PI, Math.PI, -Math.PI / 2, Math.PI / 2], note: 'Neither equal-area nor conformal: the grid is evenly spaced, so high latitudes are stretched.' },
      merc: { name: 'Web Mercator (EPSG:3857)', bb: [-Math.PI, Math.PI, -3.15, 3.15], note: 'Conformal: angles and small shapes are right, but areas grow as sec²φ.' },
      moll: { name: 'Mollweide (equal-area)', bb: [-2.83, 2.83, -1.42, 1.42], note: 'Equal-area: areas are right everywhere; shapes are sheared away from the centre.' },
      ortho: { name: 'Orthographic (globe view)', bb: [-1.05, 1.05, -1.05, 1.05], note: 'One hemisphere seen from far away: it looks like a globe, nearly undistorted only near the centre.' },
    };
    const names = Object.keys(P), FN = { eq: G.equirect, merc: G.mercator, moll: G.mollweide };
    const { view, ctl } = AI.labShell(root, d.title || 'From globe to map', 'the same grid and the same box, in each projection',
      '<canvas style="height:320px"></canvas><div class="readout" id="o" style="margin-top:10px"></div>',
      `<label>Projection <select id="pj">${names.map((k) => `<option value="${k}"${k === (d.proj || 'merc') ? ' selected' : ''}>${P[k].name}</option>`).join('')}</select></label>
       <label>Box centre latitude <span class="v" id="lav"></span><input type="range" id="lat" min="-70" max="70" step="1" value="${+(d.lat || 50)}"></label>
       <label>Box centre longitude <span class="v" id="lov"></span><input type="range" id="lon" min="-180" max="180" step="1" value="${+(d.lon || 10)}"></label>
       <label>Box size, degrees <span class="v" id="sv"></span><input type="range" id="size" min="2" max="30" step="1" value="${+(d.size || 10)}"></label>
       <div class="caption">The orange box is the same ground in every view. Change the projection and watch the grid morph (about 1.4 s) while the area readout updates.</div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    let from = q('#pj').value, to = from, t = 1, key = '';
    q('#pj').addEventListener('change', () => { from = to; to = q('#pj').value; t = 0; key = ''; });
    const proj = (k, lon, lat, lc, pc) => (k === 'ortho' ? G.ortho(lon, lat, lc, pc) : FN[k](lon, lat));
    AI.loop(root, (dt) => {
      if (t < 1) t = Math.min(1, t + dt / 1.4);
      const s = AI.smooth(t);
      const lc = +q('#lon').value, pc = +q('#lat').value, size = +q('#size').value;
      q('#lav').textContent = `${pc}°`; q('#lov').textContent = `${lc}°`; q('#sv').textContent = `${size}°`;
      const k = [from, to, t.toFixed(3), lc, pc, size, c.w, c.h, document.documentElement.dataset.theme].join('|');
      if (k === key) return;
      key = k;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      const bA = P[from].bb, bB = P[to].bb, bb = bA.map((v, i) => AI.lerp(v, bB[i], s));
      const sc = Math.min(W / (bb[1] - bb[0]), H / (bb[3] - bb[2])) * 0.94;
      const X = (u) => W / 2 + u * sc, Y = (v) => H / 2 - v * sc;
      const pt = (lon, lat) => {
        const a = proj(from, lon, lat, lc, pc), b = proj(to, lon, lat, lc, pc);
        return a && b ? [AI.lerp(a[0], b[0], s), AI.lerp(a[1], b[1], s)] : null;
      };
      const poly = (pts) => {
        g.beginPath(); let pen = false;
        for (const p of pts) {
          if (!p) { pen = false; continue; }
          if (pen) g.lineTo(X(p[0]), Y(p[1])); else g.moveTo(X(p[0]), Y(p[1]));
          pen = true;
        }
        g.stroke();
      };
      g.clearRect(0, 0, W, H);
      const disc = (from === 'ortho' ? 1 - s : 0) + (to === 'ortho' ? s : 0);
      if (disc > 0.01) { g.globalAlpha = disc * 0.5; g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath(); g.arc(X(0), Y(0), sc, 0, 2 * Math.PI); g.stroke(); g.globalAlpha = 1; }
      g.strokeStyle = C.muted; g.globalAlpha = 0.5; g.lineWidth = 1;
      for (let lo = -150; lo <= 150; lo += 30) poly(Array.from({ length: 61 }, (_, i) => pt(lo, -90 + i * 3)));
      for (let la = -60; la <= 60; la += 30) poly(Array.from({ length: 121 }, (_, i) => pt(-180 + i * 3, la)));
      g.globalAlpha = 1;
      const l1 = lc - size / 2, l2 = lc + size / 2, p1 = AI.clamp(pc - size / 2, -84, 84), p2 = AI.clamp(pc + size / 2, -84, 84);
      const ring = G.boxRing(l1, l2, p1, p2, 24);
      const rp = ring.map(([lo, la]) => pt(lo, la));
      if (rp.every(Boolean)) { g.globalAlpha = 0.28; g.fillStyle = C.orange; g.beginPath(); rp.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])))); g.closePath(); g.fill(); g.globalAlpha = 1; }
      g.strokeStyle = C.orange; g.lineWidth = 2.5; poly(rp);
      const trueA = G.sphereArea(l1, l2, p1, p2) / 1e6;
      let line2;
      if (to === 'ortho') line2 = 'a globe view has no planar area; only the true area applies';
      else {
        const planarA = (G.shoelace(ring.map(([lo, la]) => proj(to, lo, la, lc, pc))) * R * R) / 1e6;
        line2 = `area measured on this map = ${F(planarA)} km², ratio to true area = <b class="o">${F(planarA / trueA, 2)}</b>`;
      }
      q('#o').innerHTML = `view: <b>${P[to].name}</b>${t < 1 ? ' <span class="m">(morphing…)</span>' : ''}\n`
        + `true area of the box (sphere, R = 6 371 km) = <b class="b">${F(trueA)} km²</b>\n${line2}\n<span class="m">${P[to].note}</span>`;
    });
  };

  /* ==================================================================== */
  /* 3. geocrs: one point in several coordinate systems                     */
  /* ==================================================================== */
  AI.labs.geocrs = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, d.title || 'One point, several coordinate systems', 'change the point; the same place gets different numbers',
      '<canvas style="height:150px"></canvas><table class="tbl" id="tb" style="font-size:12px"></table>',
      `<label>Latitude φ (°N) <input type="number" id="lat" step="0.0001" min="-84" max="84" value="${+(d.lat || 30.0444)}"></label>
       <label>Longitude λ (°E) <span class="v" id="lv"></span><input type="range" id="lon" min="18" max="42" step="0.0001" value="${+(d.lon || 31.2357)}"></label>
       <label>Compare with UTM zone <select id="wz">${[34, 35, 36, 37, 38].map((z) => `<option${z === +(d.wrong || 35) ? ' selected' : ''}>${z}</option>`).join('')}</select></label>
       <div class="err" id="er"></div>
       <div class="caption">The orange line is the central meridian of the zone you compare with. Crossing a zone edge changes the zone number and jumps the easting.</div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    let shownLon = +q('#lon').value, key = '';
    const X = (lon, W) => 30 + ((lon - 18) / 24) * (W - 60);
    AI.loop(root, (dt) => {
      const lat = +q('#lat').value, lon = +q('#lon').value, wz = +q('#wz').value;
      if (!Number.isFinite(lat) || Math.abs(lat) > 84) { q('#er').textContent = 'Latitude must be between −84 and 84 for UTM.'; return; }
      q('#er').textContent = '';
      q('#lv').textContent = `${F(lon, 3)}°`;
      shownLon += (lon - shownLon) * (1 - Math.exp(-dt * 6));
      if (Math.abs(lon - shownLon) < 1e-4) shownLon = lon;
      const k = [shownLon.toFixed(4), lat, lon, wz, c.w, c.h, document.documentElement.dataset.theme].join('|');
      if (k === key) return;
      key = k;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const z = G.utmZone(lon);
      for (let zz = 34; zz <= 37; zz++) {
        const l0 = 6 * (zz - 1) - 180, l1 = l0 + 6;
        g.fillStyle = zz % 2 ? C.bg2 : C.bg3; g.fillRect(X(l0, W), 8, X(l1, W) - X(l0, W), H - 34);
        g.fillStyle = C.muted; g.font = '12px sans-serif'; g.textAlign = 'center';
        g.fillText(`zone ${zz}`, (X(l0, W) + X(l1, W)) / 2, 24);
        g.fillText(`${l0}°E`, X(l0, W), H - 10);
      }
      g.fillText('42°E', X(42, W), H - 10);
      const cmOf = (zz) => 6 * zz - 183;
      g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(X(cmOf(z), W), 8); g.lineTo(X(cmOf(z), W), H - 26); g.stroke();
      g.strokeStyle = C.orange; g.beginPath(); g.moveTo(X(cmOf(wz), W), 8); g.lineTo(X(cmOf(wz), W), H - 26); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.orange; g.textAlign = 'left'; g.fillText(`central meridian of zone ${wz}: ${cmOf(wz)}°E`, 34, 44);
      g.fillStyle = C.blue; g.beginPath(); g.arc(X(shownLon, W), H * 0.52, 7, 0, 7); g.fill();
      g.fillStyle = C.fg; g.textAlign = 'center'; g.fillText(`${F(lat, 2)}°N, ${F(lon, 2)}°E`, X(shownLon, W), H * 0.52 - 12);
      const cm = cmOf(z);
      const [e1, n1] = G.utm(lon, lat, z), [e2, n2] = G.utm(lon, lat, wz);
      const mx = 6378137 * lon * D2R, my = 6378137 * Math.log(Math.tan(Math.PI / 4 + (AI.clamp(lat, -85, 85) * D2R) / 2));
      const scale = 1 / Math.cos(lat * D2R);
      const epsg = (zz) => (lat < 0 ? 32700 : 32600) + zz;
      q('#tb').innerHTML = `<thead><tr><th>system</th><th>what it is</th><th>this point</th><th>note</th></tr></thead><tbody>
        <tr><td>EPSG:4326</td><td>WGS 84, degrees</td><td>φ ${F(lat, 4)}°, λ ${F(lon, 4)}°</td><td>angles, not distances</td></tr>
        <tr><td>EPSG:3857</td><td>Web Mercator, metres</td><td>x ${F(mx, 1)} m, y ${F(my, 1)} m</td><td>scale ${F(scale, 3)}; area off by ${F(scale * scale, 2)}×</td></tr>
        <tr><td>EPSG:${epsg(z)}</td><td>UTM zone ${z}, metres</td><td>E ${F(e1, 1)} m, N ${F(n1, 1)} m</td><td>correct zone: central meridian ${cm}°E</td></tr>
        <tr><td>EPSG:${epsg(wz)}</td><td>UTM zone ${wz}, metres</td><td>E ${F(e2, 1)} m, N ${F(n2, 1)} m</td><td>${wz === z ? 'same as the correct zone' : `wrong zone: ${F(lon - cmOf(wz), 2)}° from its central meridian`}</td></tr></tbody>`;
    });
  };

  /* ==================================================================== */
  /* 4. georaster: the same map as vector objects and as cells              */
  /* ==================================================================== */
  AI.labs.georaster = function (root) {
    const d = root.dataset;
    const W0 = 100, H0 = 70;
    const LAKE = [[56, 10], [78, 8], [90, 20], [86, 34], [66, 38], [54, 28]];
    const ROAD = [[4, 58], [30, 48], [52, 52], [70, 36], [96, 42]];
    const WELL = [22, 18];
    const CL = ['background', 'lake', 'road', 'well'];
    const { view, ctl } = AI.labShell(root, d.title || 'Vector or raster?', 'the same map, stored two ways',
      '<canvas style="height:300px;cursor:crosshair;touch-action:none"></canvas><div class="readout" id="o" style="margin-top:10px"></div>',
      `<div class="row"><button class="btn small" data-m="vector">Vector</button><button class="btn small" data-m="raster">Raster</button><button class="btn small" data-m="both">Both</button></div>
       <label>Cells across the map <span class="v" id="nv"></span><input type="range" id="n" min="6" max="40" step="1" value="${+(d.n || 16)}"></label>
       <div class="caption">Click a cell to read its centre and class from the geotransform. A raster line is at least one cell wide, so thin objects are approximated.</div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    let N = 16, cell = W0 / N, rows = 11, H = rows * cell, cls = [], sel = null, mode = d.mode || 'both', va = 1, ra = 0.55, key = '';
    const build = () => {
      N = +q('#n').value; cell = W0 / N; rows = Math.max(1, Math.round(H0 / cell)); H = rows * cell;
      q('#nv').textContent = `${N} × ${rows}`;
      cls = [];
      for (let r = 0; r < rows; r++) {
        const row = [];
        for (let cc = 0; cc < N; cc++) {
          const x = (cc + 0.5) * cell, y = H - (r + 0.5) * cell;
          let v = 0;
          if (pip(LAKE, x, y)) v = 1;
          if (lineDist(ROAD, x, y) <= cell / 2) v = 2;
          row.push(v);
        }
        cls.push(row);
      }
      const wc = Math.floor(WELL[0] / cell), wr = Math.floor((H - WELL[1]) / cell);
      if (wr >= 0 && wr < rows && wc >= 0 && wc < N) cls[wr][wc] = 3;
      key = '';
    };
    build();
    q('#n').addEventListener('input', build);
    AI.$$('[data-m]', root).forEach((b) => b.addEventListener('click', () => { mode = b.dataset.m; key = ''; }));
    c.cv.addEventListener('pointerdown', (e) => {
      const r = c.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      const sc = Math.min((c.w - 20) / W0, (c.h - 20) / H), ox = (c.w - W0 * sc) / 2, oy = (c.h - H * sc) / 2;
      const cc = Math.floor((px - ox) / (cell * sc)), rr = Math.floor((py - oy) / (cell * sc));
      if (cc >= 0 && cc < N && rr >= 0 && rr < rows) { sel = [rr, cc]; key = ''; }
    });
    const lakeV = G.shoelace(LAKE), roadV = polyLen(ROAD);
    AI.loop(root, (dt) => {
      const ta = mode === 'raster' ? 0 : 1, tr = mode === 'vector' ? 0 : mode === 'raster' ? 1 : 0.55;
      va += (ta - va) * (1 - Math.exp(-dt * 4));
      ra += (tr - ra) * (1 - Math.exp(-dt * 4));
      if (Math.abs(va - ta) < 0.005) va = ta;
      if (Math.abs(ra - tr) < 0.005) ra = tr;
      const k = [mode, N, va.toFixed(3), ra.toFixed(3), sel && sel.join(','), c.w, c.h, document.documentElement.dataset.theme].join('|');
      if (k === key) return;
      key = k;
      const C = AI.col(), g = c.ctx, Wc = c.w, Hc = c.h;
      const sc = Math.min((Wc - 20) / W0, (Hc - 20) / H), ox = (Wc - W0 * sc) / 2, oy = (Hc - H * sc) / 2;
      const X = (x) => ox + x * sc, Y = (y) => oy + (H - y) * sc;
      g.clearRect(0, 0, Wc, Hc);
      if (ra > 0.01) {
        g.globalAlpha = ra;
        const colr = [C.bg2, C.blue, C.orange, C.red];
        for (let r = 0; r < rows; r++) for (let cc = 0; cc < N; cc++) {
          g.fillStyle = colr[cls[r][cc]];
          g.fillRect(ox + cc * cell * sc, oy + r * cell * sc, cell * sc + 0.5, cell * sc + 0.5);
        }
        g.strokeStyle = C.border; g.lineWidth = 1; g.globalAlpha = ra * 0.7;
        g.beginPath();
        for (let cc = 0; cc <= N; cc++) { g.moveTo(ox + cc * cell * sc, oy); g.lineTo(ox + cc * cell * sc, oy + rows * cell * sc); }
        for (let r = 0; r <= rows; r++) { g.moveTo(ox, oy + r * cell * sc); g.lineTo(ox + N * cell * sc, oy + r * cell * sc); }
        g.stroke();
        g.globalAlpha = 1;
      }
      if (va > 0.01) {
        g.globalAlpha = va;
        g.beginPath(); LAKE.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)))); g.closePath();
        g.fillStyle = C.blue; g.globalAlpha = va * 0.35; g.fill(); g.globalAlpha = va; g.strokeStyle = C.blue; g.lineWidth = 2; g.stroke();
        g.beginPath(); ROAD.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))));
        g.strokeStyle = C.orange; g.lineWidth = 2.2 * sc; g.lineCap = 'round'; g.lineJoin = 'round'; g.stroke();
        g.fillStyle = C.red; g.beginPath(); g.arc(X(WELL[0]), Y(WELL[1]), 5, 0, 7); g.fill();
        g.globalAlpha = 1;
      }
      if (sel) {
        g.strokeStyle = C.fg; g.lineWidth = 2;
        g.strokeRect(ox + sel[1] * cell * sc + 1, oy + sel[0] * cell * sc + 1, cell * sc - 2, cell * sc - 2);
      }
      let lakeR = 0, roadR = 0;
      for (const row of cls) for (const v of row) { if (v === 1) lakeR++; if (v === 2) roadR++; }
      lakeR *= cell * cell;
      let txt = `lake: vector area = ${F(lakeV, 1)} units²,  raster area = ${F(lakeR, 1)} units² (${lakeR >= lakeV ? '+' : ''}${F(100 * (lakeR - lakeV) / lakeV, 1)} %)\n`
        + `road: vector length = ${F(roadV, 1)} units (one-cell-wide raster band: ${roadR} cells)\n`
        + `grid ${N} × ${rows}, cell = ${F(cell, 2)} units. geotransform: x₀ = 0, a = ${F(cell, 2)}, b = 0, y₀ = ${F(H, 2)}, d = 0, e = −${F(cell, 2)}\n`;
      txt += sel ? `selected: row ${sel[0]}, col ${sel[1]} → centre x = ${F(cell * (sel[1] + 0.5), 2)}, y = ${F(H - cell * (sel[0] + 0.5), 2)}; class = <b class="o">${CL[cls[sel[0]][sel[1]]]}</b>` : '<span class="m">click a cell</span>';
      q('#o').innerHTML = txt;
    });
  };

  /* ==================================================================== */
  /* 5. geobands: a band stack, one pixel's spectrum, an editable index     */
  /* ==================================================================== */
  AI.labs.geobands = function (root) {
    const d = root.dataset;
    const CLS = { V: 'vegetation', W: 'water', S: 'bare soil', U: 'urban' };
    const REF = { V: [0.04, 0.07, 0.04, 0.40], W: [0.06, 0.05, 0.03, 0.02], S: [0.12, 0.15, 0.2, 0.28], U: [0.14, 0.15, 0.17, 0.22] };
    const MAP = ['VVVWWU', 'VVVWWU', 'SSVVWU', 'SSSVUU', 'SSSSUU'];
    const ROWS = 5, COLS = 6, PIX = ROWS * COLS;
    const BANDS = ['B2', 'B3', 'B4', 'B8'], WL = [490, 560, 665, 842], TINT = ['#3b82f6', '#22c55e', '#ef4444', '#a855f7'];
    const { view, ctl } = AI.labShell(root, d.title || 'Band stack and spectrum of one pixel', 'each band is a layer; a pixel is a column through the stack',
      '<canvas style="height:250px"></canvas><canvas style="height:150px"></canvas><canvas style="height:120px"></canvas>',
      `<div class="row"><button class="btn primary" id="play">⏸ Pause tour</button></div>
       <label>Tour speed <span class="v" id="sv"></span><input type="range" id="speed" min="0.25" max="2.5" step="0.05" value="1"></label>
       <label>Index formula (B2, B3, B4, B8 are reflectances) <input type="text" id="ix" value="${d.index || '(B8-B4)/(B8+B4)'}" spellcheck="false"></label>
       <div class="err" id="er"></div>
       <div class="readout" id="o"></div>
       <div class="caption">Blue = B2, green = B3, red = B4, violet = B8. Click any pixel to inspect it; the spectrum fills in one band at a time.</div>`);
    const q = (s) => AI.$(s, root), cvs = AI.$$('canvas', view);
    const cube = AI.canvas(cvs[0]), spec = AI.canvas(cvs[1]), idx = AI.canvas(cvs[2]);
    const val = (b, r, cc) => REF[MAP[r][cc]][b];
    let fn = null, sel = 0, tour = 0, playing = true, reveal = 0, breathe = 0, ex = 1, key = '';
    AI.exprInput(q('#ix'), q('#er'), { allowed: BANDS, onChange: (f) => { fn = f; key = ''; } }).run();
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause tour' : '▶ Resume tour'; };
    const geom = (W, H, e) => {
      const cp = Math.min((W * 0.45) / (COLS + 3 * 1.6 * e), (H * 0.7) / (ROWS + 3 * 1.1 * e));
      return { cp, dx: cp * 1.6 * e, dy: cp * 1.1 * e, x0: 8, y0: H - 10 - ROWS * cp };
    };
    cube.cv.addEventListener('pointerdown', (e) => {
      const r = cube.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      const gm = geom(cube.w, cube.h, ex);
      const cc = Math.floor((px - gm.x0) / gm.cp), rr = Math.floor((py - gm.y0) / gm.cp);
      if (cc >= 0 && cc < COLS && rr >= 0 && rr < ROWS) { sel = rr * COLS + cc; tour = sel; reveal = 0; key = ''; }
    });
    idx.cv.addEventListener('pointerdown', (e) => {
      const r = idx.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      const cp = Math.min(idx.w / COLS, idx.h / ROWS), ox = (idx.w - COLS * cp) / 2, oy = (idx.h - ROWS * cp) / 2;
      const cc = Math.floor((px - ox) / cp), rr = Math.floor((py - oy) / cp);
      if (cc >= 0 && cc < COLS && rr >= 0 && rr < ROWS) { sel = rr * COLS + cc; tour = sel; reveal = 0; key = ''; }
    });
    AI.loop(root, (dt) => {
      const speed = +q('#speed').value;
      q('#sv').textContent = `${F(speed, 2)}×`;
      if (playing) { tour += dt * speed * 0.4; sel = Math.floor(tour) % PIX; }
      reveal = Math.min(reveal + dt, 6);
      breathe += dt * (playing ? 1 : 0.2);
      ex = 1 + 0.12 * Math.sin((breathe * 2 * Math.PI) / 8);
      const r0 = Math.floor(sel / COLS), c0 = sel % COLS;
      const sv = BANDS.map((_, b) => val(b, r0, c0));
      const fv = fn ? fn({ B2: sv[0], B3: sv[1], B4: sv[2], B8: sv[3] }) : NaN;
      const k = [sel, Math.round(reveal * 20), ex.toFixed(3), cube.w, cube.h, spec.h, document.documentElement.dataset.theme].join('|');
      if (k === key) return;
      key = k;
      const C = AI.col();
      /* cube */
      const g = cube.ctx, W = cube.w, H = cube.h, gm = geom(W, H, ex);
      g.clearRect(0, 0, W, H);
      for (let b = 3; b >= 0; b--) {
        const ox = gm.x0 + b * gm.dx, oy = gm.y0 - b * gm.dy;
        const [tr, tg, tb] = AI.rgb(TINT[b]);
        for (let r = 0; r < ROWS; r++) for (let cc = 0; cc < COLS; cc++) {
          const I = Math.min(1, val(b, r, cc) / 0.4);
          g.fillStyle = `rgb(${Math.round(12 + tr * I)},${Math.round(12 + tg * I)},${Math.round(12 + tb * I)})`;
          g.fillRect(ox + cc * gm.cp, oy + r * gm.cp, gm.cp - 1, gm.cp - 1);
        }
        g.strokeStyle = TINT[b]; g.lineWidth = 1.5; g.strokeRect(ox - 1, oy - 1, COLS * gm.cp + 2, ROWS * gm.cp + 2);
        g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left';
        g.fillText(`${BANDS[b]} · ${WL[b]} nm`, gm.x0 + 3 * gm.dx + COLS * gm.cp + 8, oy + 12);
      }
      for (let b = 0; b < 4; b++) {
        const ox = gm.x0 + b * gm.dx, oy = gm.y0 - b * gm.dy;
        g.strokeStyle = C.orange; g.lineWidth = 2; g.strokeRect(ox + c0 * gm.cp, oy + r0 * gm.cp, gm.cp - 1, gm.cp - 1);
      }
      g.setLineDash([3, 3]); g.strokeStyle = C.purple; g.lineWidth = 1.5; g.beginPath();
      for (let b = 0; b < 4; b++) {
        const px = gm.x0 + b * gm.dx + (c0 + 0.5) * gm.cp, py = gm.y0 - b * gm.dy + (r0 + 0.5) * gm.cp;
        b ? g.lineTo(px, py) : g.moveTo(px, py);
      }
      g.stroke(); g.setLineDash([]);
      /* spectrum */
      const s2 = spec.ctx, v2 = AI.view(spec, 440, 880, 0, 0.5, { l: 40, r: 12, t: 14, b: 26 });
      s2.clearRect(0, 0, spec.w, spec.h);
      AI.axes(s2, v2);
      s2.fillStyle = C.muted; s2.font = '11px sans-serif'; s2.textAlign = 'left'; s2.fillText('reflectance vs wavelength (nm): one pixel', v2.pad.l + 4, 11);
      s2.strokeStyle = C.orange; s2.lineWidth = 2.5; s2.beginPath();
      let drawn = 0;
      for (let b = 0; b < 4; b++) {
        if (reveal < b * 0.9) break;
        const x = v2.X(WL[b]), y = v2.Y(sv[b]);
        b ? s2.lineTo(x, y) : s2.moveTo(x, y);
        drawn = b + 1;
      }
      if (drawn > 1) s2.stroke();
      for (let b = 0; b < drawn; b++) { s2.fillStyle = TINT[b]; s2.beginPath(); s2.arc(v2.X(WL[b]), v2.Y(sv[b]), 5, 0, 7); s2.fill(); }
      /* index map */
      const i2 = idx.ctx, cp = Math.min(idx.w / COLS, idx.h / ROWS), ox2 = (idx.w - COLS * cp) / 2, oy2 = (idx.h - ROWS * cp) / 2;
      i2.clearRect(0, 0, idx.w, idx.h);
      for (let r = 0; r < ROWS; r++) for (let cc = 0; cc < COLS; cc++) {
        const v = fn ? fn({ B2: val(0, r, cc), B3: val(1, r, cc), B4: val(2, r, cc), B8: val(3, r, cc) }) : NaN;
        i2.fillStyle = Number.isFinite(v) ? AI.mix(C.orange, C.green, AI.clamp((v + 1) / 2, 0, 1)) : C.muted;
        i2.fillRect(ox2 + cc * cp, oy2 + r * cp, cp - 1, cp - 1);
      }
      i2.strokeStyle = C.fg; i2.lineWidth = 2; i2.strokeRect(ox2 + c0 * cp + 1, oy2 + r0 * cp + 1, cp - 2, cp - 2);
      i2.fillStyle = C.muted; i2.font = '11px sans-serif'; i2.textAlign = 'left';
      i2.fillText('index map (orange = low, green = high)', 4, 12);
      q('#o').innerHTML = `pixel row ${r0}, col ${c0}: class <b>${CLS[MAP[r0][c0]]}</b>\n`
        + BANDS.map((b, i) => `${b} = <b>${F(sv[i], 3)}</b>`).join('   ')
        + `\nindex = <b class="o">${F(fv, 3)}</b>   <span class="m">(${q('#ix').value})</span>`;
    });
  };

  /* ==================================================================== */
  /* 6. geosensors: footprint on the ground, revisit and cloud chance      */
  /* ==================================================================== */
  AI.labs.geosensors = function (root) {
    const d = root.dataset;
    const PRESET = {
      s2: { name: 'Sentinel-2-like (optical)', gsd: 10, rev: 5, bands: 13, bytes: 2, radar: false, cloud: 0.5 },
      l8: { name: 'Landsat-like (optical)', gsd: 30, rev: 8, bands: 11, bytes: 2, radar: false, cloud: 0.5 },
      vhr: { name: 'Very high resolution (illustrative)', gsd: 0.5, rev: 2, bands: 4, bytes: 2, radar: false, cloud: 0.5 },
      sar: { name: 'Radar-like (illustrative, cloud-blind)', gsd: 10, rev: 6, bands: 2, bytes: 4, radar: true, cloud: 0.5 },
    };
    const { view, ctl } = AI.labShell(root, d.title || 'Sensor comparison', 'footprint on the ground, revisit and clouds',
      '<canvas style="height:300px"></canvas><canvas style="height:90px"></canvas><div class="readout" id="o" style="margin-top:10px"></div>',
      `<label>Preset <select id="pre">${Object.keys(PRESET).map((k) => `<option value="${k}"${k === (d.preset || 's2') ? ' selected' : ''}>${PRESET[k].name}</option>`).join('')}</select></label>
       <label>Ground sample distance (m) <span class="v" id="gv"></span><input type="range" id="gsd" min="0.5" max="60" step="0.5" value="10"></label>
       <label>Revisit (days) <span class="v" id="rv"></span><input type="range" id="rev" min="1" max="30" step="1" value="5"></label>
       <label>Bands <span class="v" id="bv"></span><input type="range" id="bands" min="1" max="13" step="1" value="13"></label>
       <label>Chance that one look is cloudy <span class="v" id="cv"></span><input type="range" id="cloud" min="0" max="1" step="0.05" value="0.5"></label>
       <label>Scene side (km) <span class="v" id="av"></span><input type="range" id="aoi" min="1" max="20" step="1" value="10"></label>
       <label>Sweep speed <span class="v" id="sp"></span><input type="range" id="speed" min="0.25" max="2.5" step="0.05" value="1"></label>
       <div class="row"><button class="btn primary" id="pause">⏸ Pause</button></div>
       <div class="caption">The house is 20 m × 12 m and the road is 10 m wide, on a 240 m × 240 m patch. Each pixel's colour is the mix of what covers it. The white square is the sensor footprint.</div>`);
    const q = (s) => AI.$(s, root), cvs = AI.$$('canvas', view), ground = AI.canvas(cvs[0]), tl = AI.canvas(cvs[1]);
    const P = 240, HOUSE = [110, 130, 90, 102], ROAD = [0, 240, 150, 160];
    const U = Array.from({ length: 200 }, AI.rng(2026));
    const off = document.createElement('canvas');
    let gsdKey = null, ncell = 0, cover = null, sweep = 0, tlT = 0, playing = true, key = '';
    const sample = (x, y) => (x >= HOUSE[0] && x <= HOUSE[1] && y >= HOUSE[2] && y <= HOUSE[3]) ? 'h' : (x >= ROAD[0] && x <= ROAD[1] && y >= ROAD[2] && y <= ROAD[3]) ? 'r' : '';
    const buildCover = (gsd) => {
      ncell = Math.ceil(P / gsd);
      cover = new Float32Array(ncell * ncell * 2);
      for (let j = 0; j < ncell; j++) for (let i = 0; i < ncell; i++) {
        let h = 0, r = 0;
        for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
          const x = Math.min(P, (i + (a + 0.5) / 2) * gsd), y = Math.min(P, (j + (b + 0.5) / 2) * gsd);
          const t = sample(x, y);
          if (t === 'h') h++; else if (t === 'r') r++;
        }
        const o = (j * ncell + i) * 2;
        cover[o] = h / 4; cover[o + 1] = r / 4;
      }
      off.width = ncell; off.height = ncell;
      const ctx = off.getContext('2d'), img = ctx.createImageData(ncell, ncell);
      for (let j = 0; j < ncell; j++) for (let i = 0; i < ncell; i++) {
        const o = (j * ncell + i) * 2, hf = cover[o], rf = cover[o + 1], gf = Math.max(0, 1 - hf - rf);
        const p = ((ncell - 1 - j) * ncell + i) * 4;   // image row 0 is the top (north)
        img.data[p] = 219 * hf + 139 * rf + 58 * gf;
        img.data[p + 1] = 109 * hf + 148 * rf + 110 * gf;
        img.data[p + 2] = 40 * hf + 158 * rf + 62 * gf;
        img.data[p + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      gsdKey = gsd;
    };
    q('#pre').addEventListener('change', () => {
      const p = PRESET[q('#pre').value];
      q('#gsd').value = p.gsd; q('#rev').value = p.rev; q('#bands').value = p.bands; q('#cloud').value = p.cloud;
      key = '';
    });
    q('#pause').onclick = () => { playing = !playing; q('#pause').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    AI.loop(root, (dt) => {
      const gsd = +q('#gsd').value, rev = +q('#rev').value, bands = +q('#bands').value, cloud = +q('#cloud').value, aoi = +q('#aoi').value;
      const speed = +q('#speed').value, pre = PRESET[q('#pre').value];
      q('#gv').textContent = `${F(gsd, 1)} m`; q('#rv').textContent = `${rev} d`; q('#bv').textContent = bands;
      q('#cv').textContent = `${F(cloud * 100)} %`; q('#av').textContent = `${aoi} km`; q('#sp').textContent = `${F(speed, 2)}×`;
      if (playing) { sweep += dt * speed * Math.max(4, ncell / 6); tlT += dt * speed * 15; if (tlT > 110) tlT = 0; }
      if (gsdKey !== gsd) buildCover(gsd);
      const pc = pre.radar ? 0 : cloud;
      const n = Math.floor(90 / rev);
      const k = [gsd, Math.floor(sweep), Math.floor(tlT), rev, bands, cloud, aoi, ground.w, ground.h, tl.w, document.documentElement.dataset.theme].join('|');
      if (k === key) return;
      key = k;
      const C = AI.col(), g = ground.ctx, W = ground.w, H = ground.h;
      const ps = (Math.min(W, H) * 0.95) / P, ox = (W - P * ps) / 2, oy = (H + P * ps) / 2;
      const X = (m) => ox + m * ps, Y = (m) => oy - m * ps;
      g.clearRect(0, 0, W, H);
      g.imageSmoothingEnabled = false;
      g.drawImage(off, X(0), Y(ncell * gsd), ncell * gsd * ps, ncell * gsd * ps);
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(X(0), Y(P), P * ps, P * ps);
      g.fillStyle = C.fg; g.font = '11px sans-serif'; g.textAlign = 'left';
      g.fillText('house 20 m', X(HOUSE[0]) + 2, Y(HOUSE[3]) - 4);
      g.fillText('road 10 m', X(4), Y(ROAD[3]) - 4);
      g.fillText('240 m', X(P) - 40, Y(0) - 6);
      const cellsPerRow = ncell, pos = sweep % (cellsPerRow * cellsPerRow);
      const ci = Math.floor(pos) % cellsPerRow, rj = Math.floor(Math.floor(pos) / cellsPerRow);
      const fy = (cellsPerRow - 1 - rj) * gsd, fx = ci * gsd;
      const fw = Math.max(3, gsd * ps), fh = Math.max(3, gsd * ps);
      g.fillStyle = 'rgba(255,255,255,0.25)'; g.fillRect(X(fx), Y(fy + gsd), fw, fh);
      g.strokeStyle = '#ffffff'; g.lineWidth = 2; g.strokeRect(X(fx), Y(fy + gsd), fw, fh);
      const pixels = (aoi * 1000 / gsd) ** 2, bytes = pixels * bands * pre.bytes;
      const looks = n;
      const pClear = looks ? 1 - Math.pow(pc, looks) : 0;
      const size = bytes >= 1e9 ? `${F(bytes / 1e9, 2)} GB` : `${F(bytes / 1e6, 1)} MB`;
      q('#o').innerHTML = `one pixel covers <b>${F(gsd * gsd, 2)} m²</b>;  house width = <b class="o">${F(20 / gsd, 2)}</b> pixels, road width = <b class="o">${F(10 / gsd, 2)}</b> pixels\n`
        + `scene ${aoi} km × ${aoi} km: ${F(aoi * 1000 / gsd)} × ${F(aoi * 1000 / gsd)} = <b>${F(pixels)}</b> pixels per band; ${bands} band(s) × ${pre.bytes} bytes ⇒ <b class="b">${size}</b> uncompressed\n`
        + `revisit ${rev} d ⇒ <b>${n}</b> looks in 90 days; chance of at least one clear look = <b class="g">${F(100 * pClear, 2)} %</b>`
        + (pre.radar ? ` <span class="m">(radar: cloud does not block it)</span>` : ` <span class="m">(independent looks)</span>`);
      /* timeline */
      const s2 = tl.ctx, TW = tl.w, TH = tl.h, tx0 = 30, tx1 = TW - 10, dayX = (dd) => tx0 + (dd / 90) * (tx1 - tx0);
      s2.clearRect(0, 0, TW, TH);
      s2.strokeStyle = C.border; s2.lineWidth = 1; s2.beginPath(); s2.moveTo(tx0, TH - 26); s2.lineTo(tx1, TH - 26); s2.stroke();
      s2.fillStyle = C.muted; s2.font = '11px sans-serif'; s2.textAlign = 'center';
      for (let dd = 0; dd <= 90; dd += 30) s2.fillText(`day ${dd}`, dayX(dd), TH - 10);
      s2.fillText('one tick per look: green clear, grey cloudy', TW / 2, 14);
      const shownDay = Math.min(90, tlT);
      for (let i = 1; i <= n; i++) {
        const dd = i * rev;
        if (dd > shownDay) break;
        const clearLook = U[i] >= pc;
        s2.fillStyle = clearLook ? C.green : C.muted;
        s2.beginPath(); s2.arc(dayX(dd), TH - 40, 5, 0, 7); s2.fill();
      }
      s2.strokeStyle = C.orange; s2.beginPath(); s2.moveTo(dayX(shownDay), 22); s2.lineTo(dayX(shownDay), TH - 26); s2.stroke();
    });
  };
})();
