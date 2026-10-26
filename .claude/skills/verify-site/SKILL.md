---
name: verify-site
description: Verify AI Hub pages before declaring work done - syntax check JS, load pages in headless Chrome to catch console errors, and review screenshots for rendering, overflow and animation problems.
---

# Verify the site

Run from the repo root (`E:\On Going Projects\Knowledge Growth\AI-Hub`). Use the scratchpad dir for screenshots.

```bash
# 1. syntax
for f in assets/*.js; do node --check "$f" || echo "BAD $f"; done

# 2. console errors (no output from grep = clean) + screenshot per page
CH="/c/Program Files/Google/Chrome/Application/chrome.exe"
for p in index mathematics math-foundations machine-learning; do
  "$CH" --headless=new --disable-gpu --enable-logging=stderr --v=0 --virtual-time-budget=9000 \
    --window-size=1500,6000 --screenshot="<scratchpad>/$p.png" \
    "file:///E:/On%20Going%20Projects/Knowledge%20Growth/AI-Hub/$p.html" 2>&1 | grep CONSOLE
done
```
(`ResizeObserver loop` warnings are harmless. KaTeX loads from a CDN, so network is needed for equations.)

3. **Read each screenshot** (Read tool; tall images are scaled — crop with a smaller `--window-size` and several `--screenshot` runs
   at different heights if detail is needed). Look for: raw `\frac`/`$` TeX, empty lab boxes, text overflow, broken layout at 1500px
   and at `--window-size=420,2400` (mobile), missing sidebar entries, wrong topic order.
4. **Maths check**: recompute worked examples and quiz answers in node/python when you touched content.
5. **Animation check**: confirm speeds are slow by default and nothing redraws needlessly (labs use `AI.ticker`, state-key skipping).
6. **Links/ids**: every lesson `id` matches `curriculum/<topic>.md`; sidebar/outline anchors resolve; module prev/next links exist.
7. After pushing, check the Actions run: `gh run list --limit 1` and `curl -s -o /dev/null -w "%{http_code}" https://abdelrahmankatkat.github.io/AI-Hub/` (expect 200).

Report failures with the exact console text; fix before saying done.
