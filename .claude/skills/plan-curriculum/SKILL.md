---
name: plan-curriculum
description: Design a complete zero-to-hero curriculum for an AI Hub topic (modules, ordered lessons, what must be understood, AI links, lab ideas) and generate the site data file. Use before writing any lesson content for a new or deepened topic.
---

# Plan a curriculum (curriculum-first)

This is how the Mathematics curriculum (9 modules, 58 lessons) was produced. Do this **before** writing lessons.

## Steps
1. **Define the end state.** One sentence: what can a graduate of this topic do? (Math: read an ML paper, derive backprop, understand transformers/VAEs/diffusion.)
2. **List the prerequisite chain**, then cut it into modules ordered strictly so every lesson only uses earlier ideas. Start with a
   "Foundations" module if the topic has any vocabulary/notation to learn. Aim for 5–10 modules, 3–11 lessons each.
3. **For each lesson decide**: `id` (kebab-case, unique), title, *Must understand* (concrete skills/concepts, semicolon separated),
   *AI link* (named algorithms/models where it appears — never vague), and a lab idea (what the learner will change and watch).
4. **Check coverage against the real field**: compare with a standard syllabus/textbooks; ensure nothing needed for the end state is
   missing and nothing is included that is never used. Order must pay off: each module should unlock the next.
5. **Write `curriculum/<topic>.md`** using the exact table format of `curriculum/mathematics.md`:
   `## Module N · Title  (\`page-file.html\`)` then a table with header `| id | Lesson | Must understand | AI link |`.
   Keep the "Lesson template" and "Content standards" sections.
6. **Generate the data file.** `tools/gen-curriculum.js` currently handles mathematics only; when adding a second topic, generalise it
   (take topic slug as argument, emit `assets/curriculum-<slug>.js` exposing `window.<SLUG>_MODULES`) and mirror the math wiring in
   `assets/shell.js` (topic.modules → sections, sidebar tree, module cards, prev/next). Run it and check lesson counts.
7. **Hub page** `<topic>.html`: hero, "how to learn from this", the map (path bar + animated pipeline), module cards (`#mod-list`, `#pathbar`).
8. Report the module/lesson counts and any scope decisions to the user before authoring lessons.

## Quality bar for the curriculum itself
- Every lesson has a *why it matters for AI* and a lab concept; no pure-trivia lessons.
- Terminology introduced once, in the earliest lesson that needs it.
- Modules are independent enough to be authored in parallel (one page + one labs file each).
