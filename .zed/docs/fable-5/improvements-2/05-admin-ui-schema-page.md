# 05 — Admin UI: Jev Schema Page

**Priority:** P1 · **Depends:** 02 (03 to observe effect) · **Status:** Implemented · **Phase:** J4

## Summary

Add a **Jev** page to the zero-JS admin (`:8081`) where schemas are listed, edited, validated, deleted, and
launched. Everything is plain HTML forms with server-side validation, POST-redirect-GET, and the existing
`SecurityMiddleware` (Host + Origin) and CSP (`script-src 'none'`). No JavaScript, no CDN, no new dependency.

## Pages and routes (`admin/routes/jev.py`, `APIRouter`, registered in `build_app`)

| Route | Method | Behaviour |
| --- | --- | --- |
| `/jev` | GET | Active-Jev panel (name, source, sha12, applied-at, **stale** badge, **Detach** button) + schema table (name, source badge `built-in`/`user`/`overrides built-in`, sha12, size, active marker, actions) + **Launch** form (model `<select>` from `ListModels`, schema `<select>`, "force reload" checkbox) |
| `/jev/new` | GET | Empty editor (name input + textarea pre-filled with a minimal template) |
| `/jev/{name}` | GET | Editor for one schema. Built-in only: read-only banner + **"Create editable copy"** (opens the editor with `name` pre-filled, content = built-in) |
| `/jev/{name}` | POST | `action=save` · `action=validate` · `action=save_launch` (with `alias`, `force`) · `action=launch_unsaved` |
| `/jev/{name}/delete` | GET | Confirmation page (like `cache_rm_confirm.html`); for an override: "Reverts to the built-in" |
| `/jev/{name}/delete` | POST | Delete, flash, 303 → `/jev` |
| `/jev/launch` | POST | Launch an existing schema on a model (`alias`, `schema_name`, `force`) → same job redirect as `/models/load` |
| `/jev/detach` | POST | `LoadModel(active_alias, JEV_OFF)` → 303 `/jobs/<id>` |
| `/jev/build` | POST | **Quick builder** (P2): turns simple fields into a schema text and re-renders the editor (does not save) |

All POST handlers `303` to a GET (flash on success). A **validation failure** re-renders the editor with HTTP 200,
the user's exact text preserved in the `<textarea>`, and an error list — never a redirect that would lose input.

## Requirements

| ID | Requirement | Sev |
| --- | --- | --- |
| JEV-05.1 | Nav gets a **Jev** link (`page == 'jev'`) in `base.html`; `path_for('jev')`. | P1 |
| JEV-05.2 | List page via `ListJevSchemas` + `GetStatus` + `ListModels` under `safe_grpc` (3 s deadline, "controller unreachable" banner on failure, same degradation as `/models`). Reuse `client_of`, `render`, `redirect_with_flash`. | P1 |
| JEV-05.3 | Editor: `<textarea name="schema_json" rows="28" spellcheck="false" wrap="off">`, monospace via **existing vendored CSS** (add ≤ 10 lines to `vibe.css`; still no `@import`/`url()`), a name `<input>` (pattern attr mirrors the server regex), and distinct submit buttons: **Validate**, **Save**, **Save & launch**, **Launch without saving**. Buttons use `<button name="action" value="…">`. | P1 |
| JEV-05.4 | `action=validate` → `ValidateJevSchema`; shows ✔ "valid (sha 12, N bytes)" or an error list, plus warnings, **without** saving. | P1 |
| JEV-05.5 | `action=save` → `PutJevSchema`; on `ok` → flash "saved" + 303 `/jev/{name}`; on errors → re-render. Saving the currently active schema's name shows "Active Jev is now **stale** — re-launch to apply" (never silently changes live behaviour). | P1 |
| JEV-05.6 | `action=save_launch` → `Put` then `LoadModel(alias, JEV_ON, JevSpec(schema_name=name))` via the existing `start_background_job`; redirect to `/jobs/<id>`. `launch_unsaved` sends `JevSpec(schema_json=text)` (optionally `save_as`). Errors from the first event (`jev_schema_invalid`, `jev_schema_not_found`) are flashed/re-rendered, not 500s. | P1 |
| JEV-05.7 | **Body cap**: reject admin POST bodies > 128 KiB for `/jev*` routes with 413 (HTML) **before** `parse_qsl` (urlencoding inflates up to 3×; 64 KiB schema + fields). `parse_form` itself stays unchanged for other routes. | P1 |
| JEV-05.8 | Output escaping: schema text is rendered **only** inside `<textarea>` (`autoescape` on; never `|safe`); error strings from `jsonschema` are escaped. Test with `</textarea><script>` payloads. CSP unchanged. | P1 |
| JEV-05.9 | Name handling: path param validated with the store regex; invalid → 404 page; `new` flow validates the name field and errors inline. Names cannot collide with the reserved route words (`new`, `launch`, `detach`, `build`) — reject at save, reserved words listed in doc 01's store rules. | P1 |
| JEV-05.10 | CSRF: no token (consistent with v1 admin, loopback, `SecurityMiddleware` Origin/Sec-Fetch-Site checks). Add tests that a cross-origin `POST /jev/{name}` and `POST /jev/launch` get 403. | P1 |
| JEV-05.11 | Launch form guards: disabled when controller down; "force reload" explained (restarts container, 30–120 s); selecting a model with `tags` containing `reasoning` shows a static note (rendered server-side from `ModelInfo.tags`). Without JS the note shows for any reasoning model in the select via a `<small>` list below the form. | P2 |
| JEV-05.12 | Dashboard (`/`) shows a **Jev: <name> (sha12)** badge next to the active model when `Status.HasField("jev")`; `/models` table gets a small `Jev` marker column for the active row. No new controls there. | P2 |
| JEV-05.13 | **Quick builder** `/jev/build` (P2): fields — decision types (checkboxes: choice / boolean / score), choice options (one per line → `enum`), score min/max, include confidence (checked), `additionalProperties: false` (checked). Server generates the discriminated `oneOf` text (same generator that produced the built-in default, kept in `vibe.jev.builder` and unit-tested against it), then renders the editor with that text so the user can tweak and save. Pure server-side; no JS. | P2 |
| JEV-05.14 | After launch the job page (`/jobs/<id>`, meta-refresh) shows Jev events unchanged (they're ordinary `JobEvent`s; `error_code` rendered as a badge). Minor `job.html` tweak. | P2 |
| JEV-05.15 | Accessibility: labels bound to inputs, errors in a `role="alert"` region, warnings in `role="status"`, buttons have text (no icon-only), table has `<caption>`. | P2 |
| JEV-05.16 | `HEAD` support is inherited from `SecurityMiddleware`; add one test (`HEAD /jev` → 200, empty body). | P2 |

## Design notes

- **Why a textarea, not a form-per-keyword editor?** JSON Schema is open-ended; a faithful keyword form is a
  product in itself. Textarea + validation + the quick builder covers both power users and the common
  "choose YES/NO/…" case without JS.
- **Preserving input on error** is the single most important UX detail: re-render, don't redirect.
- **Parsing**: forms are `application/x-www-form-urlencoded`; `parse_qsl` handles newlines in the textarea
  (`%0A`). Browsers normalise line endings to CRLF — normalise to `\n` server-side before hashing/validating (the
  canonical hash is whitespace-insensitive anyway, but stored text should be tidy). Store the **pretty-printed**
  canonical text (`indent=2`) on `put` from the UI? Decision: store what the user typed after CRLF→LF; do not
  reformat (preserves comments-in-strings, ordering). The CLI `show` pretty-prints on display only.
- **Concurrency**: two operators editing the same schema is last-write-wins; show `mtime` and the sha in the form
  as hidden fields, and refuse the save if the on-disk sha differs from the hidden `base_sha` (re-render with
  "changed since you opened it" + current text) — cheap optimistic concurrency, P2 (JEV-05.17).
- **Where launch goes**: always `LoadModel` (XR-4, XR-2). The admin never starts containers or writes the store
  itself.

## Task breakdown

- [x] `admin/routes/jev.py` + register router in `admin/app.py`; `path_for` names.
- [x] Templates: `jev.html` (list + launch + active panel), `jev_edit.html`, `jev_delete_confirm.html`; nav link.
- [x] `vibe.css` additions (textarea, badges) — keep file free of `@import`/`url()`.
- [x] Body-cap helper for `/jev*` POSTs.
- [x] Quick builder + `/jev/build` (P2) — implemented as `admin/jev_builder.py` (same output, tested as E11);
      moving it to `vibe.jev.builder` is a non-blocking cleanup.
- [x] Dashboard/models/job template tweaks.
- [x] Optimistic-concurrency hidden field (P2, JEV-05.17).
- [x] Tests (doc 06 §D): route happy/sad paths with a fake controller client, escaping, 413, 403 cross-origin,
      HEAD, controller-down degradation.

## Acceptance criteria

- With JS disabled (and CSP enforced) every flow works in a real browser: create schema → validate → save →
  launch → watch job → dashboard badge → detach.
- Pasting invalid JSON keeps the text in the textarea and lists the parse error with line/column.
- Editing `default` creates an override; deleting it shows the built-in again.
- A `</textarea><script>alert(1)</script>` schema body renders as inert text.
- Cross-origin POST is rejected 403; a > 128 KiB POST is rejected 413 without parsing.
- `GET /jev` with the controller down renders the "unreachable" banner and disabled launch form (no 500).

## Risks

| Risk | Mitigation |
| --- | --- |
| Large textarea posts hit `parse_qsl` limits | explicit cap (JEV-05.7); `parse_qsl` default `max_num_fields` is None — rely on the byte cap |
| Operators assume saving changes live behaviour | **stale** badge + explicit "Save & launch"; copy on the edit page |
| Browser autofill/spellcheck mangling JSON | `spellcheck="false" autocomplete="off" autocapitalize="off"` on the textarea |
| Reserved route words as schema names | reserved-name list in store validation (shared with doc 01 JEV-01.9) |
