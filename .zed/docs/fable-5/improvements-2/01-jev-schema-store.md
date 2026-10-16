# 01 — Jev Schema Model, Default, Validation, and Store

**Priority:** P1 · **Depends:** — · **Status:** Implemented · **Phase:** J0

## Summary

Introduce a torch-free, pure-Python package `vibe.jev` that owns everything about a Jev *schema as data*: the
built-in `default` schema, a validator (the single gatekeeper used by CLI, controller and admin), content hashing,
and an on-disk store for user schemas. No Docker, gRPC, or HTTP in this package — it is unit-testable in isolation.

## Goals / Non-goals

**Goals**
- A built-in default schema whose `choice` branch is `["YES", "NO"]`.
- One validator with typed, user-readable errors and non-fatal warnings.
- A store with atomic writes, name validation, and built-in/user shadowing.
- Stable canonical hash so "same schema" is decidable across restarts and clients.

**Non-goals**
- Talking to vLLM (doc 02 does the probe).
- Editing UI (doc 05) or CLI (doc 04).
- Supporting every JSON Schema feature — we support what vLLM's backend can constrain (see warnings).

## The example schema as given — what to fix

The user's `JevDecisionOutput` is the intended *shape*; the following issues should be corrected in the built-in:

| # | Issue in the example | Fix in the built-in default |
| --- | --- | --- |
| 1 | `"$schema": "http://json-schema.org"` is not a dialect URI | `https://json-schema.org/draft/2020-12/schema` |
| 2 | `anyOf` on `result` lets `{decision_type:"score", result:true}` validate — nothing couples the two fields | Top-level `oneOf` of three branches, each with `decision_type: {const: …}` and the matching `result` type. Avoid `if/then/else` (poor support in constrained-decoding backends). |
| 3 | `option_a/b/c` are placeholders | `["YES","NO"]` in the default (see below); other sets are user schemas |
| 4 | `boolean` description says "boolean/null" but type is only `boolean` | Description says "boolean"; `null` deliberately not allowed (grammar-hostile, ambiguous) |
| 5 | `additionalProperties: false` | Keep (it makes constrained decoding tighter) — set on **every** branch |
| 6 | Branch keywords repeated inside `anyOf` | Share `confidence` via `properties` in each branch; `required` repeated per branch (needed for `oneOf` strictness) |

### Built-in `default` (proposed content, `src/vibe/jev/builtin/default.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "JevDecisionOutput",
  "description": "Default Jev decision: a YES/NO choice, a boolean verdict, or a 0..1 score, with a confidence.",
  "oneOf": [
    {
      "type": "object",
      "properties": {
        "decision_type": { "const": "choice" },
        "result": { "type": "string", "enum": ["YES", "NO"], "description": "The chosen action." },
        "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
      },
      "required": ["decision_type", "result", "confidence"],
      "additionalProperties": false
    },
    {
      "type": "object",
      "properties": {
        "decision_type": { "const": "boolean" },
        "result": { "type": "boolean" },
        "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
      },
      "required": ["decision_type", "result", "confidence"],
      "additionalProperties": false
    },
    {
      "type": "object",
      "properties": {
        "decision_type": { "const": "score" },
        "result": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
      },
      "required": ["decision_type", "result", "confidence"],
      "additionalProperties": false
    }
  ]
}
```

> **Top-level `type: object` caveat.** OpenAI's strict mode and some vLLM backends want the root to be an
> object with `properties`; a root `oneOf` is accepted by xgrammar but not by OpenAI. Since this stack is vLLM-only
> that is fine, but the runbook (doc 06 C2) **must** confirm the root-`oneOf` form is accepted by the pinned
> `v0.11.2` backend. Fallback if not: keep the user's original single-object form (`decision_type` enum +
> `result` `anyOf`) as `default`, and move the discriminated form to a `strict` preset. The validator treats
> both shapes identically.

### Optional presets (P2, same directory, read-only)

| Name | Purpose |
| --- | --- |
| `yes-no` | Only the `choice` branch with `["YES","NO"]` — forces a binary answer; best for first real tests |
| `yes-no-unsure` | `choice` with `["YES","NO","UNSURE"]` |

Presets cost nothing (JSON files) and make the admin page useful before anyone writes a schema.

## Requirements

| ID | Requirement | Sev |
| --- | --- | --- |
| JEV-01.1 | `src/vibe/jev/__init__.py` exports `load_builtin(name)`, `list_builtin()`, `validate_schema_text(text) -> ValidationResult`, `canonical_hash(schema) -> str`, `SchemaStore`. No imports of `docker`, `grpc`, `httpx`, `fastapi`. | P1 |
| JEV-01.2 | Built-ins ship as package data (hatch `packages = ["src/vibe"]` already includes non-py files under the package — **verify** with `uv build` / wheel listing). `default.json` content as above. | P1 |
| JEV-01.3 | `ValidationResult(ok, errors: list[str], warnings: list[str], schema: dict | None, sha256: str, size_bytes: int)`. Never raises on user input. | P1 |
| JEV-01.4 | Hard errors: not valid JSON (message includes line/col); root not an object; fails `Draft202012Validator.check_schema` (unknown `$schema` dialects other than draft 2020-12/2019-09/07 rejected); `len(text.encode()) > MAX_SCHEMA_BYTES` (64 KiB); nesting depth > 16; any `$ref` not starting with `#` (no remote/file refs); `$ref` cycles that make `jsonschema` recurse; empty `enum`; `enum` of > 256 values. | P1 |
| JEV-01.5 | Warnings (non-fatal, shown in CLI/admin, logged): `additionalProperties` not `false` on an object; `if/then/else`, `not`, `patternProperties`, `unevaluated*`, `dependent*` present (poor/unsupported in constrained-decoding backends — list to be **confirmed against the pinned vLLM backend** in the runbook and kept in one constant `RISKY_KEYWORDS`); `$schema` missing; unbounded `string` with no `maxLength` ⟹ warn "may run to max_tokens"; no `required`. | P2 |
| JEV-01.6 | `canonical_hash(schema)` = SHA-256 of `json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`. Whitespace/key-order edits do not change the hash. Display as `sha256[:12]`. | P1 |
| JEV-01.7 | `wire_schema(schema) -> dict`: copy with `$schema` removed (some backends choke on it); everything else untouched. Used by gateway and probe. The stored form always keeps `$schema`. | P1 |
| JEV-01.8 | `schema_name_for_openai(schema, fallback)`: `title` sanitised to `^[A-Za-z0-9_-]{1,64}$`, else the store name, else `"jev"`. | P2 |
| JEV-01.9 | `SchemaStore(root: Path)`: `list() -> [SchemaEntry(name, source, sha256, size, mtime)]`, `get(name)`, `put(name, text)`, `delete(name)`. `source ∈ {"builtin", "user", "override"}` (override = user file shadowing a builtin of the same name). Names match `^[a-z0-9][a-z0-9._-]{0,63}$`; reject path separators and leading dots (so `.active.json` can't be addressed), and the reserved words `new`, `launch`, `detach`, `build` (they collide with admin routes, doc 05). | P1 |
| JEV-01.10 | `put` validates first, writes via temp-file + `os.replace` (atomic), `chmod 0o644`, returns the `ValidationResult`. Refuses to exceed `MAX_USER_SCHEMAS` (256). `delete` of a builtin-only name → `StoreError("built-in schemas are read-only")`; deleting an override removes the file and reverts to the builtin. | P1 |
| JEV-01.11 | `Settings.vibe_jev_dir: Path` (env `VIBE_JEV_DIR`, default `<repo>/jev`); created on demand by `SchemaStore`. Added to `.env.example` and the README `.env` table. | P2 |
| JEV-01.12 | Add `jsonschema>=4.23` to `[project.dependencies]`; `uv lock` (needs network, retry ≤3× per dev-workflow.md), commit `uv.lock`; `uv sync --dry-run --offline` must still show **no torch/CUDA**. | P1 |
| JEV-01.13 | `.gitignore`: `jev/*` + `!jev/.gitkeep` (open question in README — default gitignored). | P2 |

## Design notes

- **Why one validator?** The same text is validated three times (CLI offline, controller authoritative, admin
  live). Sharing `validate_schema_text` guarantees identical errors; the controller result is the one that counts.
- **Depth/size/ref limits** protect the controller (it runs `check_schema` on untrusted admin input) and vLLM
  (grammar compilation cost grows with schema size).
- **No remote `$ref` ever**: `jsonschema` can fetch refs; `referencing` does not by default, but make it explicit by
  passing an empty registry and rejecting non-`#` refs up front.
- **Hash scope**: the hash identifies *content*, not name. Admin edits that don't change content keep the active
  Jev "unchanged"; an edit of a schema that is currently active produces a **stale** flag (doc 02) instead of
  silently changing live behaviour.
- **Shadowing `default`** is how "modify the default schema" works from the admin page without mutating package
  data. Deleting the override restores the built-in; the UI says so.

## Task breakdown

- [x] Add `jsonschema` dep, `uv lock`, verify torch-free (`uv sync --dry-run --offline`).
- [x] Create `src/vibe/jev/{__init__,schema,store,errors}.py` + `builtin/{default,yes-no,yes-no-unsure}.json`.
- [x] Implement `validate_schema_text` + limits + warnings constants.
- [x] Implement `canonical_hash`, `wire_schema`, `schema_name_for_openai`.
- [x] Implement `SchemaStore` with atomic writes, shadowing, name rules.
- [x] `Settings.vibe_jev_dir`; `.env.example`; README `.env` row; `.gitignore`; `jev/.gitkeep`.
- [x] Tests (see doc 06 §A): every error and warning path, hash stability, store round-trip, shadow/revert,
      path-traversal names, atomic-write (simulated crash leaves old file).
- [x] Verify wheel contains `builtin/*.json` (`uv build --wheel` then list, or `importlib.resources` test that
      runs from an installed copy).

## Acceptance criteria

- `load_builtin("default")` validates with zero errors; its `choice` enum is exactly `["YES","NO"]`.
- The user's original example schema validates (with the `$schema` warning/normalisation noted), and
  `{decision_type:"score", result:true}` is **rejected** by the built-in default but accepted by the original shape
  (test documents the difference).
- Whitespace-only edit ⇒ same `sha256`; a key change ⇒ different `sha256`.
- `put("../x", …)`, `put(".active", …)`, `put("A B", …)` all rejected; nothing written outside `vibe_jev_dir`.
- Root `uv sync --dry-run --offline` shows no new heavy packages beyond `jsonschema`, `jsonschema-specifications`,
  `referencing`, `rpds-py`, `attrs` (already present).

## Risks

| Risk | Mitigation |
| --- | --- |
| `uv lock` needs network, which is flaky in the sandbox | Do the lock step first and alone; retry ≤3×; the rest of the work is offline-safe |
| Root-level `oneOf` rejected by the pinned vLLM backend | Fallback shape noted above; runbook C2 decides before J2 ships |
| `RISKY_KEYWORDS` list is partly guesswork | Treat as warnings only; refine from runbook evidence |
| `rpds-py` is a compiled wheel | Pure wheels exist for linux x86_64 cp312; confirmed by `uv lock` resolution |
