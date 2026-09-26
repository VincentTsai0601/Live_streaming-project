# Recommendation harness

## Scope and implementation record

Approved scope: deterministic pytest coverage, repository instructions, Ruff,
local hooks, CI checks, and code-review guidance. No ranking algorithm change.
The only production-code cleanup is removal of an unused dotenv import.
The runtime dependency on python-dotenv is now declared directly.

- [x] Align AGENTS.md with pfid, embedding-row mapping, and absent features.
- [x] Isolate test discovery, dependencies, and temporary caches.
- [x] Cover parsing, documents, validation, scoring, filters, and reranking.
- [x] Cover cache and explanation fallbacks with deterministic doubles.
- [x] Configure lint, hooks, CI, and the pull-request checklist.
- [x] Record final verification and review results.

These tests characterize existing behavior; no artificial failing production
change was introduced just to create a red/green cycle. Behavioral fixes
remain separate and should start with a failing regression test.

## Coverage

- Unsorted pfids and known dense vectors verify position-to-ID mapping.
- Dense and real sparse TF-IDF paths verify filters and backend selection.
- Explicit score expectations verify 65/35 blending and semantic-only routing.
- Required tags use AND logic; explicit gender overrides inferred gender.
- Reranking tests exclude a noncandidate with the highest relevance score,
  check uniqueness and small candidate sets, and verify diversity penalties.
- Cache tests cover reuse, CSV/model key changes, and row-count mismatch.
- Mocked Gemini tests cover success, empty results, missing SDK/key, and errors.
- Existing CI retrieval assertions also run offline against the real CSV.

## Known defects and untested behavior

- `gemini_model` is ignored. Its characterization test records environment
  selection, not desired behavior; replace it with the selected-model
  regression when fixing the function.
- Corrupt caches currently raise. Same-row-count caches are not checked for
  dimensions or nonfinite values; document-building changes do not alter keys.
- Reranking tie-breaking uses row position despite the pfid comment.
- An installed encoder that fails to load does not fall back to TF-IDF.
- CLI does not load .env; Streamlit does, with override=True.
- Empty datasets and null/noninteger pfids lack a defined validation policy.
- Generated explanations are not checked for factual grounding or length.
- Live Gemini, actual transformer inference, model quality, UI interactions,
  Docker builds, and hosted CI execution are not proven by offline tests.
- FAISS, reviewed-item exclusion, and explicit user cold-start routing are
  absent: their checks are N/A, not passing. Backend and scoring routes are
  tested but are not substitutes for user cold-start behavior.
- Score labels were manually inspected; similarity is not a probability.

## Review gate

Before merging, record full-suite and lint results, review the diff for
algorithm changes, check ID mapping/filter/reranker invariants, and state
untested integrations. CI cannot determine explanation truth or ranking
quality; these still require evaluation and human review.

## Verification record (2026-09-27)

- `python -m pytest -q --tb=short`: 48 passed; one environment warning.
- `python -m pytest --collect-only -q`: 48 tests; live Gemini probe excluded.
- `python -m ruff check .`: passed.
- `python -m ruff format --check .`: passed.
- Explicit production filenames also honor formatting exclusions.
- `python scripts/check_hygiene.py`: passed.
- `git diff --check`: passed.
- Pre-commit configuration validation and hook installation: passed.
- Pre-commit lint, formatting, and hygiene hooks: passed.
- Pre-push offline suite: passed.
- Independent read-only review: APPROVE; no blocking findings remain.

Local execution used Python 3.10.9 in a project virtual environment inheriting
system numerical packages. Pandas warned about the inherited optional
bottleneck 1.3.5; this did not fail tests. Clean Python 3.11 is configured in
CI but was not executed locally or on GitHub in this session.

Initial pytest runs encountered Windows permissions on the system temporary
directory. Test scratch now uses `.cache/pytest-harness`, which pytest clears
on each run. Do not store data there. Serialize local runs or give concurrent
runs distinct `--basetemp` paths. This is separate from embedding caches.

Hook checks were verified with the project virtual environment's Scripts
directory prepended to PATH inside the launcher process. An initial hook run
selected the machine's Anaconda Python and could not find Ruff. Keep the
project environment active when using these system-language hooks.

No commits, pushes, live API requests, or recommendation algorithm changes
were made.