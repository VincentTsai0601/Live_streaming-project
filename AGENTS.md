# Repository Working Rules

## System and scope
- Recommend streamers identified by `pfid`; core logic is in streamer_recommender.py.
- Gemini generates explanations only and must not change ranked IDs.
- Explain changes to scoring, filtering, embedding inputs, routing, or tie-breaking.
- Keep algorithm changes separate from harness and formatting work.

## Required checks before completing a code change
1. Run existing automated tests and tests relevant to the change.
2. Verify embedding rows and ranked positions map to the correct pfid.
   If FAISS is introduced, verify its ID-to-pfid mapping too.
3. Verify gender and required-tag filters remain enforced.
4. Verify reranking returns only unique members of the candidate set.
5. Test tag-aware versus semantic-only scoring and backend selection.
6. Do not describe similarity or hybrid scores as probabilities.
7. Run lint and formatting checks.
8. Report commands, results, untested behavior, and potential failures.

## Absent features
- Reviewed-item exclusion and explicit user cold-start routing are not implemented.
- Report these checks as not applicable, never as passing.
- If introduced, require reviewed-item exclusion and cold-start regression tests.

## Test isolation and review
- Default tests must not call Gemini, download models, or require secrets.
- Use synthetic data, deterministic encoder doubles, and temporary caches.
- Keep real-model and live-API tests explicitly opt-in.
- Review ID mapping, cache alignment, failure paths, and explanation grounding.
- Never weaken assertions or skip failures to make checks pass.
- Never commit secrets, .env files, or generated caches.
