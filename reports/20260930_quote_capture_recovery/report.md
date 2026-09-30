# 2026-09-30 quote capture recovery follow-up

## Finding

The scheduled read-only quote capture failed on 2026-09-28, 2026-09-29, and 2026-09-30 because the request targeted retired Tachibana API v4r9 and received HTTP 404. The application default and `.env.example` were already v4r10; the local ignored `.env` overrode them with a v4r9 URL. The capture entry point loads that file on each invocation, so the mismatch survived the earlier code-default update.

## Changes

- Updated only `TACHIBANA_API_URL` in the local ignored `.env` to the production v4r10 base URL. Other local credentials were left untouched.
- Added a pre-request guard so capture rejects v4r9 configuration with an actionable error.
- Redacted URL query strings before capture errors are written to JSONL or the stage report. This prevents authentication request parameters from entering new capture logs.
- Documented that the scheduled capture loads the project `.env` at runtime in [the daily operations manual](../../docs/日次運用手順書.md).

## Verification and remaining acceptance

- `tests/unit/test_0910_microstructure.py`: 7 passed, including retired-endpoint rejection, successful v4r10 acceptance, and error-query redaction through the capture record path.
- The installed LaunchAgent and repository template do not define a competing `TACHIBANA_API_URL` override.
- No broker request was made during this repair. The next scheduled real-market capture must confirm the corrected local environment; the 2026-09-30 capture remains a failed immutable observation.
- This addresses the v4r9 configuration cause for issue #27 and enables future issue #23 observations. It does not provide actual-account PnL evidence or satisfy the 250-day forward gate.
