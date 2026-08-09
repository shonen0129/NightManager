# ADR-0004: Timeout Standardization and Transactional Cache Store

- Date: 2026-08-09
- Status: accepted
- Deciders: Devin

## Context

AGENTS.md lists five CLI hang patterns: yfinance, fcntl file locks, auto-close,
API retry back-off, and order-filter confirmation. Timeouts were scattered and
hard-coded; cache relies on advisory file locks.

## Decision

- Centralize timeout constants in `leadlag.core.timeouts`.
- Add `leadlag.data.cache_store.SqliteCacheStore` as a transactional
  SQLite-backed alternative to pickle files and fcntl locks.
- Existing `run_with_timeout` and `file_lock(timeout=...)` are retained; the
  new modules provide the canonical defaults and an escape path from fcntl.

## Consequences

- Timeout values live in one file and can be tuned globally.
- SQLite/WAL removes the need for explicit advisory file locks for new caches.
- A future migration can replace pkl file usage with `SqliteCacheStore`.
