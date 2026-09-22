"""Durable execution state for safe restart and retry decisions.

The broker API is not transactional with the local process.  This store keeps
the local intent and every broker observation in SQLite so a timeout or process
crash is represented as ``executing`` or ``reconciliation_required`` instead
of being mistaken for a failed submission.  It deliberately does not submit, cancel, or retry
orders; callers must reconcile with the broker before creating a new plan.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from leadlag.core.pnl import Fill
from leadlag.core.types import OrderRequest
from leadlag.execution.contracts import ExecutionPlan

DEFAULT_STATE_TIMEOUT_SECONDS = 5.0
EXECUTION_STATE_SCHEMA_VERSION = 2


class ExecutionStateError(RuntimeError):
    """Base class for durable execution-state errors."""


class ExecutionStateConflict(ExecutionStateError):
    """Raised when a run or lease would permit a duplicate execution."""

    def __init__(self, message: str, *, run_id: str | None = None) -> None:
        super().__init__(message)
        self.run_id = run_id


class ExecutionStateTransitionError(ExecutionStateError):
    """Raised when a caller attempts an invalid run-state transition."""


class ExecutionRunStatus:
    PREPARED = "prepared"
    EXECUTING = "executing"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    COMPLETED = "completed"
    FAILED_BEFORE_SUBMISSION = "failed_before_submission"


@dataclass(frozen=True)
class ExecutionRun:
    run_id: str
    job_key: str
    account_key: str
    strategy_key: str
    trade_date: str
    job_type: str
    decision_id: str | None
    status: str
    attempt_id: str
    created_at: str
    updated_at: str
    last_error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionLease:
    """A single-host, cross-process lease held in the execution state DB."""

    store: ExecutionStateStore
    scope: str
    owner: str
    ttl_seconds: float

    def renew(self) -> None:
        self.store.renew_lease(self.scope, self.owner, self.ttl_seconds)

    def release(self) -> None:
        self.store.release_lease(self.scope, self.owner)

    def __enter__(self) -> ExecutionLease:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _order_payload(order: OrderRequest) -> dict[str, Any]:
    return {
        "ticker": order.ticker,
        "side": order.side.value,
        "quantity": int(order.quantity),
        "order_type": order.order_type.value,
        "limit_price": order.limit_price,
        "margin_trade_type": order.margin_trade_type,
        "account_type": order.account_type,
        "is_close": bool(order.is_close),
        "close_position_order": int(order.close_position_order),
    }


def _intent_id(run_id: str, ordinal: int, order: OrderRequest) -> str:
    payload = {"run_id": run_id, "ordinal": ordinal, "order": _order_payload(order)}
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:32]


class ExecutionStateStore:
    """SQLite state store for run intent, broker observations, and leases."""

    def __init__(self, path: str | Path, *, timeout: float = DEFAULT_STATE_TIMEOUT_SECONDS) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = max(0.01, float(timeout))
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        conn.row_factory = sqlite3.Row
        # SQLite does not accept bind parameters in PRAGMA assignments.  The
        # value is derived from our validated float and therefore numeric-only.
        conn.execute(f"PRAGMA busy_timeout = {max(1, int(self.timeout * 1000))}")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS execution_runs (
                    run_id TEXT PRIMARY KEY,
                    job_key TEXT NOT NULL UNIQUE,
                    account_key TEXT NOT NULL,
                    strategy_key TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    decision_id TEXT,
                    status TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT,
                    metadata_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_execution_runs_status
                    ON execution_runs(status, updated_at);
                CREATE TABLE IF NOT EXISTS execution_schema (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS order_intents (
                    intent_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES execution_runs(run_id),
                    ordinal INTEGER NOT NULL,
                    purpose TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    is_close INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    broker_order_id TEXT,
                    latest_status TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(run_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_order_intents_run ON order_intents(run_id, ordinal);
                CREATE TABLE IF NOT EXISTS order_observations (
                    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_key TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL REFERENCES execution_runs(run_id),
                    intent_id TEXT REFERENCES order_intents(intent_id),
                    broker_order_id TEXT,
                    ticker TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_quantity INTEGER NOT NULL,
                    filled_quantity INTEGER,
                    remaining_quantity INTEGER,
                    observed_at TEXT NOT NULL,
                    raw_reference TEXT,
                    side TEXT,
                    fill_price REAL,
                    fee REAL,
                    fill_id TEXT,
                    fill_source TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_order_observations_run
                    ON order_observations(run_id, observation_id);
                CREATE TABLE IF NOT EXISTS execution_reconciliations (
                    reconciliation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES execution_runs(run_id),
                    checked_at TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    references_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_execution_reconciliations_run
                    ON execution_reconciliations(run_id, reconciliation_id);
                CREATE TABLE IF NOT EXISTS execution_leases (
                    scope TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    metadata_json TEXT
                );
                """
            )
            conn.execute(
                """INSERT OR IGNORE INTO execution_schema(singleton, schema_version, updated_at)
                VALUES (1, ?, ?)""",
                (EXECUTION_STATE_SCHEMA_VERSION, _utc_now()),
            )
            row = conn.execute(
                "SELECT schema_version FROM execution_schema WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise ExecutionStateError(
                    "Unsupported execution state schema version: "
                    "missing"
                )
            current_version = int(row["schema_version"])
            if current_version > EXECUTION_STATE_SCHEMA_VERSION:
                raise ExecutionStateError(
                    "Unsupported execution state schema version: "
                    f"{current_version}"
                )
            columns = {
                str(column[1])
                for column in conn.execute("PRAGMA table_info(order_observations)").fetchall()
            }
            required_columns = {
                "side",
                "fill_price",
                "fee",
                "fill_id",
                "fill_source",
            }
            if current_version < 2 or not required_columns.issubset(columns):
                for name, declaration in (
                    ("side", "TEXT"),
                    ("fill_price", "REAL"),
                    ("fee", "REAL"),
                    ("fill_id", "TEXT"),
                    ("fill_source", "TEXT"),
                ):
                    if name not in columns:
                        conn.execute(
                            f"ALTER TABLE order_observations ADD COLUMN {name} {declaration}"
                        )
                conn.execute(
                    "UPDATE execution_schema SET schema_version = ?, updated_at = ? WHERE singleton = 1",
                    (EXECUTION_STATE_SCHEMA_VERSION, _utc_now()),
                )

    @property
    def schema_version(self) -> int:
        """Return the on-disk schema version for migration checks."""
        return EXECUTION_STATE_SCHEMA_VERSION

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> ExecutionRun:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None
        return ExecutionRun(
            run_id=row["run_id"],
            job_key=row["job_key"],
            account_key=row["account_key"],
            strategy_key=row["strategy_key"],
            trade_date=row["trade_date"],
            job_type=row["job_type"],
            decision_id=row["decision_id"],
            status=row["status"],
            attempt_id=row["attempt_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_error=row["last_error"],
            metadata=metadata,
        )

    def get_run(self, run_id: str) -> ExecutionRun | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM execution_runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def get_run_by_job_key(self, job_key: str) -> ExecutionRun | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM execution_runs WHERE job_key = ?", (job_key,)).fetchone()
        return self._row_to_run(row) if row else None

    @staticmethod
    def _unresolved_run(
        conn: sqlite3.Connection,
        account_key: str,
        strategy_key: str,
        *,
        exclude_run_id: str | None = None,
    ) -> sqlite3.Row | None:
        """Return another run that still has an ambiguous broker outcome."""
        query = (
            "SELECT run_id FROM execution_runs "
            "WHERE account_key = ? AND strategy_key = ? "
            "AND status IN (?, ?)"
        )
        params: list[Any] = [
            str(account_key),
            str(strategy_key),
            ExecutionRunStatus.EXECUTING,
            ExecutionRunStatus.RECONCILIATION_REQUIRED,
        ]
        if exclude_run_id is not None:
            query += " AND run_id <> ?"
            params.append(exclude_run_id)
        return cast(sqlite3.Row | None, conn.execute(query + " LIMIT 1", params).fetchone())

    def prepare_run(
        self,
        *,
        account_key: str,
        strategy_key: str,
        trade_date: str,
        job_type: str,
        decision_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionRun:
        """Create or safely resume an unsubmitted run for one job key.

        A completed or in-flight run always blocks a second submission.  Only a
        run that is explicitly marked ``failed_before_submission`` can return to
        ``prepared``; an ambiguous broker outcome remains reconciliation-only.
        """
        job_key = "|".join((str(account_key), str(strategy_key), str(trade_date), str(job_type)))
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM execution_runs WHERE job_key = ?", (job_key,)).fetchone()
            if row:
                existing = self._row_to_run(row)
                if existing.status in {
                    ExecutionRunStatus.PREPARED,
                    ExecutionRunStatus.FAILED_BEFORE_SUBMISSION,
                }:
                    unresolved = self._unresolved_run(
                        conn,
                        existing.account_key,
                        existing.strategy_key,
                        exclude_run_id=existing.run_id,
                    )
                    if unresolved:
                        raise ExecutionStateConflict(
                            "Account has an unresolved execution; reconcile before resuming this job",
                            run_id=unresolved["run_id"],
                        )
                    if decision_id and existing.decision_id and decision_id != existing.decision_id:
                        raise ExecutionStateConflict(
                            f"Job key already prepared with a different decision: {job_key}",
                            run_id=existing.run_id,
                        )
                    conn.execute(
                        "UPDATE execution_runs SET status = ?, updated_at = ?, last_error = NULL WHERE run_id = ?",
                        (ExecutionRunStatus.PREPARED, now, existing.run_id),
                    )
                    refreshed = conn.execute(
                        "SELECT * FROM execution_runs WHERE run_id = ?", (existing.run_id,)
                    ).fetchone()
                    assert refreshed is not None
                    conn.execute("COMMIT")
                    return self._row_to_run(refreshed)
                raise ExecutionStateConflict(
                    f"Execution for job key is already {existing.status}; reconcile before retry: {job_key}",
                    run_id=existing.run_id,
                )

            unresolved = self._unresolved_run(conn, str(account_key), str(strategy_key))
            if unresolved:
                raise ExecutionStateConflict(
                    "Account has an unresolved execution; reconcile before a new job or trade date",
                    run_id=unresolved["run_id"],
                )

            run_id = uuid.uuid4().hex
            attempt_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO execution_runs
                (run_id, job_key, account_key, strategy_key, trade_date, job_type,
                 decision_id, status, attempt_id, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, job_key, str(account_key), str(strategy_key), str(trade_date),
                    str(job_type), decision_id, ExecutionRunStatus.PREPARED, attempt_id,
                    now, now, _json(dict(metadata)) if metadata else None,
                ),
            )
            row = conn.execute("SELECT * FROM execution_runs WHERE run_id = ?", (run_id,)).fetchone()
            assert row is not None
            conn.execute("COMMIT")
            return self._row_to_run(row)

    def record_plan(self, run_id: str, plan: ExecutionPlan) -> tuple[str, ...]:
        """Persist all order intents before the first broker submission."""
        now = _utc_now()
        intent_ids: list[str] = []
        orders: list[tuple[str, OrderRequest]] = [
            ("close", order) for order in plan.close_orders
        ] + [("new", order) for order in plan.new_orders]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute("SELECT status FROM execution_runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                raise ExecutionStateError(f"Unknown execution run: {run_id}")
            if run["status"] not in {ExecutionRunStatus.PREPARED, ExecutionRunStatus.FAILED_BEFORE_SUBMISSION}:
                raise ExecutionStateTransitionError(f"Cannot record an order plan while run is {run['status']}")
            saved = conn.execute(
                "SELECT payload_json FROM order_intents WHERE run_id = ? ORDER BY ordinal", (run_id,),
            ).fetchall()
            if saved and [row["payload_json"] for row in saved] != [_json(_order_payload(order)) for _, order in orders]:
                raise ExecutionStateConflict(f"Order plan changed for {run_id}", run_id=run_id)
            metadata_row = conn.execute("SELECT metadata_json FROM execution_runs WHERE run_id = ?", (run_id,)).fetchone()
            metadata = json.loads(metadata_row["metadata_json"] or "{}")
            metadata["execution_plan"] = plan.to_dict()
            conn.execute("UPDATE execution_runs SET metadata_json = ? WHERE run_id = ?", (_json(metadata), run_id))
            for ordinal, (purpose, order) in enumerate(orders):
                intent_id = _intent_id(run_id, ordinal, order)
                payload = _json(_order_payload(order))
                existing = conn.execute(
                    "SELECT payload_json FROM order_intents WHERE intent_id = ?", (intent_id,)
                ).fetchone()
                if existing and existing["payload_json"] != payload:
                    conn.execute("ROLLBACK")
                    raise ExecutionStateConflict(f"Order intent changed for {intent_id}", run_id=run_id)
                conn.execute(
                    """INSERT OR IGNORE INTO order_intents
                    (intent_id, run_id, ordinal, purpose, ticker, side, quantity, is_close,
                     payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        intent_id, run_id, ordinal, purpose, order.ticker, order.side.value,
                        int(order.quantity), int(order.is_close), payload, now, now,
                    ),
                )
                intent_ids.append(intent_id)
            conn.execute("COMMIT")
        return tuple(intent_ids)

    def mark_submission_started(self, run_id: str) -> None:
        self._transition(run_id, {ExecutionRunStatus.PREPARED}, ExecutionRunStatus.EXECUTING)

    def mark_failed_before_submission(self, run_id: str, error: str) -> None:
        self._transition(
            run_id,
            {ExecutionRunStatus.PREPARED},
            ExecutionRunStatus.FAILED_BEFORE_SUBMISSION,
            last_error=error,
        )

    def mark_reconciliation_required(self, run_id: str, error: str | None = None) -> None:
        self._transition(
            run_id,
            {
                ExecutionRunStatus.PREPARED,
                ExecutionRunStatus.EXECUTING,
                ExecutionRunStatus.RECONCILIATION_REQUIRED,
                # Allow reopening legacy completed runs discovered by recovery.
                ExecutionRunStatus.COMPLETED,
            },
            ExecutionRunStatus.RECONCILIATION_REQUIRED,
            last_error=error,
        )

    def mark_completed(self, run_id: str) -> None:
        self._transition(
            run_id,
            {ExecutionRunStatus.EXECUTING, ExecutionRunStatus.RECONCILIATION_REQUIRED},
            ExecutionRunStatus.COMPLETED,
        )

    def _transition(
        self,
        run_id: str,
        allowed: set[str],
        new_status: str,
        *,
        last_error: str | None = None,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM execution_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise ExecutionStateError(f"Unknown execution run: {run_id}")
            if row["status"] not in allowed:
                conn.execute("ROLLBACK")
                raise ExecutionStateTransitionError(
                    f"Invalid transition {row['status']} -> {new_status} for {run_id}"
                )
            if new_status == ExecutionRunStatus.EXECUTING:
                owner = conn.execute(
                    "SELECT account_key, strategy_key FROM execution_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                assert owner is not None
                unresolved = self._unresolved_run(
                    conn,
                    owner["account_key"],
                    owner["strategy_key"],
                    exclude_run_id=run_id,
                )
                if unresolved:
                    conn.execute("ROLLBACK")
                    raise ExecutionStateConflict(
                        "Account has an unresolved execution; reconcile before submission",
                        run_id=unresolved["run_id"],
                    )
            if new_status == ExecutionRunStatus.COMPLETED:
                checkpoint = conn.execute(
                    "SELECT outcome, errors_json FROM execution_reconciliations "
                    "WHERE run_id = ? ORDER BY reconciliation_id DESC LIMIT 1", (run_id,),
                ).fetchone()
                if checkpoint is None or checkpoint["outcome"] != "complete" or json.loads(checkpoint["errors_json"]):
                    raise ExecutionStateTransitionError("Completion requires a successful reconciliation checkpoint")
            conn.execute(
                "UPDATE execution_runs SET status = ?, updated_at = ?, last_error = ? WHERE run_id = ?",
                (new_status, now, last_error, run_id),
            )
            conn.execute("COMMIT")

    def record_observation(
        self,
        *,
        run_id: str,
        intent_id: str | None,
        broker_order_id: str | None,
        ticker: str,
        status: str,
        requested_quantity: int,
        filled_quantity: int | None = None,
        remaining_quantity: int | None = None,
        observed_at: str | None = None,
        raw_reference: str | None = None,
        side: str | None = None,
        fill_price: float | None = None,
        fee: float | None = None,
        fill_id: str | None = None,
        fill_source: str = "observed",
        observation_key: str | None = None,
    ) -> int:
        """Append one observation, idempotently for the same observation key."""
        observed = observed_at or _utc_now()
        key_payload = {
            "run_id": run_id,
            "intent_id": intent_id,
            "broker_order_id": broker_order_id,
            "ticker": ticker,
            "status": str(status),
            "requested_quantity": int(requested_quantity),
            "filled_quantity": filled_quantity,
            "remaining_quantity": remaining_quantity,
            "observed_at": observed,
            "side": side,
            "fill_price": fill_price,
            "fee": fee,
            "fill_id": fill_id,
            "fill_source": fill_source,
        }
        resolved_key = observation_key or hashlib.sha256(_json(key_payload).encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO order_observations
                (observation_key, run_id, intent_id, broker_order_id, ticker, status,
                 requested_quantity, filled_quantity, remaining_quantity, observed_at, raw_reference,
                 side, fill_price, fee, fill_id, fill_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    resolved_key, run_id, intent_id, broker_order_id, ticker, str(status),
                    int(requested_quantity), filled_quantity, remaining_quantity, observed, raw_reference,
                    side, fill_price, fee, fill_id, str(fill_source),
                ),
            )
            row = conn.execute(
                "SELECT observation_id FROM order_observations WHERE observation_key = ?",
                (resolved_key,),
            ).fetchone()
            assert row is not None
            if intent_id:
                conn.execute(
                    "UPDATE order_intents SET broker_order_id = COALESCE(?, broker_order_id), latest_status = ?, updated_at = ? WHERE intent_id = ?",
                    (broker_order_id, str(status), _utc_now(), intent_id),
                )
        return int(row["observation_id"])

    def record_result_set(
        self,
        run_id: str,
        plan: ExecutionPlan,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        """Persist compatible legacy result dictionaries against plan intents."""
        orders = list(plan.close_orders) + list(plan.new_orders)
        intent_ids = [_intent_id(run_id, idx, order) for idx, order in enumerate(orders)]
        # Submission and post-fill summaries have different list ordering.
        # Broker IDs, once observed, must retain their original intent, including
        # multiple split orders belonging to a single planned quantity.
        with self._connect() as conn:
            saved = conn.execute(
                "SELECT broker_order_id, intent_id, MAX(requested_quantity) AS quantity "
                "FROM order_observations WHERE run_id = ? AND intent_id IS NOT NULL "
                "AND broker_order_id IS NOT NULL GROUP BY broker_order_id, intent_id", (run_id,),
            ).fetchall()
        assignments = {str(row["broker_order_id"]): str(row["intent_id"]) for row in saved}
        allocated = dict.fromkeys(intent_ids, 0)
        for row in saved:
            allocated[str(row["intent_id"])] += int(row["quantity"])
        for record in records:
            ticker = str(record.get("ticker", ""))
            side = str(record.get("side", ""))
            quantity = int(record.get("quantity", 0) or 0)
            broker_id = str(record.get("order_id", ""))
            intent_id = assignments.get(broker_id)
            match = next(
                (
                    idx for idx in range(len(orders))
                    if orders[idx].ticker == ticker
                    and orders[idx].side.value == side
                    and allocated[intent_ids[idx]] + quantity <= int(orders[idx].quantity)
                ),
                None,
            ) if intent_id is None else None
            if match is not None:
                intent_id = intent_ids[match]
                allocated[intent_id] += quantity
                if broker_id:
                    assignments[broker_id] = intent_id
            requested = quantity
            filled = record.get("fill_quantity", record.get("filled_quantity"))
            remaining = record.get("remaining_quantity")
            if remaining is None and filled is not None:
                remaining = max(0, requested - int(filled))
            fill_detail = record.get("fill_detail") or {}
            if not isinstance(fill_detail, Mapping):
                fill_detail = {}
            fill_price = record.get("fill_price")
            fee = fill_detail.get(
                "sBaiBaiTesuryo",
                record.get("fee", record.get("commission")),
            )
            if fee is None:
                fee = record.get("fee", record.get("commission"))
            fill_id = record.get("fill_id", record.get("execution_id"))
            fill_source = "simulated" if str(record.get("status", "")).upper() == "SIMULATED" else "observed"
            self.record_observation(
                run_id=run_id,
                intent_id=intent_id,
                broker_order_id=str(record.get("order_id", "")) or None,
                ticker=ticker,
                status=str(record.get("status", "UNKNOWN")),
                requested_quantity=requested,
                filled_quantity=int(filled) if filled is not None else None,
                remaining_quantity=int(remaining) if remaining is not None else None,
                raw_reference=str(record.get("raw_reference")) if record.get("raw_reference") else None,
                side=side,
                fill_price=float(fill_price) if fill_price is not None else None,
                fee=float(fee) if fee is not None else None,
                fill_id=str(fill_id) if fill_id is not None else None,
                fill_source=fill_source,
                # Result dictionaries are a snapshot of one submission call;
                # retries of that call must not duplicate the same observation.
                observation_key=hashlib.sha256(
                    _json(
                        {
                            "run_id": run_id,
                            "intent_id": intent_id,
                            "broker_order_id": str(record.get("order_id", "")) or None,
                            "ticker": ticker,
                            "status": str(record.get("status", "UNKNOWN")),
                            "requested_quantity": requested,
                            "filled_quantity": int(filled) if filled is not None else None,
                            "remaining_quantity": int(remaining) if remaining is not None else None,
                            "fill_price": fill_price,
                            "fee": fee,
                            "fill_id": fill_id,
                        }
                    ).encode("utf-8")
                ).hexdigest(),
            )

    def list_observed_fills(self, run_id: str) -> tuple[Fill, ...]:
        """Return the latest price-bearing fill observation for each order.

        Broker polling often reports cumulative filled quantity.  The state
        store therefore selects the latest observation per broker order (or
        explicit fill ID) instead of adding every poll as a new execution.
        A broker that supplies individual fill IDs can be replayed exactly;
        cumulative-only responses remain one auditable average-price Fill.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT o.*, r.trade_date
                FROM order_observations AS o
                JOIN execution_runs AS r ON r.run_id = o.run_id
                WHERE o.run_id = ? AND o.fill_price IS NOT NULL
                  AND o.filled_quantity IS NOT NULL AND o.filled_quantity > 0
                ORDER BY o.observation_id""",
                (run_id,),
            ).fetchall()
        latest: dict[str, sqlite3.Row] = {}
        for row in rows:
            key = str(row["fill_id"] or row["broker_order_id"] or row["observation_id"])
            latest[key] = row
        fills: list[Fill] = []
        for row in latest.values():
            if row["fee"] is None and str(row["fill_source"] or "observed") == "observed":
                continue
            try:
                fills.append(
                    Fill(
                        trade_date=row["trade_date"],
                        ticker=row["ticker"],
                        side=row["side"],
                        quantity=int(row["filled_quantity"]),
                        price=float(row["fill_price"]),
                        fee=float(row["fee"] or 0.0),
                        source=str(row["fill_source"] or "observed"),
                        order_id=row["broker_order_id"],
                    )
                )
            except (TypeError, ValueError):
                # Incomplete broker payloads stay observable in SQLite but do
                # not become a zero/invalid accounting event.
                continue
        return tuple(fills)

    def record_reconciliation(
        self,
        run_id: str,
        *,
        outcome: str,
        errors: Sequence[str] = (),
        references: Mapping[str, Any] | None = None,
    ) -> int:
        """Persist one post-submission reconciliation checkpoint.

        Reconciliation is append-only: a later position, wallet, fill, or
        journal check must remain auditable even when an earlier checkpoint was
        incomplete.  The run status is transitioned by the caller because the
        caller owns the safe point at which a retry may be considered.
        """
        normalized = str(outcome).lower()
        if normalized not in {"complete", "incomplete", "pending"}:
            raise ValueError(f"Unsupported reconciliation outcome: {outcome}")
        if self.get_run(run_id) is None:
            raise ExecutionStateError(f"Unknown execution run: {run_id}")
        checked_at = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO execution_reconciliations
                (run_id, checked_at, outcome, errors_json, references_json)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    run_id,
                    checked_at,
                    normalized,
                    _json(list(errors)),
                    _json(dict(references)) if references else None,
                ),
            )
            if cursor.lastrowid is None:
                raise ExecutionStateError("SQLite did not return a reconciliation id")
            return int(cursor.lastrowid)

    def latest_reconciliation(self, run_id: str) -> dict[str, Any] | None:
        """Return the newest reconciliation checkpoint for a run."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM execution_reconciliations
                WHERE run_id = ? ORDER BY reconciliation_id DESC LIMIT 1""",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["errors"] = json.loads(result.pop("errors_json"))
        result["references"] = (
            json.loads(result.pop("references_json"))
            if result.get("references_json")
            else None
        )
        return result

    def list_recovery_candidates(self) -> list[ExecutionRun]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM execution_runs
                WHERE status IN (?, ?) ORDER BY updated_at""",
                (ExecutionRunStatus.EXECUTING, ExecutionRunStatus.RECONCILIATION_REQUIRED),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def owns_lease(self, scope: str, owner: str) -> bool:
        """Check the persisted owner and expiration of an enclosing job."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM execution_leases WHERE scope = ? AND owner = ? AND expires_at > ?",
                (scope, owner, time.time()),
            ).fetchone()
        return row is not None

    def acquire_lease(
        self,
        scope: str,
        *,
        owner: str | None = None,
        ttl_seconds: float = 1800.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionLease:
        """Acquire a cross-process lease; expired leases may be reclaimed."""
        owner_id = owner or uuid.uuid4().hex
        now = time.time()
        expires = now + max(0.1, float(ttl_seconds))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM execution_leases WHERE scope = ?", (scope,)).fetchone()
            if row and float(row["expires_at"]) > now and row["owner"] != owner_id:
                conn.execute("ROLLBACK")
                raise ExecutionStateConflict(
                    f"Execution lease is held for scope {scope!r} by another owner"
                )
            conn.execute(
                "INSERT OR REPLACE INTO execution_leases(scope, owner, acquired_at, expires_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (scope, owner_id, now, expires, _json(dict(metadata)) if metadata else None),
            )
            conn.execute("COMMIT")
        return ExecutionLease(self, scope, owner_id, max(0.1, float(ttl_seconds)))

    def renew_lease(self, scope: str, owner: str, ttl_seconds: float) -> None:
        now = time.time()
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE execution_leases SET expires_at = ? WHERE scope = ? AND owner = ?",
                (now + max(0.1, float(ttl_seconds)), scope, owner),
            ).rowcount
        if not updated:
            raise ExecutionStateConflict(f"Cannot renew lease not owned by {owner}: {scope}")

    def release_lease(self, scope: str, owner: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM execution_leases WHERE scope = ? AND owner = ?", (scope, owner))

    def lease_info(self, scope: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM execution_leases WHERE scope = ?", (scope,)).fetchone()
        return dict(row) if row else None
