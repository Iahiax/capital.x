"""Small SQLite store for decisions, orders, and daily equity state."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path


class TradeStore:
    def __init__(self, path: str | Path = "trading_state.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    signal_time TEXT,
                    signal TEXT,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    deal_id TEXT,
                    direction TEXT NOT NULL,
                    size REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    trading_date TEXT PRIMARY KEY,
                    equity REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def record_decision(self, signal: str | None, payload: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO decisions(created_at, signal_time, signal, payload) "
                "VALUES (?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    str(payload.get("signal_time", "")),
                    signal,
                    json.dumps(payload, default=str),
                ),
            )

    def record_order(
        self,
        direction: str,
        size: float,
        response: dict | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO orders(created_at, deal_id, direction, size, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    str((response or {}).get("dealId", "")),
                    direction,
                    size,
                    json.dumps(response or {}, default=str),
                ),
            )

    def get_or_create_day_start(self, trading_date: str, equity: float) -> float:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT equity FROM equity_snapshots WHERE trading_date = ?",
                (trading_date,),
            ).fetchone()
            if row:
                return float(row[0])
            connection.execute(
                "INSERT INTO equity_snapshots(trading_date, equity, created_at) "
                "VALUES (?, ?, ?)",
                (trading_date, equity, datetime.now(UTC).isoformat()),
            )
            return equity