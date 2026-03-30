from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from advanced_scanner.config import DEFAULT_CAPITAL

_RUNTIME_DIR = Path(__file__).resolve().parent / ".runtime"
STATE_FILE = Path(os.getenv("SIM_STATE_FILE", _RUNTIME_DIR / "simulation_state.json"))
_STATE_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _default_starting_balance() -> float:
    return _as_float(os.getenv("SIM_STARTING_BALANCE"), _as_float(DEFAULT_CAPITAL, 100.0))


def _default_state() -> dict[str, Any]:
    starting_balance = _default_starting_balance()
    return {
        "starting_balance": starting_balance,
        "balance": starting_balance,
        "positions": [],
        "closed_trades": [],
        "scanner_results": [],
        "tickers": [],
        "metadata": {},
        "session": os.getenv("SIM_SESSION", "railway"),
        "source": "deeptrader-api",
        "last_scan_at": None,
        "updated_at": _utc_now(),
    }


def _ensure_runtime_dir() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_simulation_state_unlocked() -> dict[str, Any]:
    _ensure_runtime_dir()

    if not STATE_FILE.exists():
        state = _default_state()
        STATE_FILE.write_text(json.dumps(state, indent=2))
        return state

    try:
        state = json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        state = _default_state()
        state["metadata"] = {"reinitialized": True}
        STATE_FILE.write_text(json.dumps(state, indent=2))
        return state

    normalized = _normalize_state(state)
    if normalized != state:
        STATE_FILE.write_text(json.dumps(normalized, indent=2))
    return normalized


def _normalize_position(position: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(position)

    if "mark_price" not in normalized and "markPrice" in normalized:
        normalized["mark_price"] = normalized["markPrice"]
    if "entry" not in normalized and "entry_price" in normalized:
        normalized["entry"] = normalized["entry_price"]
    if "entry" not in normalized and "entryPrice" in normalized:
        normalized["entry"] = normalized["entryPrice"]
    if "entry_time" not in normalized and "entryTime" in normalized:
        normalized["entry_time"] = normalized["entryTime"]
    if "timestamp" not in normalized and normalized.get("entry_time"):
        normalized["timestamp"] = normalized["entry_time"]

    direction = str(normalized.get("direction", "")).upper()
    side = str(normalized.get("side", "")).upper()
    if not side and direction in {"LONG", "SHORT"}:
        normalized["side"] = "Buy" if direction == "LONG" else "Sell"
    elif side == "BUY":
        normalized["side"] = "Buy"
    elif side == "SELL":
        normalized["side"] = "Sell"

    return normalized


def _normalize_trade(trade: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(trade)
    if "pnl" not in normalized and "pnl_usd" in normalized:
        normalized["pnl"] = normalized["pnl_usd"]
    if "timestamp" not in normalized and "exit_time" in normalized:
        normalized["timestamp"] = normalized["exit_time"]
    return normalized


def _normalize_state(state: Any) -> dict[str, Any]:
    base = _default_state()
    raw = _as_dict(state)

    if "starting_balance" not in raw and "startingEquity" in raw:
        raw["starting_balance"] = raw["startingEquity"]
    if "positions" not in raw and "open_positions" in raw:
        raw["positions"] = raw["open_positions"]
    if "positions" not in raw and "openPositions" in raw:
        raw["positions"] = raw["openPositions"]
    if "closed_trades" not in raw and "recent_closed_positions" in raw:
        raw["closed_trades"] = raw["recent_closed_positions"]
    if "closed_trades" not in raw and "recentClosedPositions" in raw:
        raw["closed_trades"] = raw["recentClosedPositions"]
    if "scanner_results" not in raw and "scannerResults" in raw:
        raw["scanner_results"] = raw["scannerResults"]
    if "last_scan_at" not in raw and "lastScanAt" in raw:
        raw["last_scan_at"] = raw["lastScanAt"]

    base.update(raw)
    base["starting_balance"] = _as_float(base.get("starting_balance"), _default_starting_balance())
    base["balance"] = _as_float(base.get("balance"), base["starting_balance"])
    base["positions"] = [
        _normalize_position(position)
        for position in _as_list(base.get("positions"))
        if isinstance(position, dict)
    ]
    base["closed_trades"] = [
        _normalize_trade(trade)
        for trade in _as_list(base.get("closed_trades"))
        if isinstance(trade, dict)
    ]
    base["scanner_results"] = _as_list(base.get("scanner_results"))
    base["tickers"] = _as_list(base.get("tickers"))
    base["metadata"] = _as_dict(base.get("metadata"))
    base["updated_at"] = str(base.get("updated_at") or _utc_now())
    return base


def _canonicalize_patch(patch: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(_as_dict(patch))
    alias_map = {
        "startingEquity": "starting_balance",
        "open_positions": "positions",
        "openPositions": "positions",
        "recent_closed_positions": "closed_trades",
        "recentClosedPositions": "closed_trades",
        "scannerResults": "scanner_results",
        "lastScanAt": "last_scan_at",
    }

    for alias, canonical in alias_map.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized[alias]

    for alias in alias_map:
        normalized.pop(alias, None)

    return normalized


def load_simulation_state() -> dict[str, Any]:
    with _STATE_LOCK:
        return _load_simulation_state_unlocked()


def save_simulation_state(state: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_state(state)
    normalized["updated_at"] = _utc_now()

    with _STATE_LOCK:
        _ensure_runtime_dir()
        STATE_FILE.write_text(json.dumps(normalized, indent=2))

    return normalized


def merge_simulation_state(patch: dict[str, Any]) -> dict[str, Any]:
    with _STATE_LOCK:
        _ensure_runtime_dir()
        current = _load_simulation_state_unlocked()
        patch = _canonicalize_patch(patch)

        merged = dict(current)
        for key, value in _as_dict(patch).items():
            if key == "metadata":
                merged["metadata"] = {**_as_dict(current.get("metadata")), **_as_dict(value)}
            else:
                merged[key] = value

        normalized = _normalize_state(merged)
        normalized["updated_at"] = _utc_now()
        STATE_FILE.write_text(json.dumps(normalized, indent=2))
        return normalized


def reset_simulation_state() -> dict[str, Any]:
    return save_simulation_state(_default_state())


def _position_direction(position: dict[str, Any]) -> str:
    side = str(position.get("side", "")).upper()
    direction = str(position.get("direction", "")).upper()

    if side == "BUY" or direction == "LONG":
        return "LONG"
    if side == "SELL" or direction == "SHORT":
        return "SHORT"
    return "UNKNOWN"


def _position_unrealized_pnl(position: dict[str, Any]) -> float:
    if position.get("mark_price") is None:
        return _as_float(position.get("pnl"), 0.0)

    entry = _as_float(position.get("entry"), 0.0)
    mark = _as_float(position.get("mark_price"), entry)
    size = abs(_as_float(position.get("size"), 0.0))
    direction = _position_direction(position)

    if direction == "SHORT":
        return (entry - mark) * size
    return (mark - entry) * size


def get_simulation_snapshot() -> dict[str, Any]:
    state = load_simulation_state()
    positions = []

    locked_margin = 0.0
    unrealized_pnl = 0.0

    for raw_position in state["positions"]:
        position = dict(raw_position)
        direction = _position_direction(position)
        position["direction"] = direction
        position["unrealized_pnl"] = _position_unrealized_pnl(position)
        positions.append(position)
        locked_margin += _as_float(position.get("margin"), 0.0)
        unrealized_pnl += _as_float(position.get("unrealized_pnl"), 0.0)

    closed_trades = state["closed_trades"]
    available_balance = _as_float(state.get("balance"), 0.0)
    starting_balance = _as_float(state.get("starting_balance"), _default_starting_balance())
    realized_pnl = sum(
        _as_float(trade.get("pnl"), 0.0)
        for trade in closed_trades
    )
    equity = available_balance + locked_margin + unrealized_pnl
    equity_change = equity - starting_balance
    equity_change_percent = (equity_change / starting_balance * 100.0) if starting_balance else 0.0
    drawdown_percent = ((starting_balance - equity) / starting_balance * 100.0) if starting_balance else 0.0

    summary = {
        "starting_balance": starting_balance,
        "available_balance": available_balance,
        "locked_margin": locked_margin,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "equity": equity,
        "equity_change": equity_change,
        "equity_change_percent": equity_change_percent,
        "drawdown_percent": max(drawdown_percent, 0.0),
        "open_positions_count": len(positions),
        "closed_trades_count": len(closed_trades),
    }

    snapshot = dict(state)
    snapshot["positions"] = positions
    snapshot["summary"] = summary
    return snapshot
