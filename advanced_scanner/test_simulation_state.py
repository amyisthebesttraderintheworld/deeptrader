import pytest

import simulation_state


def test_simulation_snapshot_calculates_equity_for_long_and_short(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation_state, "STATE_FILE", tmp_path / "simulation_state.json")

    simulation_state.reset_simulation_state()
    simulation_state.merge_simulation_state(
        {
            "starting_balance": 100.0,
            "balance": 70.0,
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "entry": 10.0,
                    "mark_price": 11.0,
                    "size": 2.0,
                    "margin": 10.0,
                },
                {
                    "symbol": "ETHUSDT",
                    "direction": "SHORT",
                    "entry": 20.0,
                    "markPrice": 18.0,
                    "size": 1.5,
                    "margin": 5.0,
                },
            ],
            "closed_trades": [{"symbol": "SOLUSDT", "pnl_usd": 3.5}],
            "metadata": {"origin": "pytest"},
        }
    )

    snapshot = simulation_state.get_simulation_snapshot()
    summary = snapshot["summary"]

    assert summary["available_balance"] == pytest.approx(70.0)
    assert summary["locked_margin"] == pytest.approx(15.0)
    assert summary["unrealized_pnl"] == pytest.approx(5.0)
    assert summary["realized_pnl"] == pytest.approx(3.5)
    assert summary["equity"] == pytest.approx(90.0)
    assert summary["equity_change"] == pytest.approx(-10.0)
    assert summary["open_positions_count"] == 2
    assert snapshot["positions"][0]["direction"] == "LONG"
    assert snapshot["positions"][1]["direction"] == "SHORT"
    assert snapshot["positions"][1]["mark_price"] == 18.0


def test_simulation_state_accepts_existing_workflow_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation_state, "STATE_FILE", tmp_path / "simulation_state.json")

    simulation_state.reset_simulation_state()
    simulation_state.merge_simulation_state(
        {
            "startingEquity": 125.0,
            "openPositions": [{"symbol": "ADAUSDT", "side": "Buy", "entryPrice": 1.0, "size": 3, "margin": 6}],
            "recentClosedPositions": [{"symbol": "XRPUSDT", "pnl": 1.25}],
            "scannerResults": [{"symbol": "OPUSDT", "score": 88.1}],
        }
    )

    snapshot = simulation_state.get_simulation_snapshot()

    assert snapshot["starting_balance"] == pytest.approx(125.0)
    assert snapshot["positions"][0]["entry"] == pytest.approx(1.0)
    assert snapshot["summary"]["closed_trades_count"] == 1
    assert snapshot["scanner_results"][0]["symbol"] == "OPUSDT"
