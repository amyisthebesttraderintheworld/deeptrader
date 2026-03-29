import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from advanced_scanner.backtest import backtest_portfolio
from advanced_scanner.trade_objects import Order, Trade

class TestBacktestLogic(unittest.TestCase):

    def setUp(self):
        self.starting_capital = 1000.0
        # Mock historical data for a single symbol
        # [ts, interval, last, open, high, low, close, vol, turnover]
        self.mock_ohlcv_data = []
        for i in range(100): # 100 bars, more than MIN_LOOKBACK
            ts = 1678886400 + i * 3600 # Hourly bars
            self.mock_ohlcv_data.append([
                float(ts), 3600.0, 1000.0 + i, 1000.0 + i, 1001.0 + i, 999.0 + i, 1000.0 + i, 1000.0, 1000000.0
            ])
        
        self.sym_rows_map = {"TESTUSDT": self.mock_ohlcv_data}
        self.funding_map = {"TESTUSDT": 0.0001}
        self.vol_map = {"TESTUSDT": 10000000.0}

    @patch('backtest.calculate_all_scores')
    @patch('backtest.adx')
    @patch('backtest.atr')
    def test_trade_generation_with_low_threshold(self, mock_atr, mock_adx, mock_calculate_all_scores):
        # Configure mocks
        mock_calculate_all_scores.return_value = np.array([0] * 60 + [50] * 40) # Score 50 for last 40 bars
        mock_adx.return_value = np.array([30] * 100) # ADX always above 20
        mock_atr.return_value = np.array([10] * 100) # ATR value

        # Run backtest with parameters designed to generate trades
        all_trades, equity_curve, final_capital, concurrent_log = backtest_portfolio(
            sym_rows_map=self.sym_rows_map,
            funding_map=self.funding_map,
            vol_map=self.vol_map,
            threshold=10,        # Very low threshold to ensure entry
            hold_bars=1,         # Short hold duration to ensure exit
            cooldown=0,          # No cooldown
            starting_capital=self.starting_capital,
            risk_per_trade=0.05,
            adx_filter=True,    # ADX filter active, but mocked to pass
            sl_mult=1.0,
            tp_mult=1.0
        )

        self.assertGreater(len(all_trades), 0, "Expected trades to be generated")
        self.assertNotEqual(final_capital, self.starting_capital, "Expected capital to change, indicating trades were processed")

    @patch('backtest.calculate_all_scores')
    @patch('backtest.adx')
    @patch('backtest.atr')
    def test_no_trade_generation_with_high_threshold(self, mock_atr, mock_adx, mock_calculate_all_scores):
        # Configure mocks
        mock_calculate_all_scores.return_value = np.array([0] * 60 + [20] * 40) # Score 20, below high threshold
        mock_adx.return_value = np.array([30] * 100)
        mock_atr.return_value = np.array([10] * 100)

        # Run backtest with parameters designed NOT to generate trades
        all_trades, equity_curve, final_capital, concurrent_log = backtest_portfolio(
            sym_rows_map=self.sym_rows_map,
            funding_map=self.funding_map,
            vol_map=self.vol_map,
            threshold=25,        # High threshold to prevent entry
            hold_bars=1,
            cooldown=0,
            starting_capital=1000.0,
            risk_per_trade=0.05,
            adx_filter=True,
            sl_mult=1.0,
            tp_mult=1.0
        )

        self.assertEqual(len(all_trades), 0, "Expected no trades to be generated")
        self.assertEqual(final_capital, 1000.0, "Expected capital to remain unchanged")

if __name__ == '__main__':
    unittest.main()
