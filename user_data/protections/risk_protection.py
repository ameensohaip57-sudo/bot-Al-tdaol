from freqtrade.strategy import IStrategy


class RiskProtection(IStrategy):

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 5
        },

        {
            "method": "StoplossGuard",
            "lookback_period_candles": 48,
            "trade_limit": 3,
            "stop_duration_candles": 24,
            "only_per_pair": False
        },

        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 96,
            "trade_limit": 20,
            "stop_duration_candles": 48,
            "max_allowed_drawdown": 0.10
        }
    ]