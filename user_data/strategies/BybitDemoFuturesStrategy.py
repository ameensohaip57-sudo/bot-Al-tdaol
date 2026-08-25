"""
BybitDemoFuturesStrategy

استراتيجية اتجاه/ارتداد قصيرة الأجل لعقود Bybit Futures.
مهم: هذه الاستراتيجية ليست ضماناً للربح. اختبرها على بيانات مناسبة وDry-run قبل أي استخدام فعلي.

الافتراضات:
- timeframe: 5m
- trading_mode: futures
- margin_mode: isolated
- can_short: True
- leverage الافتراضي من callback هو 1x لتقليل خطر التصفية.
"""

from datetime import datetime
from typing import Optional

import talib.abstract as ta
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from freqtrade.vendor.qtpylib import indicators as qtpylib


class BybitDemoFuturesStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # تشغيل الإشارات على آخر شمعة مكتملة فقط
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 240

    # العقود الآجلة: دعم Long وShort
    can_short = True

    # وقف طوارئ واسع نسبياً، والوقف الديناميكي أدق من خلال custom_stoploss
    stoploss = -0.025
    use_custom_stoploss = True

    # أهداف ربح واقعية نسبياً قبل احتساب الرسوم والتمويل والانزلاق
    minimal_roi = {
        "0": 0.012,
        "20": 0.008,
        "60": 0.004,
        "120": 0.0,
    }

    # أوامر Limit تقلل الانزلاق، مع إلغاء الأوامر العالقة بسرعة
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC",
    }

    @property
    def protections(self):
        """حماية من التكرار السريع وسلسلة الخسائر."""
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 3,
                "stop_duration_candles": 6,
                "required_profit": 0.0,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 12,
                "stop_duration_candles": 8,
                "max_allowed_drawdown": 0.12,
            },
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """حساب المؤشرات بطريقة vectorized لتكون سريعة في backtest/live."""
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        dataframe["volume_mean"] = dataframe["volume"].rolling(20).mean()
        dataframe["volatility_mean"] = dataframe["atr_pct"].rolling(50).mean()

        # Bollinger Bands لاكتشاف الارتداد من مناطق التطرف
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]

        # يمنع التداول في سوق ضعيف الحركة أو شديد الاضطراب بشكل غير طبيعي.
        dataframe["volatility_ok"] = (
            (dataframe["atr_pct"] > 0.0008)
            & (dataframe["atr_pct"] < 0.035)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """إشارات دخول تجمع الاتجاه، الزخم، السيولة، والارتداد."""
        long_trend = (
            (dataframe["close"] > dataframe["ema_200"])
            & (dataframe["ema_20"] > dataframe["ema_50"])
            & (dataframe["adx"] > 18)
            & dataframe["volatility_ok"]
            & (dataframe["volume"] > dataframe["volume_mean"] * 0.8)
        )
        long_trigger = (
            qtpylib.crossed_above(dataframe["close"], dataframe["ema_20"])
            | (
                (dataframe["low"] <= dataframe["ema_20"] * 1.002)
                & (dataframe["close"] > dataframe["ema_20"])
            )
            | qtpylib.crossed_above(dataframe["close"], dataframe["bb_mid"])
        )
        long_momentum = dataframe["rsi"].between(48, 70)

        dataframe.loc[
            long_trend & long_trigger & long_momentum & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "trend_pullback_long")

        short_trend = (
            (dataframe["close"] < dataframe["ema_200"])
            & (dataframe["ema_20"] < dataframe["ema_50"])
            & (dataframe["adx"] > 18)
            & dataframe["volatility_ok"]
            & (dataframe["volume"] > dataframe["volume_mean"] * 0.8)
        )
        short_trigger = (
            qtpylib.crossed_below(dataframe["close"], dataframe["ema_20"])
            | (
                (dataframe["high"] >= dataframe["ema_20"] * 0.998)
                & (dataframe["close"] < dataframe["ema_20"])
            )
            | qtpylib.crossed_below(dataframe["close"], dataframe["bb_mid"])
        )
        short_momentum = dataframe["rsi"].between(30, 52)

        dataframe.loc[
            short_trend & short_trigger & short_momentum & (dataframe["volume"] > 0),
            ["enter_short", "enter_tag"],
        ] = (1, "trend_pullback_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """خروج عند فقدان الاتجاه أو وصول السعر إلى الطرف المقابل للنطاق."""
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe["close"], dataframe["ema_20"])
                & (dataframe["rsi"] < 48)
            )
            | qtpylib.crossed_below(dataframe["close"], dataframe["bb_mid"]),
            ["exit_long", "exit_tag"],
        ] = (1, "momentum_fade_long")

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["close"], dataframe["ema_20"])
                & (dataframe["rsi"] > 52)
            )
            | qtpylib.crossed_above(dataframe["close"], dataframe["bb_mid"]),
            ["exit_short", "exit_tag"],
        ] = (1, "momentum_fade_short")

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """وقف مبني على ATR: يتحول إلى حماية ربح بعد تحقق حركة إيجابية."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        candle = dataframe.iloc[-1]
        atr = float(candle.get("atr", 0.0))
        if atr <= 0 or current_rate <= 0:
            return self.stoploss

        if current_profit >= 0.010:
            # حماية ربح تقريبية بعد +1%، مع ترك مساحة للتذبذب.
            if trade.is_short:
                stop_price = min(trade.open_rate * 0.998, current_rate + atr * 0.8)
            else:
                stop_price = max(trade.open_rate * 1.002, current_rate - atr * 0.8)
        elif current_profit >= 0.004:
            if trade.is_short:
                stop_price = current_rate + atr * 1.5
            else:
                stop_price = current_rate - atr * 1.5
        else:
            if trade.is_short:
                stop_price = trade.open_rate + atr * 2.0
            else:
                stop_price = trade.open_rate - atr * 2.0

        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """رافعة محافظة؛ يمكن رفعها فقط بعد اجتياز اختبارات مستقلة."""
        return min(1.0, max_leverage)
