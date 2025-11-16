# -*- coding: utf-8 -*-
# Trading Bot v14 AGGRESSIVE - "True Yuichi Method" (CLI Version)
# Version plus agressive du v14 : plus gros sizing, martingale plus profonde,
# logs détaillés pour dashboard externe.

"""
PHILOSOPHIE DE YUICHI (raccourcie) :

1. N'AGIT QUE quand il a 3+ coups d'avance
2. OBSERVE pour détecter les patterns et pièges
3. RETOURNE les manipulations contre les manipulateurs
4. PATIENCE ABSOLUE -> attend le setup parfait
5. FRAPPE FORT quand l'adversaire est piégé
"""

import ccxt
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timezone
from collections import deque
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import os
import json

# Forcer l'output console en UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - [YUICHI-TRUE] - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("yuichi_bot_v14_aggressive_cli.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ENUMS & DATACLASSES
# ---------------------------------------------------------------------


class TrapType(Enum):
    BULL_TRAP = "bull_trap"
    BEAR_TRAP = "bear_trap"
    LIQUIDITY_GRAB = "liquidity_grab"
    WHALE_ACCUMULATION = "whale_accumulation"
    WHALE_DISTRIBUTION = "whale_distribution"
    RETAIL_FOMO = "retail_fomo"
    RETAIL_PANIC = "retail_panic"


class MarketSentiment(Enum):
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"
    MANIPULATION = "manipulation"


class GameState(Enum):
    OBSERVING = "observing"
    TRAP_DETECTED = "trap_detected"
    SETUP_FORMING = "setup_forming"
    GAME_OVER = "game_over"
    UNCERTAIN = "uncertain"


@dataclass
class MarketPsychology:
    sentiment: MarketSentiment
    fear_greed_index: float
    retail_positioning: float
    smart_money_flow: float
    volatility_regime: str
    manipulation_detected: bool
    trap_type: Optional[TrapType]
    trap_confidence: float
    game_state: GameState


# ---------------------------------------------------------------------
# CONFIG YUICHI v14 AGGRESSIVE
# ---------------------------------------------------------------------


class YuichiConfig:
    bot_name = "yuichi_v14_aggressive_cli"

    symbols = ["BTC/USDT"]
    timeframe = "1m"
    capital = 1000.0

    timeframes = {
        "micro": "1m",
        "tactical": "5m",
        "strategic": "15m",
        "oversight": "1h",
        "macro": "4h",
    }

    # Frais réalistes
    trading_fee_rate = 0.001   # 0.1 %
    slippage_rate = 0.0003     # 0.03 %
    min_profit_threshold = 0.004  # 0.4 %

    # Un peu plus agressif que v14 normal
    max_daily_loss_pct = 0.20
    max_trades_per_day = 14

    # Position sizing agressif
    base_positions = {
        "observing": 0,
        "trap_detected": 60,
        "setup_forming": 100,
        "game_over": 220,  # FRAPPE FORT
    }

    # Martingale plus profonde
    martingale_multipliers = [1.0, 1.8, 3.0, 5.0, 8.0]
    current_martingale_level = 0
    max_martingale_level = 4

    min_trap_confidence = 70.0
    min_setup_score = 7.0

    min_time_between_trades = 12  # minutes (plus rapide que v14 normal)
    last_trade_time: Optional[datetime] = None

    min_observation_candles = 50

    avoid_choppy_markets = True
    avoid_extreme_volatility = True

    trades_executed_today = 0
    daily_loss = 0.0
    cumulative_winnings = 0.0
    running = True

    # Stats psycho
    whale_patterns_detected: Dict[str, int] = {}
    retail_patterns_detected: Dict[str, int] = {}
    manipulation_history = deque(maxlen=100)

    # Pour dashboard
    status_dir = "status"
    battle_step = 0  # numéro du "combat" Yuichi


config = YuichiConfig()
os.makedirs(config.status_dir, exist_ok=True)

# ---------------------------------------------------------------------
# EXCHANGE
# ---------------------------------------------------------------------


class EliteExchange:
    def __init__(self):
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        self.rate_limit_buffer = 1.2

    def fetch_with_retry(self, func, *args, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded:
                wait_time = (2 ** attempt) * self.rate_limit_buffer
                logger.warning(f"[RATE] Rate limit - wait {wait_time:.1f}s")
                time.sleep(wait_time)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[EXCHANGE] Failed after {max_retries}: {e}")
                    return None
                logger.warning(f"[EXCHANGE] Error: {e} | retry {attempt + 1}")
                time.sleep(2 ** attempt)
        return None

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        return self.fetch_with_retry(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)

    def fetch_order_book(self, symbol, limit=20):
        return self.fetch_with_retry(self.exchange.fetch_order_book, symbol, limit=limit)


exchange = EliteExchange()

# ---------------------------------------------------------------------
# PERFORMANCE TRACKER
# ---------------------------------------------------------------------


class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0

        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0

    def log_trade(self, trade_data):
        self.trades.append(trade_data)
        pnl = trade_data["profit_loss"]

        if pnl > 0:
            self.wins += 1
            self.total_profit += pnl
            self.largest_win = max(self.largest_win, pnl)
        else:
            self.losses += 1
            self.total_loss += abs(pnl)
            self.largest_loss = max(self.largest_loss, abs(pnl))

        total = self.wins + self.losses
        self.win_rate = (self.wins / total * 100) if total > 0 else 0.0
        self.avg_win = (self.total_profit / self.wins) if self.wins > 0 else 0.0
        self.avg_loss = (self.total_loss / self.losses) if self.losses > 0 else 0.0
        self.profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else 0.0


performance = PerformanceTracker()

# ---------------------------------------------------------------------
# INDICATEURS
# ---------------------------------------------------------------------


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_vwap(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    return vwap


def detect_order_flow_imbalance(df, window=10):
    buying_pressure = []
    selling_pressure = []

    for i in range(len(df)):
        if i < 1:
            buying_pressure.append(0)
            selling_pressure.append(0)
            continue

        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            buying_pressure.append(df["volume"].iloc[i])
            selling_pressure.append(0)
        else:
            buying_pressure.append(0)
            selling_pressure.append(df["volume"].iloc[i])

    df["buy_pressure"] = pd.Series(buying_pressure).rolling(window).sum()
    df["sell_pressure"] = pd.Series(selling_pressure).rolling(window).sum()
    df["order_flow_delta"] = df["buy_pressure"] - df["sell_pressure"]
    return df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["ATR"] = calculate_atr(df)
        df["RSI"] = calculate_rsi(df["close"])
        df["EMA_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        df["SMA_200"] = df["close"].rolling(window=200).mean()
        df["VWAP"] = calculate_vwap(df)
        df = detect_order_flow_imbalance(df)

        sma20 = df["close"].rolling(window=20).mean()
        std20 = df["close"].rolling(window=20).std()
        df["BB_Upper"] = sma20 + (std20 * 2)
        df["BB_Lower"] = sma20 - (std20 * 2)
        df["BB_Middle"] = sma20

        return df
    except Exception as e:
        logger.error(f"[INDICATORS] Error: {e}")
        return df

# ---------------------------------------------------------------------
# TRAP DETECTION & PSYCHO
# ---------------------------------------------------------------------


def detect_yuichi_trap(
    df_5m: pd.DataFrame,
    df_15m: Optional[pd.DataFrame],
    df_1h: Optional[pd.DataFrame],
    order_book: Optional[dict],
) -> Tuple[Optional[TrapType], float]:
    if df_5m is None or df_5m.empty or len(df_5m) < 50:
        return None, 0.0

    price = df_5m["close"].iloc[-1]
    rsi_5m = df_5m["RSI"].iloc[-1]
    volume = df_5m["volume"].iloc[-1]
    avg_volume = df_5m["volume"].iloc[-30:-1].mean()

    recent_high_5m = df_5m["high"].iloc[-40:-1].max()
    recent_low_5m = df_5m["low"].iloc[-40:-1].min()

    confidence = 0.0
    trap: Optional[TrapType] = None

    # BULL TRAP
    if price > recent_high_5m:
        volume_ratio = volume / avg_volume
        if volume_ratio < 0.8:
            confidence += 30
        if rsi_5m > 70:
            confidence += 25
        if df_15m is not None and len(df_15m) > 5:
            rsi_15m_current = df_15m["RSI"].iloc[-1]
            rsi_15m_prev = df_15m["RSI"].iloc[-3]
            if rsi_15m_current < rsi_15m_prev:
                confidence += 20

        if order_book:
            bids = sum(b[1] for b in order_book.get("bids", [])[:5])
            asks = sum(a[1] for a in order_book.get("asks", [])[:5])
            if asks > bids * 2:
                confidence += 20

        if confidence >= 55:
            trap = TrapType.BULL_TRAP
            logger.warning(f"[TRAP] BULL TRAP detected! Conf={confidence:.0f}%")

    # BEAR TRAP
    elif price < recent_low_5m:
        volume_ratio = volume / avg_volume
        if volume_ratio < 0.8:
            confidence += 30
        if rsi_5m < 30:
            confidence += 25
        if df_15m is not None and len(df_15m) > 5:
            rsi_15m_current = df_15m["RSI"].iloc[-1]
            rsi_15m_prev = df_15m["RSI"].iloc[-3]
            if rsi_15m_current > rsi_15m_prev:
                confidence += 20

        if order_book:
            bids = sum(b[1] for b in order_book.get("bids", [])[:5])
            asks = sum(a[1] for a in order_book.get("asks", [])[:5])
            if bids > asks * 2:
                confidence += 20

        if confidence >= 55:
            trap = TrapType.BEAR_TRAP
            logger.warning(f"[TRAP] BEAR TRAP detected! Conf={confidence:.0f}%")

    # LIQUIDITY GRAB
    if df_5m is not None and len(df_5m) >= 5:
        candle_1 = df_5m.iloc[-5]
        candle_3 = df_5m.iloc[-3]
        candle_now = df_5m.iloc[-1]

        drop = (candle_3["low"] - candle_1["low"]) / candle_1["low"]
        bounce = (candle_now["close"] - candle_3["low"]) / candle_3["low"]

        if drop < -0.01 and bounce > 0.006:
            volume_spike = df_5m["volume"].iloc[-5:-2].max() / avg_volume
            if volume_spike > 1.5:
                confidence = max(confidence, 75)
                trap = TrapType.LIQUIDITY_GRAB
                logger.warning(f"[TRAP] LIQUIDITY GRAB detected! Conf={confidence:.0f}%")

    # RETAIL FOMO / PANIC (ultra agressif)
    if rsi_5m > 80:
        volume_explosion = volume / avg_volume
        candle_range = (df_5m["high"].iloc[-1] - df_5m["low"].iloc[-1]) / df_5m["close"].iloc[-1]
        if volume_explosion > 2.0 and candle_range < 0.003:
            confidence = max(confidence, 80)
            trap = TrapType.RETAIL_FOMO
            logger.warning(f"[TRAP] RETAIL FOMO detected! Conf={confidence:.0f}%")
    elif rsi_5m < 20:
        volume_explosion = volume / avg_volume
        candle_range = (df_5m["high"].iloc[-1] - df_5m["low"].iloc[-1]) / df_5m["close"].iloc[-1]
        if volume_explosion > 2.0 and candle_range < 0.003:
            confidence = max(confidence, 80)
            trap = TrapType.RETAIL_PANIC
            logger.warning(f"[TRAP] RETAIL PANIC detected! Conf={confidence:.0f}%")

    return trap, confidence


def determine_game_state(
    trap: Optional[TrapType],
    trap_confidence: float,
    sentiment: MarketSentiment,
    vol_regime: str,
    df_5m: pd.DataFrame,
    df_1h: Optional[pd.DataFrame],
) -> GameState:
    if trap and trap_confidence >= config.min_trap_confidence:
        if df_1h is not None and not df_1h.empty and len(df_1h) > 50:
            sma50_1h = df_1h["SMA_50"].iloc[-1]
            sma200_1h = df_1h["SMA_200"].iloc[-1]

            if trap == TrapType.BEAR_TRAP and sma50_1h > sma200_1h:
                logger.info("[GAME] GAME OVER: Bear trap in uptrend!")
                return GameState.GAME_OVER

            if trap == TrapType.BULL_TRAP and sma50_1h < sma200_1h:
                logger.info("[GAME] GAME OVER: Bull trap in downtrend!")
                return GameState.GAME_OVER

            if trap == TrapType.RETAIL_PANIC and sma50_1h > sma200_1h:
                logger.info("[GAME] GAME OVER: Retail panic in uptrend!")
                return GameState.GAME_OVER

            if trap == TrapType.RETAIL_FOMO and sma50_1h < sma200_1h:
                logger.info("[GAME] GAME OVER: Retail FOMO in downtrend!")
                return GameState.GAME_OVER

    if trap and trap_confidence >= 65:
        return GameState.SETUP_FORMING

    if trap and trap_confidence >= 55:
        return GameState.TRAP_DETECTED

    if vol_regime in ["extreme", "choppy"]:
        return GameState.UNCERTAIN

    return GameState.OBSERVING


def calculate_yuichi_psychology(multi_tf_data: Dict[str, pd.DataFrame], order_book: Optional[dict]) -> MarketPsychology:
    df_5m = multi_tf_data.get("tactical")
    df_15m = multi_tf_data.get("strategic")
    df_1h = multi_tf_data.get("oversight")

    if df_5m is None or df_5m.empty:
        return MarketPsychology(
            sentiment=MarketSentiment.NEUTRAL,
            fear_greed_index=50.0,
            retail_positioning=50.0,
            smart_money_flow=0.0,
            volatility_regime="medium",
            manipulation_detected=False,
            trap_type=None,
            trap_confidence=0.0,
            game_state=GameState.UNCERTAIN,
        )

    rsi = df_5m["RSI"].iloc[-1]
    atr_pct = (df_5m["ATR"].iloc[-1] / df_5m["close"].iloc[-1]) * 100

    # Sentiment via RSI
    if rsi < 20:
        sentiment = MarketSentiment.EXTREME_FEAR
        fear_greed = 10
    elif rsi < 40:
        sentiment = MarketSentiment.FEAR
        fear_greed = 30
    elif rsi > 80:
        sentiment = MarketSentiment.EXTREME_GREED
        fear_greed = 90
    elif rsi > 60:
        sentiment = MarketSentiment.GREED
        fear_greed = 70
    else:
        sentiment = MarketSentiment.NEUTRAL
        fear_greed = 50

    if atr_pct > 3.0:
        vol_regime = "extreme"
    elif atr_pct > 1.5:
        vol_regime = "high"
    elif atr_pct > 0.5:
        vol_regime = "medium"
    else:
        vol_regime = "low"

    trap, trap_confidence = detect_yuichi_trap(df_5m, df_15m, df_1h, order_book)
    manipulation_detected = trap is not None
    if manipulation_detected:
        sentiment = MarketSentiment.MANIPULATION

    game_state = determine_game_state(trap, trap_confidence, sentiment, vol_regime, df_5m, df_1h)

    order_flow = df_5m["order_flow_delta"].iloc[-1] if "order_flow_delta" in df_5m.columns else 0.0

    return MarketPsychology(
        sentiment=sentiment,
        fear_greed_index=fear_greed,
        retail_positioning=rsi,
        smart_money_flow=order_flow,
        volatility_regime=vol_regime,
        manipulation_detected=manipulation_detected,
        trap_type=trap,
        trap_confidence=trap_confidence,
        game_state=game_state,
    )

# ---------------------------------------------------------------------
# ENTRY LOGIC
# ---------------------------------------------------------------------


def yuichi_entry_signal(multi_tf_data: Dict[str, pd.DataFrame], psychology: MarketPsychology):
    if psychology.game_state == GameState.UNCERTAIN:
        return None, 0.0, "uncertain"

    if psychology.game_state == GameState.OBSERVING:
        return None, 0.0, "observing"

    df = multi_tf_data.get("tactical")
    if df is None or df.empty or len(df) < config.min_observation_candles:
        return None, 0.0, "not_enough_data"

    signal_type: Optional[str] = None
    score = 0.0

    trap = psychology.trap_type
    trap_conf = psychology.trap_confidence

    # Agressif : on donne plus de poids à trap_conf
    if trap == TrapType.BEAR_TRAP:
        signal_type = "buy"
        score = trap_conf / 9.0
        rsi = df["RSI"].iloc[-1]
        if rsi < 30:
            score += 1.5
        if psychology.smart_money_flow > 0:
            score += 1.0

    elif trap == TrapType.BULL_TRAP:
        signal_type = "sell"
        score = trap_conf / 9.0
        rsi = df["RSI"].iloc[-1]
        if rsi > 70:
            score += 1.5
        if psychology.smart_money_flow < 0:
            score += 1.0

    elif trap == TrapType.LIQUIDITY_GRAB:
        df_1h = multi_tf_data.get("oversight")
        if df_1h is not None and not df_1h.empty and len(df_1h) > 50:
            sma50 = df_1h["SMA_50"].iloc[-1]
            sma200 = df_1h["SMA_200"].iloc[-1]
            signal_type = "buy" if sma50 > sma200 else "sell"
            score = trap_conf / 10.0 + 1.5

    elif trap == TrapType.RETAIL_PANIC:
        signal_type = "buy"
        score = trap_conf / 10.0 + 2.0

    elif trap == TrapType.RETAIL_FOMO:
        signal_type = "sell"
        score = trap_conf / 10.0 + 2.0

    if not signal_type or score < config.min_setup_score:
        return None, score, "score_too_low"

    logger.info(
        f"[SIGNAL] {signal_type.upper()} | Score={score:.1f}/10 | "
        f"Trap={trap.value if trap else 'none'} ({trap_conf:.0f}%)"
    )
    return signal_type, score, "ok"

# ---------------------------------------------------------------------
# POSITION SIZING / SL / TP
# ---------------------------------------------------------------------


def calculate_yuichi_position_size(psychology: MarketPsychology, score: float) -> float:
    game_state = psychology.game_state
    base = config.base_positions.get(game_state.value, 0)

    if base == 0:
        return 0.0

    martingale_mult = config.martingale_multipliers[config.current_martingale_level]
    size = base * martingale_mult

    # Ajuste par volatilité
    if psychology.volatility_regime == "extreme":
        size *= 0.6
    elif psychology.volatility_regime == "low":
        size *= 1.3

    # Setup parfait -> bombe nucléaire
    if score >= 9.0:
        size *= 1.4
        logger.info("[SIZE] PERFECT SETUP: size boosted +40%")

    return size


def calculate_smart_sl_tp(entry, df: pd.DataFrame, signal_type: str, psychology: MarketPsychology, score: float):
    atr = df["ATR"].iloc[-1]

    recent_lows = df["low"].iloc[-30:].nsmallest(3).mean()
    recent_highs = df["high"].iloc[-30:].nlargest(3).mean()

    if signal_type == "buy":
        structural_sl = recent_lows * 0.997
        atr_sl = entry - (atr * 1.0)
        stop_loss = max(structural_sl, atr_sl)

        if score >= 9.0:
            tp_mult = 4.2
        elif score >= 8.0:
            tp_mult = 3.4
        elif score >= 7.0:
            tp_mult = 2.8
        else:
            tp_mult = 2.2

        take_profit = entry + (atr * tp_mult)

    else:
        structural_sl = recent_highs * 1.003
        atr_sl = entry + (atr * 1.0)
        stop_loss = min(structural_sl, atr_sl)

        if score >= 9.0:
            tp_mult = 4.2
        elif score >= 8.0:
            tp_mult = 3.4
        elif score >= 7.0:
            tp_mult = 2.8
        else:
            tp_mult = 2.2

        take_profit = entry - (atr * tp_mult)

    return stop_loss, take_profit


def calculate_realistic_pnl(entry, exit_price, size, trade_type):
    entry_fee = size * config.trading_fee_rate
    exit_fee = size * config.trading_fee_rate
    total_fees = entry_fee + exit_fee

    if trade_type == "buy":
        actual_entry = entry * (1 + config.slippage_rate)
        actual_exit = exit_price * (1 - config.slippage_rate)
        qty = size / actual_entry
        gross_pnl = (actual_exit - actual_entry) * qty
    else:
        actual_entry = entry * (1 - config.slippage_rate)
        actual_exit = exit_price * (1 + config.slippage_rate)
        qty = size / actual_entry
        gross_pnl = (actual_entry - actual_exit) * qty

    net_pnl = gross_pnl - total_fees
    return net_pnl, total_fees, gross_pnl

# ---------------------------------------------------------------------
# TRADE STATE
# ---------------------------------------------------------------------


class TradeState:
    def __init__(self):
        self.active = False
        self.type: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.position_size: float = 0.0
        self.score: float = 0.0
        self.game_state: Optional[GameState] = None
        self.trap_type: Optional[TrapType] = None
        self.psychology_snapshot: Optional[MarketPsychology] = None


trade_state = TradeState()
price_history: List[float] = []

# ---------------------------------------------------------------------
# STATUS JSON POUR DASHBOARD
# ---------------------------------------------------------------------


def write_status(extra: Optional[dict] = None):
    """Écrit un JSON de status lisible par ton dashboard Windows."""
    try:
        data = {
            "bot_name": config.bot_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),

            "capital": config.capital,
            "cumulative_winnings": config.cumulative_winnings,
            "battle_step": config.battle_step,
            "martingale_level": config.current_martingale_level,
            "trades_executed_today": config.trades_executed_today,

            "active_trade": trade_state.active,
            "trade_type": trade_state.type,
            "entry_price": trade_state.entry_price,
            "stop_loss": trade_state.stop_loss,
            "take_profit": trade_state.take_profit,
            "position_size": trade_state.position_size,

            "game_state": trade_state.game_state.value if trade_state.game_state else None,
            "trap_type": trade_state.trap_type.value if trade_state.trap_type else None,
        }
        if extra:
            data.update(extra)

        path = os.path.join(config.status_dir, f"{config.bot_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"[STATUS] Write error: {e}")

# ---------------------------------------------------------------------
# LOG HELPERS
# ---------------------------------------------------------------------


def log_result(msg: str):
    try:
        with open("yuichi_v14_aggressive_trades_cli.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {msg}\n")
        logger.info(msg)
    except Exception as e:
        logger.error(f"[LOG] Error: {e}")

# ---------------------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------------------


def fetch_multi_timeframe_data(symbol: str) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for name, tf in config.timeframes.items():
        limit = 200 if name != "macro" else 100
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        if ohlcv:
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            data[name] = df
    return data

# ---------------------------------------------------------------------
# CLOSE TRADE
# ---------------------------------------------------------------------


def close_trade(reason: str, exit_price: float):
    if not trade_state.active:
        return

    net_pnl, fees, gross_pnl = calculate_realistic_pnl(
        trade_state.entry_price,
        exit_price,
        trade_state.position_size,
        trade_state.type,
    )

    config.capital += net_pnl
    config.cumulative_winnings += net_pnl

    if net_pnl > 0:
        config.current_martingale_level = 0
    else:
        config.current_martingale_level = min(
            config.current_martingale_level + 1,
            config.max_martingale_level,
        )
        config.daily_loss += abs(net_pnl)

    result = "WIN" if net_pnl > 0 else "LOSS"
    roi_pct = (net_pnl / trade_state.position_size) * 100 if trade_state.position_size > 0 else 0

    trade_data = {
        "timestamp": datetime.now(),
        "type": trade_state.type,
        "entry": trade_state.entry_price,
        "exit": exit_price,
        "profit_loss": net_pnl,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "result": result,
        "score": trade_state.score,
        "game_state": trade_state.game_state.value if trade_state.game_state else "unknown",
        "trap_type": trade_state.trap_type.value if trade_state.trap_type else "none",
    }
    performance.log_trade(trade_data)

    log_msg = (
        f"\n{'=' * 70}\n"
        f"[CLOSE] {reason} [{result}]\n"
        f"{'=' * 70}\n"
        f"Battle step: {config.battle_step} | Martingale level: {config.current_martingale_level}\n"
        f"Entry: ${trade_state.entry_price:.2f} -> Exit: ${exit_price:.2f}\n"
        f"Position: ${trade_state.position_size:.2f}\n"
        f"SL: ${trade_state.stop_loss:.2f} | TP: ${trade_state.take_profit:.2f}\n"
        f"Gross P/L: ${gross_pnl:.2f}\n"
        f"Fees: -${fees:.2f}\n"
        f"Net P/L: ${net_pnl:.2f} ({roi_pct:+.2f}%)\n"
        f"Capital: ${config.capital:.2f}\n"
        f"Cumulative winnings: ${config.cumulative_winnings:.2f}\n"
        f"WinRate: {performance.win_rate:.1f}% | PF: {performance.profit_factor:.2f}\n"
        f"{'=' * 70}\n"
    )
    log_result(log_msg)

    trade_state.active = False
    config.trades_executed_today += 1
    config.last_trade_time = datetime.now()

    write_status({"state": "flat"})

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------


def execute_yuichi_strategy():
    logger.info("======================================================")
    logger.info("[YUICHI-TRUE] - INIT Yuichi True Method v14 AGGRESSIVE (CLI)")
    logger.info(f"[YUICHI-TRUE] - Starting capital: {config.capital:.2f}")
    logger.info(
        f"[YUICHI-TRUE] - Status JSON: {os.path.join(config.status_dir, config.bot_name + '.json')}"
    )
    logger.info("======================================================")

    while config.running:
        try:
            # Guards
            if (
                config.trades_executed_today >= config.max_trades_per_day
                or config.daily_loss >= config.capital * config.max_daily_loss_pct
                or config.current_martingale_level >= config.max_martingale_level
            ):
                log_result("[LIMIT] Daily limits reached, bot on cooldown.")
                write_status({"state": "cooldown"})
                time.sleep(60)
                continue

            if config.last_trade_time:
                elapsed = (datetime.now() - config.last_trade_time).total_seconds() / 60.0
                if elapsed < config.min_time_between_trades:
                    write_status({"state": "cooldown_wait"})
                    time.sleep(10)
                    continue

            multi_tf_data = fetch_multi_timeframe_data(config.symbols[0])
            df = multi_tf_data.get("tactical")
            if df is None or df.empty or len(df) < config.min_observation_candles:
                write_status({"state": "waiting_data"})
                time.sleep(5)
                continue

            for name in list(multi_tf_data.keys()):
                multi_tf_data[name] = calculate_all_indicators(multi_tf_data[name])

            df = multi_tf_data["tactical"]
            current_price = df["close"].iloc[-1]

            price_history.append(current_price)
            if len(price_history) > 500:
                price_history.pop(0)

            order_book = exchange.fetch_order_book(config.symbols[0], limit=20)
            psychology = calculate_yuichi_psychology(multi_tf_data, order_book)

            # SANS POSITION
            if not trade_state.active:
                signal_type, score, reason = yuichi_entry_signal(multi_tf_data, psychology)

                if not signal_type:
                    logger.info(
                        f"[STATE] observing | price={current_price:.2f} "
                        f"| battle={config.battle_step} "
                        f"| martingale={config.current_martingale_level} "
                        f"| cumu=${config.cumulative_winnings:.2f} | reason={reason}"
                    )
                    write_status({"state": "observing"})
                    time.sleep(5)
                    continue

                atr = df["ATR"].iloc[-1]
                expected_profit_pct = (atr * 2.5) / current_price
                if expected_profit_pct < config.min_profit_threshold:
                    logger.info(
                        f"[SKIP] Expected profit {expected_profit_pct:.2%} < threshold"
                    )
                    write_status({"state": "low_edge"})
                    time.sleep(5)
                    continue

                position_size = calculate_yuichi_position_size(psychology, score)
                if position_size <= 0:
                    write_status({"state": "zero_size"})
                    time.sleep(5)
                    continue

                trade_state.active = True
                trade_state.type = signal_type
                trade_state.entry_price = current_price
                trade_state.entry_time = datetime.now()
                trade_state.position_size = position_size
                trade_state.score = score
                trade_state.game_state = psychology.game_state
                trade_state.trap_type = psychology.trap_type
                trade_state.psychology_snapshot = psychology

                sl, tp = calculate_smart_sl_tp(current_price, df, signal_type, psychology, score)
                trade_state.stop_loss = sl
                trade_state.take_profit = tp

                config.battle_step += 1

                log_msg = (
                    f"\n{'=' * 70}\n"
                    f"[ENTER] YUICHI v14 AGGRESSIVE: {signal_type.upper()}\n"
                    f"{'=' * 70}\n"
                    f"Battle step: {config.battle_step} | Martingale level: {config.current_martingale_level}\n"
                    f"Game State: {psychology.game_state.value.upper()}\n"
                    f"Trap: {psychology.trap_type.value if psychology.trap_type else 'none'} "
                    f"({psychology.trap_confidence:.0f}%)\n"
                    f"Score: {score:.1f}/10\n"
                    f"Entry: ${current_price:.2f}\n"
                    f"Position: ${position_size:.2f}\n"
                    f"SL: ${sl:.2f} | TP: ${tp:.2f}\n"
                    f"Cumulative winnings: ${config.cumulative_winnings:.2f}\n"
                    f"{'=' * 70}"
                )
                log_result(log_msg)
                write_status({"state": "in_position"})

            # AVEC POSITION
            else:
                if trade_state.type == "buy" and current_price >= trade_state.take_profit:
                    close_trade("TAKE PROFIT", trade_state.take_profit)
                elif trade_state.type == "sell" and current_price <= trade_state.take_profit:
                    close_trade("TAKE PROFIT", trade_state.take_profit)
                elif trade_state.type == "buy" and current_price <= trade_state.stop_loss:
                    close_trade("STOP LOSS", trade_state.stop_loss)
                elif trade_state.type == "sell" and current_price >= trade_state.stop_loss:
                    close_trade("STOP LOSS", trade_state.stop_loss)
                else:
                    logger.info(
                        f"[POSITION] {trade_state.type.upper()} | "
                        f"entry={trade_state.entry_price:.2f} | "
                        f"SL={trade_state.stop_loss:.2f} | TP={trade_state.take_profit:.2f} | "
                        f"price={current_price:.2f} | "
                        f"battle={config.battle_step} | "
                        f"martingale={config.current_martingale_level} | "
                        f"cumu=${config.cumulative_winnings:.2f}"
                    )
                    write_status({"state": "in_position"})
                    time.sleep(5)
                    continue

            time.sleep(3)

        except Exception as e:
            logger.error(f"[LOOP] Error: {e}", exc_info=True)
            write_status({"last_error": str(e)})
            time.sleep(5)

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------


if __name__ == "__main__":
    execute_yuichi_strategy()
