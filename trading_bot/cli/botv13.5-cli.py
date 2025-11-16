# Trading Bot v13.5 - "Enhanced Method"
# v13 AMÉLIORÉE avec tous les axes d'amélioration
# GARDE la philosophie multi-confirmation MAIS optimise tout le reste

# import tkinter as tk  # removed for CLI
# from tkinter import scrolledtext  # removed for CLI
import threading
import ccxt
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  # removed for CLI
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # removed for CLI
import logging
import time
from datetime import datetime
from collections import deque
import sys
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import os
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - [YUICHI v13.5] - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("yuichi_v13_5.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------

class MarketCondition(Enum):
    """Qualité du marché (régime)"""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE_CLEAN = "range_clean"
    RANGE_CHOPPY = "range_choppy"
    VOLATILE_CLEAN = "volatile_clean"
    VOLATILE_EXTREME = "volatile_extreme"


class TrapType(Enum):
    """Pièges de marché typiques (liquidité / manipulation)"""
    BULL_TRAP = "bull_trap"
    BEAR_TRAP = "bear_trap"
    LIQUIDITY_GRAB = "liquidity_grab"
    STOP_HUNT = "stop_hunt"
    FAKE_BREAKOUT = "fake_breakout"
    NONE = "none"


@dataclass
class MarketPsychology:
    sentiment: str                    # bull, bear, neutral, fear, greed
    fear_greed_index: float           # 0-100
    retail_positioning: float         # % long / short retail (proxy)
    smart_money_flow: float           # flux des "whales"
    volatility_regime: str            # low / medium / high / extreme
    manipulation_detected: bool
    trap_type: TrapType
    trap_probability: float           # 0-1
    regime_confidence: float          # 0-1


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

class YuichiConfig:
    symbols = ["BTC/USDT"]
    timeframe = "1m"
    capital = 1000.0

    bot_name = "yuichi_v13_5_cli"
    status_dir = "status"

    timeframes = {
        "micro": "1m",
        "tactical": "5m",
        "strategic": "15m",
        "oversight": "1h",
    }

    # ✅ NOUVEAU: Frais réels
    trading_fee_rate = 0.001   # 0.1% Binance
    slippage_rate = 0.0003     # 0.03% slippage
    min_profit_threshold = 0.004  # 0.4% min pour couvrir frais

    # ✅ NOUVEAU: Limites journalières plus strictes
    max_daily_loss_pct = 0.15
    max_trades_per_day = 20

    # ✅ NOUVEAU: Position sizing dynamique
    base_position_pct = 0.05       # 5% du capital
    max_position_pct = 0.25        # 25% max

    # Martingale (améliorée, plus contrôlée)
    martingale_steps = [1.0, 1.5, 2.5, 4.0]
    current_step = 0
    max_martingale_steps = 3

    # "Battle steps" façon Yuichi (progression psychologique)
    battle_steps = ["probing", "testing", "punishing", "all_in"]
    battle_step = 0

    # ✅ NOUVEAU: Système de scoring au lieu de simple count
    min_setup_score = 4.0  # Score minimum pour entrer
    
    # ✅ NOUVEAU: Cooldown entre trades
    min_time_between_trades = 15  # minutes
    last_trade_time = None

    # RSI settings
    rsi_period = 14
    rsi_extreme_oversold = 30
    rsi_extreme_overbought = 80

    # Features
    fake_signal_filter = True
    whale_detection = True
    retail_sentiment = True
    market_regime_adaptation = True
    psychological_edge_mode = True
    counter_manipulation = True

    # ✅ SUPPRIMÉ: Hedge betting (simplifie)
    # Plus de hedge = plus de profit conservé

    max_hold_time_minutes = 30

    trades_executed_today = 0
    daily_loss = 0.0
    cumulative_winnings = 0.0
    running = True

    opponent_patterns: Dict = {}
    fake_signal_history = deque(maxlen=100)
    manipulation_attempts = 0


config = YuichiConfig()
os.makedirs(config.status_dir, exist_ok=True)

# ---------------------------------------------------------------------
# EXCHANGE
# ---------------------------------------------------------------------

class EliteExchange:
    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.rate_limit_buffer = 1.2

    def fetch_with_retry(self, func, *args, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded:
                wait_time = (2 ** attempt) * self.rate_limit_buffer
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error fetching data (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        return None

    def fetch_ohlcv(self, symbol, timeframe, limit=200):
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
        self.gold_setups = 0
        self.gold_wins = 0
        self.win_rate = 0.0

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

        if trade_data.get("is_gold_setup"):
            self.gold_setups += 1
            if pnl > 0:
                self.gold_wins += 1


performance = PerformanceTracker()

# ---------------------------------------------------------------------
# INDICATORS
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

        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            buying_pressure.append(df["volume"].iloc[i])
            selling_pressure.append(0)
        else:
            buying_pressure.append(0)
            selling_pressure.append(df["volume"].iloc[i])

    df["buy_pressure"] = pd.Series(buying_pressure).rolling(window).sum()
    df["sell_pressure"] = pd.Series(selling_pressure).rolling(window).sum()
    df["order_flow_delta"] = df["buy_pressure"] - df["sell_pressure"]
    return df


def calculate_all_indicators(df):
    try:
        df["ATR"] = calculate_atr(df)
        df["RSI"] = calculate_rsi(df["close"], period=config.rsi_period)
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
        logger.error(f"Indicator error: {e}")
        return df

# ---------------------------------------------------------------------
# MARKET REGIME & PSYCHOLOGY
# ---------------------------------------------------------------------

def detect_market_condition(df_5m, df_1h) -> MarketCondition:
    if df_5m is None or df_5m.empty:
        return MarketCondition.RANGE_CHOPPY

    price = df_5m["close"].iloc[-1]
    atr = df_5m["ATR"].iloc[-1] if "ATR" in df_5m else 0
    atr_pct = (atr / price) * 100 if price > 0 else 0

    if df_1h is not None and not df_1h.empty and "SMA_50" in df_1h and "SMA_200" in df_1h:
        sma50 = df_1h["SMA_50"].iloc[-1]
        sma200 = df_1h["SMA_200"].iloc[-1]
        trend_up = sma50 > sma200 * 1.003
        trend_down = sma50 < sma200 * 0.997
    else:
        trend_up = trend_down = False

    if atr_pct < 0.5:
        if trend_up:
            return MarketCondition.TREND_UP
        elif trend_down:
            return MarketCondition.TREND_DOWN
        else:
            return MarketCondition.RANGE_CLEAN
    elif atr_pct < 1.5:
        if trend_up:
            return MarketCondition.TREND_UP
        elif trend_down:
            return MarketCondition.TREND_DOWN
        else:
            return MarketCondition.RANGE_CHOPPY
    elif atr_pct < 3.0:
        if trend_up or trend_down:
            return MarketCondition.VOLATILE_CLEAN
        else:
            return MarketCondition.RANGE_CHOPPY
    else:
        return MarketCondition.VOLATILE_EXTREME


def detect_yuichi_trap(df_5m, df_15m, df_1h, order_book) -> (TrapType, float):
    if df_5m is None or df_5m.empty or len(df_5m) < 50:
        return TrapType.NONE, 0.0

    price = df_5m["close"].iloc[-1]
    rsi_5m = df_5m["RSI"].iloc[-1]
    volume = df_5m["volume"].iloc[-1]
    avg_volume = df_5m["volume"].iloc[-20:-1].mean()

    recent_high = df_5m["high"].iloc[-30:-1].max()
    recent_low = df_5m["low"].iloc[-30:-1].min()

    confidence = 0.0
    trap = TrapType.NONE

    if price > recent_high:
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio < 0.8:
            confidence += 0.3
        if rsi_5m > 70:
            confidence += 0.3
        if df_15m is not None and len(df_15m) > 5:
            rsi_15m_current = df_15m["RSI"].iloc[-1]
            rsi_15m_prev = df_15m["RSI"].iloc[-3]
            if rsi_15m_current < rsi_15m_prev:
                confidence += 0.2
        if order_book:
            bids = sum(b[1] for b in order_book.get('bids', [])[:5])
            asks = sum(a[1] for a in order_book.get('asks', [])[:5])
            if asks > bids * 2:
                confidence += 0.2
        if confidence >= 0.5:
            trap = TrapType.BULL_TRAP
            logger.warning(f"🪤 BULL TRAP detected with probability {confidence:.1%}")

    elif price < recent_low:
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio < 0.8:
            confidence += 0.3
        if rsi_5m < 30:
            confidence += 0.3
        if df_15m is not None and len(df_15m) > 5:
            rsi_15m_current = df_15m["RSI"].iloc[-1]
            rsi_15m_prev = df_15m["RSI"].iloc[-3]
            if rsi_15m_current > rsi_15m_prev:
                confidence += 0.2
        if order_book:
            bids = sum(b[1] for b in order_book.get('bids', [])[:5])
            asks = sum(a[1] for a in order_book.get('asks', [])[:5])
            if bids > asks * 2:
                confidence += 0.2
        if confidence >= 0.5:
            trap = TrapType.BEAR_TRAP
            logger.warning(f"🪤 BEAR TRAP detected with probability {confidence:.1%}")

    return trap, confidence


def calculate_yuichi_psychology(multi_tf_data, order_book) -> MarketPsychology:
    df_5m = multi_tf_data.get("tactical")
    df_15m = multi_tf_data.get("strategic")
    df_1h = multi_tf_data.get("oversight")

    if df_5m is None or df_5m.empty:
        return MarketPsychology(
            sentiment="neutral",
            fear_greed_index=50.0,
            retail_positioning=50.0,
            smart_money_flow=0.0,
            volatility_regime="medium",
            manipulation_detected=False,
            trap_type=TrapType.NONE,
            trap_probability=0.0,
            regime_confidence=0.5
        )

    rsi = df_5m["RSI"].iloc[-1]
    atr = df_5m["ATR"].iloc[-1]
    price = df_5m["close"].iloc[-1]
    atr_pct = (atr / price) * 100 if price > 0 else 0

    if rsi < 25:
        sentiment = "extreme_fear"
        fear_greed = 10
    elif rsi < 40:
        sentiment = "fear"
        fear_greed = 30
    elif rsi > 80:
        sentiment = "extreme_greed"
        fear_greed = 90
    elif rsi > 60:
        sentiment = "greed"
        fear_greed = 70
    else:
        sentiment = "neutral"
        fear_greed = 50

    if atr_pct < 0.5:
        vol_regime = "low"
    elif atr_pct < 1.5:
        vol_regime = "medium"
    elif atr_pct < 3.0:
        vol_regime = "high"
    else:
        vol_regime = "extreme"

    df_5m = df_5m if "order_flow_delta" in df_5m.columns else detect_order_flow_imbalance(df_5m)
    order_flow = df_5m["order_flow_delta"].iloc[-1]

    trap, trap_prob = detect_yuichi_trap(df_5m, df_15m, df_1h, order_book)

    manipulation_detected = trap != TrapType.NONE
    regime_confidence = 0.5
    if manipulation_detected:
        regime_confidence = 0.7
        sentiment = "manipulation"

    return MarketPsychology(
        sentiment=sentiment,
        fear_greed_index=fear_greed,
        retail_positioning=rsi,
        smart_money_flow=order_flow,
        volatility_regime=vol_regime,
        manipulation_detected=manipulation_detected,
        trap_type=trap,
        trap_probability=trap_prob,
        regime_confidence=regime_confidence
    )

# ---------------------------------------------------------------------
# ENTRY LOGIC (SCORING)
# ---------------------------------------------------------------------

def score_entry_setup(df, psychology: MarketPsychology, market_condition: MarketCondition) -> (float, list):
    score = 0.0
    confirmations = []

    rsi = df["RSI"].iloc[-1]
    price = df["close"].iloc[-1]
    ema9 = df["EMA_9"].iloc[-1]
    ema21 = df["EMA_21"].iloc[-1]
    vwap = df["VWAP"].iloc[-1]
    bb_upper = df["BB_Upper"].iloc[-1]
    bb_lower = df["BB_Lower"].iloc[-1]

    if psychology.sentiment in ["extreme_fear", "extreme_greed"]:
        score += 1.0
        confirmations.append("sentiment_extreme")

    if psychology.trap_type in [TrapType.BULL_TRAP, TrapType.BEAR_TRAP] and psychology.trap_probability >= 0.6:
        score += 1.5
        confirmations.append("trap_high_prob")

    if psychology.volatility_regime in ["medium", "high"] and market_condition not in [
        MarketCondition.RANGE_CHOPPY,
        MarketCondition.VOLATILE_EXTREME
    ]:
        score += 1.0
        confirmations.append("good_volatility")

    if price > ema9 > ema21 and price > vwap:
        score += 0.7
        confirmations.append("micro_trend_up")
    elif price < ema9 < ema21 and price < vwap:
        score += 0.7
        confirmations.append("micro_trend_down")

    if psychology.trap_type == TrapType.BEAR_TRAP and psychology.sentiment in ["fear", "extreme_fear"]:
        score += 1.0
        confirmations.append("bear_trap_fear")
    if psychology.trap_type == TrapType.BULL_TRAP and psychology.sentiment in ["greed", "extreme_greed"]:
        score += 1.0
        confirmations.append("bull_trap_greed")

    near_bb = False
    if abs(price - bb_lower) / price < 0.003:
        score += 0.7
        confirmations.append("near_lower_band")
        near_bb = True
    if abs(price - bb_upper) / price < 0.003:
        score += 0.7
        confirmations.append("near_upper_band")
        near_bb = True

    if not near_bb:
        score -= 0.5
        confirmations.append("not_near_band")

    if psychology.manipulation_detected:
        if psychology.trap_probability >= 0.7:
            score += 0.8
            confirmations.append("manipulation_detected_strong")
        else:
            score += 0.4
            confirmations.append("manipulation_detected_weak")

    if market_condition in [MarketCondition.RANGE_CHOPPY, MarketCondition.VOLATILE_EXTREME]:
        score -= 1.0
        confirmations.append("bad_market_condition")

    return score, confirmations


def yuichi_entry_signal(multi_tf_data, psychology: MarketPsychology, market_condition: MarketCondition):
    df = multi_tf_data.get("tactical")
    if df is None or df.empty or len(df) < 80:
        return None, 0.0, [], "Not enough data"

    score, confirmations = score_entry_setup(df, psychology, market_condition)
    if score < config.min_setup_score:
        return None, score, confirmations, "Score too low"

    price = df["close"].iloc[-1]
    ema9 = df["EMA_9"].iloc[-1]
    ema21 = df["EMA_21"].iloc[-1]
    vwap = df["VWAP"].iloc[-1]
    bb_upper = df["BB_Upper"].iloc[-1]
    bb_lower = df["BB_Lower"].iloc[-1]

    trade_type = None

    if psychology.trap_type == TrapType.BEAR_TRAP and price > vwap and price < ema9:
        trade_type = "buy"
    elif psychology.trap_type == TrapType.BULL_TRAP and price < vwap and price > ema9:
        trade_type = "sell"
    else:
        if psychology.sentiment in ["fear", "extreme_fear"] and price < bb_lower and price < vwap:
            trade_type = "buy"
        elif psychology.sentiment in ["greed", "extreme_greed"] and price > bb_upper and price > vwap:
            trade_type = "sell"
        else:
            if "micro_trend_up" in confirmations and "near_lower_band" in confirmations:
                trade_type = "buy"
            elif "micro_trend_down" in confirmations and "near_upper_band" in confirmations:
                trade_type = "sell"

    if not trade_type:
        return None, score, confirmations, "No clear direction"

    logger.info(f"🎯 YUICHI SIGNAL: {trade_type.upper()} | Score: {score:.2f} | Confirmations: {confirmations}")
    return trade_type, score, confirmations, "Valid setup"

# ---------------------------------------------------------------------
# POSITION SIZING & TP/SL
# ---------------------------------------------------------------------

def calculate_position_size(psychology: MarketPsychology, score: float) -> float:
    risk_factor = min(max(score / 10.0, 0.5), 1.5)

    if psychology.sentiment in ["extreme_fear", "extreme_greed"]:
        risk_factor *= 1.2

    if psychology.volatility_regime in ["high", "extreme"]:
        risk_factor *= 0.8

    if psychology.trap_type in [TrapType.BULL_TRAP, TrapType.BEAR_TRAP] and psychology.trap_probability >= 0.7:
        risk_factor *= 1.3

    martingale_mult = config.martingale_steps[config.current_step]

    base_size = config.capital * config.base_position_pct
    position_size = base_size * risk_factor * martingale_mult

    max_size = config.capital * config.max_position_pct
    position_size = min(position_size, max_size)

    return position_size


def calculate_adaptive_tp_mult(confirmations: list, psychology: MarketPsychology, score: float) -> float:
    tp_mult = 2.0

    if "trap_high_prob" in confirmations:
        tp_mult += 0.5
    if "sentiment_extreme" in confirmations:
        tp_mult += 0.3

    if score >= 6.0:
        tp_mult += 0.5

    if psychology.volatility_regime in ["low", "medium"]:
        tp_mult = min(tp_mult, 3.0)
    else:
        tp_mult = min(tp_mult, 2.5)

    return tp_mult


def calculate_smart_sl_tp(entry, df, trade_type, psychology: MarketPsychology, score: float, confirmations: list):
    atr = df["ATR"].iloc[-1]
    price = df["close"].iloc[-1]

    recent_lows = df["low"].iloc[-30:].nsmallest(3).mean()
    recent_highs = df["high"].iloc[-30:].nlargest(3).mean()

    tp_mult = calculate_adaptive_tp_mult(confirmations, psychology, score)

    if trade_type == "buy":
        structural_sl = recent_lows * 0.997
        atr_sl = entry - (atr * 1.2)
        stop_loss = max(structural_sl, atr_sl)
        take_profit = entry + (atr * tp_mult)
    else:
        structural_sl = recent_highs * 1.003
        atr_sl = entry + (atr * 1.2)
        stop_loss = min(structural_sl, atr_sl)
        take_profit = entry - (atr * tp_mult)

    return stop_loss, take_profit

# ---------------------------------------------------------------------
# REALISTIC PNL
# ---------------------------------------------------------------------

def calculate_realistic_pnl(entry, exit, size, trade_type):
    entry_fee = size * config.trading_fee_rate
    exit_fee = size * config.trading_fee_rate
    total_fees = entry_fee + exit_fee

    if trade_type == "buy":
        actual_entry = entry * (1 + config.slippage_rate)
        actual_exit = exit * (1 - config.slippage_rate)
        qty = size / actual_entry
        gross_pnl = (actual_exit - actual_entry) * qty
    else:
        actual_entry = entry * (1 - config.slippage_rate)
        actual_exit = exit * (1 + config.slippage_rate)
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
        self.type = None
        self.entry_price = None
        self.entry_time = None
        self.stop_loss = None
        self.take_profit = None
        self.position_size = 0.0
        self.score = 0.0
        self.confirmations = []
        self.psychology_snapshot = None


trade_state = TradeState()

def write_status(extra: dict | None = None):
    """Write current state to a JSON file for external monitoring (Windows app)."""
    try:
        data = {
            "bot_name": config.bot_name,
            "capital": getattr(config, "capital", None),
            "cumulative_winnings": getattr(config, "cumulative_winnings", 0.0),
            "daily_loss": getattr(config, "daily_loss", 0.0),
            "trades_executed_today": getattr(config, "trades_executed_today", 0),
            "martingale_step": getattr(config, "current_step", 0),
            "battle_step": getattr(config, "battle_step", 0),
            "running": getattr(config, "running", True),

            "position_active": getattr(trade_state, "active", False),
            "side": getattr(trade_state, "type", None),
            "entry_price": getattr(trade_state, "entry_price", None),
            "stop_loss": getattr(trade_state, "stop_loss", None),
            "take_profit": getattr(trade_state, "take_profit", None),
            "position_size": getattr(trade_state, "position_size", 0.0),
            "score": getattr(trade_state, "score", 0.0),
            "confirmations": getattr(trade_state, "confirmations", []),

            "last_update": datetime.utcnow().isoformat() + "Z",
        }
        if extra:
            data.update(extra)

        path = os.path.join(config.status_dir, f"{config.bot_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Status write error: {e}")

price_history = []

# ---------------------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------------------

def fetch_multi_timeframe_data(symbol):
    data = {}
    for name, tf in config.timeframes.items():
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=200)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            data[name] = df
    return data

# ---------------------------------------------------------------------
# CLOSE TRADE
# ---------------------------------------------------------------------

def close_trade(reason, exit_price, log_box, winnings_var, stats_box):
    if not trade_state.active:
        return
    
    net_pnl, fees, gross_pnl = calculate_realistic_pnl(
        trade_state.entry_price,
        exit_price,
        trade_state.position_size,
        trade_state.type
    )
    
    config.capital += net_pnl
    config.cumulative_winnings += net_pnl
    
    # Martingale
    if net_pnl > 0:
        config.battle_step = 0
        config.current_step = 0
    else:
        config.battle_step = min(config.battle_step + 1, len(config.battle_steps) - 1)
        config.current_step = min(config.current_step + 1, len(config.martingale_steps) - 1)
        config.daily_loss += abs(net_pnl)
    
    result = "WIN" if net_pnl > 0 else "LOSS"
    
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
        "confirmations": trade_state.confirmations,
        "is_gold_setup": trade_state.score >= 6.0
    }
    performance.log_trade(trade_data)
    
    roi_pct = (net_pnl / trade_state.position_size) * 100 if trade_state.position_size > 0 else 0
    
    log_msg = (
        f"\n{'='*70}\n"
        f"{reason} [{result}]\n"
        f"{'='*70}\n"
        f"Score: {trade_state.score:.1f}\n"
        f"Entry: ${trade_state.entry_price:.2f} -> Exit: ${exit_price:.2f}\n"
        f"Position: ${trade_state.position_size:.2f}\n"
        f"Gross: ${gross_pnl:.2f} | Fees: -${fees:.2f} | Net: ${net_pnl:.2f} ({roi_pct:+.2f}%)\n"
        f"Capital: ${config.capital:.2f} | Winnings: ${config.cumulative_winnings:.2f}\n"
        f"Win Rate: {performance.win_rate:.1f}%\n"
        f"{'='*70}\n"
    )
    
    log_result(log_msg, log_box)
    
    trade_state.active = False
    config.trades_executed_today += 1
    config.last_trade_time = datetime.now()
    
    update_winnings_box(winnings_var)
    update_stats_panel(stats_box)
    write_status({
        "last_trade_reason": reason,
        "last_trade_pnl": net_pnl,
        "last_trade_result": result,
    })

# ---------------------------------------------------------------------
# GUI HELPERS (still there but unused in CLI)
# ---------------------------------------------------------------------

def log_result(msg, log_box=None):
    try:
        with open("yuichi_v13_5_trades.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {msg}\n")
        if log_box:
            log_box.config(state=tk.NORMAL)
            log_box.insert(tk.END, f"{msg}\n")
            log_box.see(tk.END)
            log_box.config(state=tk.DISABLED)
    except Exception as e:
        logger.error(f"Log error: {e}")


def update_winnings_box(entry_var=None):
    try:
        entry_var.set(f"${config.cumulative_winnings:.2f}")
    except Exception:
        pass


def update_stats_panel(box=None):
    try:
        if box:
            box.config(state=tk.NORMAL)
            box.delete(1.0, tk.END)
            box.insert(tk.END, "📊 PERFORMANCE\n" + "=" * 30 + "\n\n")
            box.insert(tk.END, f"Wins: {performance.wins}\n")
            box.insert(tk.END, f"Losses: {performance.losses}\n")
            box.insert(tk.END, f"Win Rate: {performance.win_rate:.1f}%\n\n")
            box.insert(tk.END, f"Profit: ${performance.total_profit:.2f}\n")
            box.insert(tk.END, f"Loss: ${performance.total_loss:.2f}\n")
            box.insert(tk.END, f"Net: ${performance.total_profit - performance.total_loss:.2f}\n\n")
            box.insert(tk.END, f"💎 Gold Setups: {performance.gold_setups}\n")
            box.insert(tk.END, f"   Wins: {performance.gold_wins}\n")
            if performance.gold_setups > 0:
                gold_wr = (performance.gold_wins / performance.gold_setups) * 100
                box.insert(tk.END, f"   WR: {gold_wr:.1f}%\n")
            box.config(state=tk.DISABLED)
    except Exception:
        pass


def update_psychology_panel(box, psychology, market_condition):
    try:
        if box and psychology:
            box.config(state=tk.NORMAL)
            box.delete(1.0, tk.END)
            box.insert(tk.END, "🧠 PSYCHOLOGY\n" + "=" * 30 + "\n\n")
            box.insert(tk.END, f"Sentiment: {psychology.sentiment.value.upper()}\n")
            box.insert(tk.END, f"F&G: {psychology.fear_greed_index:.0f}/100\n")
            box.insert(tk.END, f"Volatility: {psychology.volatility_regime}\n\n")
            box.insert(tk.END, f"Market: {market_condition.value.upper()}\n")
            if psychology.manipulation_detected:
                box.insert(tk.END, f"\n🎭 MANIPULATION\n")
                box.insert(tk.END, f"Trap: {psychology.trap_probability:.1%}\n")
            box.config(state=tk.DISABLED)
    except Exception:
        pass


def update_chart(ax, canvas):
    # CLI version: no chart rendering
    return

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------

def execute_yuichi_strategy(log_box=None, ax=None, canvas=None, winnings_var=None, stats_box=None, psych_box=None):
    logger.info("=" * 70)
    logger.info("💎 ENHANCED METHOD v13.5 💎")
    logger.info("Scoring + Market Conditions + Adaptive TP + No Hedge")
    logger.info("=" * 70)
    
    while config.running:
        try:
            if (config.trades_executed_today >= config.max_trades_per_day or
                config.daily_loss >= config.capital * config.max_daily_loss_pct or
                config.current_step >= config.max_martingale_steps):
                log_result("Daily limits reached.", log_box)
                write_status()
                time.sleep(60)
                continue
            
            if config.last_trade_time:
                elapsed = (datetime.now() - config.last_trade_time).total_seconds() / 60.0
                if elapsed < config.min_time_between_trades:
                    write_status()
                    time.sleep(10)
                    continue
            
            multi_tf_data = fetch_multi_timeframe_data(config.symbols[0])
            df = multi_tf_data.get("tactical")
            if not df or len(df) < 50:
                write_status()
                time.sleep(5)
                continue
            
            for name in multi_tf_data:
                multi_tf_data[name] = calculate_all_indicators(multi_tf_data[name])
            
            df = multi_tf_data["tactical"]
            current_price = df["close"].iloc[-1]
            
            price_history.append(current_price)
            if len(price_history) > 500:
                price_history.pop(0)
            
            df_1h = multi_tf_data.get("oversight")
            market_condition = detect_market_condition(df, df_1h)
            
            order_book = exchange.fetch_order_book(config.symbols[0], limit=20)
            psychology = calculate_yuichi_psychology(multi_tf_data, order_book)
            
            update_psychology_panel(psych_box, psychology, market_condition)
            
            if market_condition in [MarketCondition.RANGE_CHOPPY, MarketCondition.VOLATILE_EXTREME]:
                write_status()
                time.sleep(10)
                continue
            
            if not trade_state.active:
                signal_type, score, confirmations, reason = yuichi_entry_signal(
                    multi_tf_data, psychology, market_condition
                )
                
                if not signal_type:
                    write_status()
                    time.sleep(5)
                    continue
                
                atr = df["ATR"].iloc[-1]
                tp_mult = calculate_adaptive_tp_mult(confirmations, psychology, score)
                expected_profit_pct = (atr * tp_mult) / current_price
                
                if expected_profit_pct < config.min_profit_threshold:
                    logger.info(f"⚠️ Expected profit {expected_profit_pct:.2%} < {config.min_profit_threshold:.2%}, skip")
                    write_status()
                    time.sleep(5)
                    continue
                
                position_size = calculate_position_size(psychology, score)
                
                trade_state.active = True
                trade_state.type = signal_type
                trade_state.entry_price = current_price
                trade_state.entry_time = datetime.now()
                trade_state.position_size = position_size
                trade_state.score = score
                trade_state.confirmations = confirmations
                trade_state.psychology_snapshot = psychology
                
                sl, tp = calculate_smart_sl_tp(current_price, df, signal_type, psychology, score, confirmations)
                trade_state.stop_loss = sl
                trade_state.take_profit = tp
                
                log_msg = (
                    f"\n{'='*70}\n"
                    f"🎯 YUICHI ENTERS: {signal_type.upper()}\n"
                    f"{'='*70}\n"
                    f"Score: {score:.1f} | Confirmations: {confirmations}\n"
                    f"Entry: ${current_price:.2f}\n"
                    f"Position: ${position_size:.2f}\n"
                    f"SL: ${sl:.2f} | TP: ${tp:.2f}\n"
                    f"{'='*70}"
                )
                log_result(log_msg, log_box)
                write_status()
            
            else:
                if trade_state.type == "buy" and current_price >= trade_state.take_profit:
                    close_trade("TAKE PROFIT", trade_state.take_profit, log_box, winnings_var, stats_box)
                elif trade_state.type == "sell" and current_price <= trade_state.take_profit:
                    close_trade("TAKE PROFIT", trade_state.take_profit, log_box, winnings_var, stats_box)
                elif trade_state.type == "buy" and current_price <= trade_state.stop_loss:
                    close_trade("STOP LOSS", trade_state.stop_loss, log_box, winnings_var, stats_box)
                elif trade_state.type == "sell" and current_price >= trade_state.stop_loss:
                    close_trade("STOP LOSS", trade_state.stop_loss, log_box, winnings_var, stats_box)
                else:
                    write_status()
            
            update_stats_panel(stats_box)
            update_chart(ax, canvas)
            write_status()
            time.sleep(3)
        
        except Exception as e:
            logger.error(f"Loop error: {e}", exc_info=True)
            write_status({"loop_error": str(e)})
            time.sleep(5)

# ---------------------------------------------------------------------
# (Ancienne GUI Tkinter, laissée mais non utilisée en CLI)
# ---------------------------------------------------------------------

def create_gui():
    # Cette fonction existe toujours mais n'est plus appelée en mode CLI.
    # On la laisse pour que tu puisses éventuellement réactiver une GUI locale.
    import tkinter as tk
    from tkinter import scrolledtext
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("💎 Enhanced Method v13.5 💎")
    root.geometry("1200x900")
    
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(3, weight=1)
    root.grid_rowconfigure(6, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)
    
    tk.Label(root, text="Yuichi v13.5 - Enhanced Method", font=("Arial", 16, "bold")).grid(
        row=0, column=0, columnspan=3, pady=10
    )
    
    log_box = scrolledtext.ScrolledText(root, width=80, height=12, font=("Courier", 9))
    log_box.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="nsew")
    log_box.config(state=tk.DISABLED)
    
    tk.Label(root, text="Cumulative Winnings", font=("Arial", 12, "bold")).grid(
        row=2, column=0, sticky="w", padx=10, pady=(10,0)
    )
    winnings_var = tk.StringVar()
    tk.Entry(root, textvariable=winnings_var, font=("Arial", 14, "bold"), width=15, justify="center").grid(
        row=2, column=1, sticky="w", padx=10, pady=(10,0)
    )
    
    stats_frame = tk.LabelFrame(root, text="Stats", font=("Arial", 10, "bold"))
    stats_frame.grid(row=3, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")
    stats_box = tk.Text(stats_frame, height=15, width=30, font=("Courier", 9))
    stats_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    stats_box.config(state=tk.DISABLED)
    
    psych_frame = tk.LabelFrame(root, text="Psychology", font=("Arial", 10, "bold"))
    psych_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    psych_box = tk.Text(psych_frame, height=15, width=60, font=("Courier", 9))
    psych_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    psych_box.config(state=tk.DISABLED)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    canvas = FigureCanvasTkAgg(fig, master=root)
    tk.Label(root, text="Price Chart", font=("Arial", 12, "bold")).grid(
        row=5, column=0, columnspan=3, pady=(10,0)
    )
    canvas.get_tk_widget().grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
    
    btn_frame = tk.Frame(root)
    btn_frame.grid(row=7, column=0, columnspan=3, pady=10)
    
    def stop():
        config.running = False
        log_result("Bot stopped.", log_box)
    
    tk.Button(btn_frame, text="Stop", command=stop, bg="#c0392b", fg="white",
              font=("Arial", 12, "bold"), width=12).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Quit", command=root.destroy, bg="#7f8c8d", fg="white",
              font=("Arial", 12, "bold"), width=12).pack(side=tk.LEFT, padx=10)
    
    update_winnings_box(winnings_var)
    
    t = threading.Thread(
        target=execute_yuichi_strategy,
        args=(log_box, ax, canvas, winnings_var, stats_box, psych_box),
        daemon=True
    )
    t.start()
    
    root.mainloop()

# ---------------------------------------------------------------------
# ENTRY POINT (CLI)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("💎 Initializing Enhanced Method v13.5 (CLI) 💎")
    logger.info(f"Capital: ${config.capital}")
    logger.info("Mode: CLI – GUI disabled, status JSON enabled.")
    execute_yuichi_strategy()
