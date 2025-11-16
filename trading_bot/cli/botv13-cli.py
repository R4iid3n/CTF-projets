# -*- coding: utf-8 -*-
# Trading Bot v13 - "Special Method v13" - CLI VERSION
# Psychological Warfare with Hedge Bets, Battle-Level Martingale, and Balanced Long/Short
# FIXES: Increased position sizes, optimized hedging, gentler SL shrinkage, better confirmations

import threading
import ccxt
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timezone
from collections import deque
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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
    format="%(asctime)s - [YUICHI] - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("yuichi_bot_v13_cli.log"),
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
    NONE = "none"


@dataclass
class MarketPsychology:
    sentiment: str                    # bull, bear, neutral, manipulation
    fear_greed_index: float           # 0-100
    retail_positioning: float         # proxy RSI
    smart_money_flow: float           # order flow delta
    volatility_regime: str            # low / medium / high / extreme
    manipulation_detected: bool
    trap_type: TrapType
    trap_confidence: float            # 0-100
    game_state: str                   # observing / setup / game_over / uncertain
    trap_probability: float           # 0-1
    confirmations: List[str] = field(default_factory=list)


@dataclass
class TradeState:
    active: bool = False
    type: Optional[str] = None        # "buy" ou "sell"
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None

    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    position_size_main: float = 0.0
    position_size_hedge: float = 0.0

    score: float = 0.0
    confirmations: List[str] = field(default_factory=list)
    battle_pnl: float = 0.0
    psychology_snapshot: Optional[MarketPsychology] = None


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


# ---------------------------------------------------------------------
# CONFIG YUICHI v13
# ---------------------------------------------------------------------


class YuichiConfig:
    bot_name = "yuichi_v13_cli"

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

    trading_fee_rate = 0.001
    slippage_rate = 0.0003
    min_profit_threshold = 0.004

    max_daily_loss_pct = 0.18
    max_trades_per_day = 16

    # Martingale battle-level
    base_position = 40.0
    hedge_ratio = 0.4              # hedge size = 40% du main
    martingale_steps = [1.0, 1.6, 2.4, 3.6, 5.0, 7.5]
    current_step = 0
    max_martingale_steps = 4

    min_trap_confidence = 60.0
    min_setup_score = 6.5

    min_time_between_trades = 8     # minutes
    last_trade_time: Optional[datetime] = None

    min_observation_candles = 50

    trades_executed_today = 0
    daily_loss = 0.0
    cumulative_winnings = 0.0
    running = True

    battle_step = 0                 # numéro du "combat" Yuichi

    status_dir = "status"


config = YuichiConfig()
performance = PerformanceTracker()
trade_state = TradeState()
price_history: List[float] = []

os.makedirs(config.status_dir, exist_ok=True)

# ---------------------------------------------------------------------
# EXCHANGE WRAPPER
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


def calculate_all_indicators(df):
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
# MARKET / PSYCHOLOGY
# ---------------------------------------------------------------------


def detect_yuichi_trap(df_5m, df_15m, df_1h, order_book) -> (TrapType, float):
    if df_5m is None or df_5m.empty or len(df_5m) < 50:
        return TrapType.NONE, 0.0

    price = df_5m["close"].iloc[-1]
    rsi_5m = df_5m["RSI"].iloc[-1]
    volume = df_5m["volume"].iloc[-1]
    avg_volume = df_5m["volume"].iloc[-30:-1].mean()

    recent_high_5m = df_5m["high"].iloc[-40:-1].max()
    recent_low_5m = df_5m["low"].iloc[-40:-1].min()

    confidence = 0.0
    trap = TrapType.NONE

    # BULL TRAP
    if price > recent_high_5m:
        volume_ratio = volume / avg_volume
        if volume_ratio < 0.8:
            confidence += 25
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

        if confidence >= 50:
            trap = TrapType.BULL_TRAP
            logger.warning(f"[TRAP] BULL TRAP detected! Conf={confidence:.0f}%")

    # BEAR TRAP
    elif price < recent_low_5m:
        volume_ratio = volume / avg_volume
        if volume_ratio < 0.8:
            confidence += 25
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

        if confidence >= 50:
            trap = TrapType.BEAR_TRAP
            logger.warning(f"[TRAP] BEAR TRAP detected! Conf={confidence:.0f}%")

    # LIQUIDITY GRAB simple
    if df_5m is not None and len(df_5m) >= 5:
        candle_1 = df_5m.iloc[-5]
        candle_3 = df_5m.iloc[-3]
        candle_now = df_5m.iloc[-1]

        drop = (candle_3["low"] - candle_1["low"]) / candle_1["low"]
        bounce = (candle_now["close"] - candle_3["low"]) / candle_3["low"]

        if drop < -0.01 and bounce > 0.005:
            confidence = max(confidence, 65)
            trap = TrapType.LIQUIDITY_GRAB
            logger.warning(f"[TRAP] LIQUIDITY GRAB detected! Conf={confidence:.0f}%")

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
            trap_confidence=0.0,
            game_state="uncertain",
            trap_probability=0.0,
        )

    rsi = df_5m["RSI"].iloc[-1]
    atr_pct = (df_5m["ATR"].iloc[-1] / df_5m["close"].iloc[-1]) * 100

    if rsi < 20:
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

    if atr_pct > 3.0:
        vol_regime = "extreme"
    elif atr_pct > 1.5:
        vol_regime = "high"
    elif atr_pct > 0.5:
        vol_regime = "medium"
    else:
        vol_regime = "low"

    trap, trap_confidence = detect_yuichi_trap(df_5m, df_15m, df_1h, order_book)
    manipulation_detected = trap != TrapType.NONE

    if manipulation_detected:
        sentiment = "manipulation"

    if trap != TrapType.NONE and trap_confidence >= 75:
        game_state = "game_over"
    elif trap != TrapType.NONE and trap_confidence >= 60:
        game_state = "setup_forming"
    elif trap != TrapType.NONE:
        game_state = "trap_detected"
    elif vol_regime in ["extreme"]:
        game_state = "uncertain"
    else:
        game_state = "observing"

    order_flow = df_5m["order_flow_delta"].iloc[-1] if "order_flow_delta" in df_5m.columns else 0.0

    trap_probability = trap_confidence / 100.0

    confirmations = []
    if trap != TrapType.NONE:
        confirmations.append(f"trap:{trap.value}")
    if order_flow > 0:
        confirmations.append("smart_money_buy")
    elif order_flow < 0:
        confirmations.append("smart_money_sell")

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
        trap_probability=trap_probability,
        confirmations=confirmations,
    )

# ---------------------------------------------------------------------
# ENTRY LOGIC
# ---------------------------------------------------------------------


def yuichi_entry_signal(multi_tf_data, psychology: MarketPsychology):
    if psychology.game_state in ["uncertain", "observing"]:
        return None, 0.0, ["no_setup"]

    df = multi_tf_data.get("tactical")
    if df is None or df.empty or len(df) < config.min_observation_candles:
        return None, 0.0, ["not_enough_data"]

    signal_type = None
    score = 0.0
    confirmations = []

    trap = psychology.trap_type
    trap_conf = psychology.trap_confidence

    if trap == TrapType.BEAR_TRAP:
        signal_type = "buy"
        score = trap_conf / 10.0
        rsi = df["RSI"].iloc[-1]
        if rsi < 30:
            score += 1.0
            confirmations.append("oversold_rsi")
        if psychology.smart_money_flow > 0:
            score += 1.0
            confirmations.append("smart_money_buy")

    elif trap == TrapType.BULL_TRAP:
        signal_type = "sell"
        score = trap_conf / 10.0
        rsi = df["RSI"].iloc[-1]
        if rsi > 70:
            score += 1.0
            confirmations.append("overbought_rsi")
        if psychology.smart_money_flow < 0:
            score += 1.0
            confirmations.append("smart_money_sell")

    elif trap == TrapType.LIQUIDITY_GRAB:
        df_1h = multi_tf_data.get("oversight")
        if df_1h is not None and not df_1h.empty and len(df_1h) > 50:
            sma50 = df_1h["SMA_50"].iloc[-1]
            sma200 = df_1h["SMA_200"].iloc[-1]
            signal_type = "buy" if sma50 > sma200 else "sell"
            score = trap_conf / 10.0 + 0.5
            confirmations.append("liquidity_grab")

    if not signal_type or score < config.min_setup_score:
        return None, score, confirmations

    logger.info(
        f"[SIGNAL] {signal_type.upper()} | Score={score:.1f}/10 | "
        f"Trap={trap.value if trap else 'none'} ({trap_conf:.0f}%) | Conf={confirmations}"
    )
    return signal_type, score, confirmations

# ---------------------------------------------------------------------
# POSITION SIZING & SL/TP
# ---------------------------------------------------------------------


def calculate_yuichi_position_sizes(psychology: MarketPsychology, score: float) -> (float, float):
    base = config.base_position
    mult = config.martingale_steps[config.current_step]

    main_size = base * mult
    hedge_size = main_size * config.hedge_ratio

    if psychology.volatility_regime == "extreme":
        main_size *= 0.5
        hedge_size *= 0.5
    elif psychology.volatility_regime == "low":
        main_size *= 1.2
        hedge_size *= 1.2

    if score >= 9.0:
        main_size *= 1.3
        hedge_size *= 1.3
        logger.info("[SIZE] Perfect setup, boosted +30%")

    return main_size, hedge_size


def calculate_smart_sl_tp(entry, df, signal_type, psychology: MarketPsychology, score: float):
    atr = df["ATR"].iloc[-1]

    recent_lows = df["low"].iloc[-30:].nsmallest(3).mean()
    recent_highs = df["high"].iloc[-30:].nlargest(3).mean()

    if signal_type == "buy":
        structural_sl = recent_lows * 0.997
        atr_sl = entry - (atr * 1.0)
        stop_loss = max(structural_sl, atr_sl)

        if score >= 9.0:
            tp_mult = 3.5
        elif score >= 8.0:
            tp_mult = 3.0
        elif score >= 7.0:
            tp_mult = 2.5
        else:
            tp_mult = 2.0

        take_profit = entry + (atr * tp_mult)

    else:
        structural_sl = recent_highs * 1.003
        atr_sl = entry + (atr * 1.0)
        stop_loss = min(structural_sl, atr_sl)

        if score >= 9.0:
            tp_mult = 3.5
        elif score >= 8.0:
            tp_mult = 3.0
        elif score >= 7.0:
            tp_mult = 2.5
        else:
            tp_mult = 2.0

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
# STATUS JSON POUR DASHBOARD
# ---------------------------------------------------------------------


def write_status(extra: dict | None = None):
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
            "position_size_main": getattr(trade_state, "position_size_main", 0.0),
            "position_size_hedge": getattr(trade_state, "position_size_hedge", 0.0),
            "battle_pnl": getattr(trade_state, "battle_pnl", 0.0),

            "game_state": trade_state.psychology_snapshot.game_state
            if trade_state.psychology_snapshot else None,
            "trap_type": trade_state.psychology_snapshot.trap_type.value
            if trade_state.psychology_snapshot and trade_state.psychology_snapshot.trap_type
            else None,
            "trap_probability": trade_state.psychology_snapshot.trap_probability
            if trade_state.psychology_snapshot else None,

            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            data.update(extra)

        path = os.path.join(config.status_dir, f"{config.bot_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"[STATUS] Write error: {e}")

# ---------------------------------------------------------------------
# LOG HELPERS (CLI)
# ---------------------------------------------------------------------


def log_result(msg):
    try:
        with open("yuichi_v13_trades_cli.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {msg}\n")
        logger.info(msg)
    except Exception as e:
        logger.error(f"[LOG] Error: {e}")

# ---------------------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------------------


def fetch_multi_timeframe_data(symbol):
    data = {}
    for name, tf in config.timeframes.items():
        limit = 200 if name != "macro" else 100
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        if ohlcv:
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            data[name] = df
    return data

# ---------------------------------------------------------------------
# CLOSE TRADE
# ---------------------------------------------------------------------


def close_trade(reason, exit_price):
    if not trade_state.active:
        return

    main_size = trade_state.position_size_main
    hedge_size = trade_state.position_size_hedge
    total_size = main_size + hedge_size

    net_main, fees_main, gross_main = calculate_realistic_pnl(
        trade_state.entry_price,
        exit_price,
        main_size,
        trade_state.type,
    )

    # Hedge est dans le sens inverse
    hedge_type = "sell" if trade_state.type == "buy" else "buy"
    net_hedge, fees_hedge, gross_hedge = calculate_realistic_pnl(
        trade_state.entry_price,
        exit_price,
        hedge_size,
        hedge_type,
    )

    net_pnl = net_main + net_hedge
    total_fees = fees_main + fees_hedge
    gross_pnl = gross_main + gross_hedge

    config.capital += net_pnl
    config.cumulative_winnings += net_pnl
    trade_state.battle_pnl += net_pnl

    if net_pnl > 0:
        config.current_step = 0
    else:
        config.current_step = min(
            config.current_step + 1,
            config.max_martingale_steps,
        )
        config.daily_loss += abs(net_pnl)

    result = "WIN" if net_pnl > 0 else "LOSS"
    roi_pct = (net_pnl / total_size) * 100 if total_size > 0 else 0

    psych = trade_state.psychology_snapshot

    trade_data = {
        "timestamp": datetime.now(),
        "type": trade_state.type,
        "entry": trade_state.entry_price,
        "exit": exit_price,
        "profit_loss": net_pnl,
        "gross_pnl": gross_pnl,
        "fees": total_fees,
        "result": result,
        "score": trade_state.score,
        "game_state": psych.game_state if psych else "unknown",
        "trap_type": psych.trap_type.value if psych and psych.trap_type else "none",
        "trap_probability": psych.trap_probability if psych else None,
    }

    performance.log_trade(trade_data)

    log_msg = (
        f"\n{'=' * 70}\n"
        f"[CLOSE] {reason} [{result}]\n"
        f"{'=' * 70}\n"
        f"Entry: ${trade_state.entry_price:.2f} -> Exit: ${exit_price:.2f}\n"
        f"Main: ${main_size:.2f} | Hedge: ${hedge_size:.2f}\n"
        f"Gross P/L: ${gross_pnl:.2f}\n"
        f"Fees: -${total_fees:.2f}\n"
        f"Net P/L: ${net_pnl:.2f} ({roi_pct:+.2f}%)\n"
        f"Capital: ${config.capital:.2f}\n"
        f"Total Winnings: ${config.cumulative_winnings:.2f}\n"
        f"Battle PnL: ${trade_state.battle_pnl:.2f}\n"
        f"Win Rate: {performance.win_rate:.1f}% | PF: {performance.profit_factor:.2f}\n"
        f"Battle step: {config.battle_step}\n"
        f"{'=' * 70}\n"
    )

    log_result(log_msg)

    trade_state.active = False
    config.trades_executed_today += 1
    config.last_trade_time = datetime.now()

    write_status()

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------


def execute_yuichi_strategy():
    logger.info("======================================================")
    logger.info("[YUICHI] - YUICHI BOT ACTIVATED (v13 FIXED CLI)")
    logger.info("[YUICHI] - Psychological warfare with optimized parameters")
    logger.info("======================================================")

    while config.running:
        try:
            # Daily limits
            if (
                config.trades_executed_today >= config.max_trades_per_day
                or config.daily_loss >= config.capital * config.max_daily_loss_pct
                or config.current_step >= config.max_martingale_steps
            ):
                log_result("[LIMIT] Daily limits reached, bot in cooldown.")
                write_status({"state": "cooldown"})
                time.sleep(60)
                continue

            # Cooldown between trades
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
            trade_state.psychology_snapshot = psychology

            # PAS DE POSITION
            if not trade_state.active:
                signal_type, score, confirmations = yuichi_entry_signal(multi_tf_data, psychology)

                if not signal_type:
                    logger.info(
                        f"[STATE] observing | price={current_price:.2f} | battle={config.battle_step} "
                        f"| cumu=${config.cumulative_winnings:.2f}"
                    )
                    write_status({"state": "observing"})
                    time.sleep(5)
                    continue

                atr = df["ATR"].iloc[-1]
                expected_profit_pct = (atr * 2.5) / current_price
                if expected_profit_pct < config.min_profit_threshold:
                    logger.info(f"[SKIP] Expected profit {expected_profit_pct:.2%} < threshold")
                    write_status({"state": "low_edge"})
                    time.sleep(5)
                    continue

                size_main, size_hedge = calculate_yuichi_position_sizes(psychology, score)
                if size_main <= 0:
                    write_status({"state": "zero_size"})
                    time.sleep(5)
                    continue

                trade_state.active = True
                trade_state.type = signal_type
                trade_state.entry_price = current_price
                trade_state.entry_time = datetime.now()
                trade_state.position_size_main = size_main
                trade_state.position_size_hedge = size_hedge
                trade_state.score = score
                trade_state.confirmations = confirmations
                trade_state.battle_pnl = 0.0

                sl, tp = calculate_smart_sl_tp(current_price, df, signal_type, psychology, score)
                trade_state.stop_loss = sl
                trade_state.take_profit = tp

                config.battle_step += 1

                log_msg = (
                    f"\n{'=' * 70}\n"
                    f"[ENTER] YUICHI v13 ENTERS: {signal_type.upper()}\n"
                    f"{'=' * 70}\n"
                    f"Game State: {psychology.game_state.upper()}\n"
                    f"Trap: {psychology.trap_type.value if psychology.trap_type else 'none'}\n"
                    f"Confidence: {psychology.trap_confidence:.0f}% "
                    f"(prob={psychology.trap_probability:.2f})\n"
                    f"Score: {score:.1f}/10\n"
                    f"Confirmations: {confirmations}\n"
                    f"Entry: ${current_price:.2f}\n"
                    f"Main: ${size_main:.2f} | Hedge: ${size_hedge:.2f}\n"
                    f"SL: ${sl:.2f} | TP: ${tp:.2f}\n"
                    f"Battle step: {config.battle_step}\n"
                    f"{'=' * 70}"
                )
                log_result(log_msg)
                write_status({"state": "in_position"})

            # POSITION ACTIVE
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
                    # Heartbeat pendant le trade
                    logger.info(
                        f"[POSITION] {trade_state.type.upper()} "
                        f"entry={trade_state.entry_price:.2f} "
                        f"SL={trade_state.stop_loss:.2f} TP={trade_state.take_profit:.2f} "
                        f"price={current_price:.2f} "
                        f"battle={config.battle_step} "
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
    logger.info("======================================================")
    logger.info("[YUICHI] - Initializing Trading System v13 FIXED (CLI MODE)...")
    logger.info(f"[YUICHI] - Capital: ${config.capital:.2f}")
    logger.info(
        "[YUICHI] - Mode: Psychological warfare with OPTIMIZED parameters "
        "(balanced long/short, hedge, battle-level martingale)"
    )
    logger.info(
        f"[YUICHI] - Status JSON: {os.path.join(config.status_dir, config.bot_name + '.json')}"
    )
    logger.info("======================================================")
    execute_yuichi_strategy()
