# Trading Bot v13 - "Special Method v13" - FIXED VERSION
# Psychological Warfare with Hedge Bets, Battle-Level Martingale, and Balanced Long/Short
# FIXES: Increased position sizes, optimized hedging, gentler SL shrinkage, better confirmations

# import tkinter as tk
# from tkinter import scrolledtext
import threading
import ccxt
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import time
from datetime import datetime
from collections import deque
import sys
import os
import json
from dataclasses import dataclass
from typing import Dict
from enum import Enum

# Forcer l'output console en UTF-8 (sécurité)
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
        logging.FileHandler("yuichi_bot_v13_fixed.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------

class TrapType(Enum):
    BULL_TRAP = "bull_trap"
    BEAR_TRAP = "bear_trap"
    LIQUIDITY_GRAB = "liquidity_grab"
    FAKE_BREAKOUT = "fake_breakout"


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
    trap_probability: float


# ---------------------------------------------------------------------
# CONFIG - OPTIMIZED FOR BETTER PERFORMANCE
# ---------------------------------------------------------------------

class YuichiConfig:
    symbols = ["BTC/USDT"]
    timeframe = "1m"
    capital = 1000.0

    bot_name = "yuichi_v13_cli"
    status_dir = "status"

    timeframes = {
        "micro": "1m",
        "tactical": "5m",
        "strategic": "15m",
        "oversight": "1h",
    }

    max_daily_loss_pct = 0.20
    max_trades_per_day = 25

    # FIXED: Increased base sizes for meaningful trades (6x increase on average)
    martingale_steps = [50, 75, 125, 200, 350, 600, 1000, 1600]
    max_martingale_steps = 6
    current_step = 0  # index in martingale_steps

    # FIXED: More aggressive battle-level multipliers for faster recovery
    battle_steps = [1.0, 1.5, 2.25, 3.5]
    battle_step = 0  # index in battle_steps

    rsi_period = 14
    rsi_extreme_oversold = 30
    rsi_extreme_overbought = 80

    fake_signal_filter = True
    whale_detection = True
    retail_sentiment = True

    min_confirmations = 2
    base_atr_multiplier_sl = 1.2
    base_atr_multiplier_tp = 2.8
    max_sl_shrink_factor = 0.4
    trailing_stop_factor = 0.6

    # FIXED: Hedge activation at 1% instead of 0.3%
    hedge_activation_threshold = 0.01
    # FIXED: Hedge size reduced from 0.5 (50%) to 0.3 (30%)
    hedge_size_ratio = 0.3

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
                wait_time = (2**attempt) * self.rate_limit_buffer
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error during {func.__name__}: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2**attempt)
        return None

    def fetch_ohlcv(self, symbol, timeframe, limit=200):
        return self.fetch_with_retry(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)

    def fetch_order_book(self, symbol, limit=50):
        return self.fetch_with_retry(self.exchange.fetch_order_book, symbol, limit=limit)


exchange = EliteExchange()

# ---------------------------------------------------------------------
# PERFORMANCE
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
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.current_streak = 0
        self.win_rate = 0.0

        self.psychological_edges = 0
        self.trap_avoidances = 0
        self.manipulation_counters = 0
        self.battles = 0

        self.confirmation_win_rates = {}

    def log_trade(self, trade_data):
        self.trades.append(trade_data)

        # Track confirmation effectiveness
        for conf in trade_data.get("confirmations", []):
            if conf not in self.confirmation_win_rates:
                self.confirmation_win_rates[conf] = {"wins": 0, "total": 0}
            self.confirmation_win_rates[conf]["total"] += 1
            if trade_data["profit_loss"] > 0:
                self.confirmation_win_rates[conf]["wins"] += 1

        if trade_data["profit_loss"] > 0:
            self.wins += 1
            self.total_profit += trade_data["profit_loss"]
            self.largest_win = max(self.largest_win, trade_data["profit_loss"])
            self.current_streak = max(1, self.current_streak + 1)
            self.consecutive_wins = max(self.consecutive_wins, self.current_streak)
        else:
            self.losses += 1
            self.total_loss += abs(trade_data["profit_loss"])
            self.largest_loss = max(self.largest_loss, abs(trade_data["profit_loss"]))
            self.current_streak = min(-1, self.current_streak - 1)
            self.consecutive_losses = min(self.consecutive_losses, self.current_streak)

        total = self.wins + self.losses
        if total > 0:
            self.win_rate = (self.wins / total) * 100.0

    def register_psych_edge(self):
        self.psychological_edges += 1

    def register_trap_avoidance(self):
        self.trap_avoidances += 1

    def register_manipulation_counter(self):
        self.manipulation_counters += 1

    def start_battle(self):
        self.battles += 1


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
        logger.error(f"Indicator calculation error: {e}")
        return df

# ---------------------------------------------------------------------
# PSYCHOLOGY & TRAPS
# ---------------------------------------------------------------------

def detect_trap(df_5m, df_15m, df_1h, order_book) -> float:
    if df_5m is None or df_5m.empty or len(df_5m) < 50:
        return 0.0

    price = df_5m["close"].iloc[-1]
    rsi_5m = df_5m["RSI"].iloc[-1]
    volume = df_5m["volume"].iloc[-1]
    avg_volume = df_5m["volume"].iloc[-20:-1].mean()

    recent_high_5m = df_5m["high"].iloc[-30:-1].max()
    recent_low_5m = df_5m["low"].iloc[-30:-1].min()

    trap_probability = 0.0

    if price > recent_high_5m:
        trap_probability += 25
        if volume < avg_volume * 0.8:
            trap_probability += 15
        if rsi_5m > config.rsi_extreme_overbought:
            trap_probability += 15
        if df_15m is not None and len(df_15m) > 5:
            rsi_15m_current = df_15m["RSI"].iloc[-1]
            rsi_15m_prev = df_15m["RSI"].iloc[-3]
            if rsi_15m_current < rsi_15m_prev:
                trap_probability += 15
        if order_book:
            best_bids = sum(b[1] for b in order_book.get("bids", [])[:5])
            best_asks = sum(a[1] for a in order_book.get("asks", [])[:5])
            if best_asks > best_bids * 2:
                trap_probability += 15

    if price < recent_low_5m:
        trap_probability += 25
        if volume < avg_volume * 0.8:
            trap_probability += 15
        if rsi_5m < config.rsi_extreme_oversold:
            trap_probability += 15
        if df_15m is not None and len(df_15m) > 5:
            rsi_15m_current = df_15m["RSI"].iloc[-1]
            rsi_15m_prev = df_15m["RSI"].iloc[-3]
            if rsi_15m_current > rsi_15m_prev:
                trap_probability += 15
        if order_book:
            best_bids = sum(b[1] for b in order_book.get("bids", [])[:5])
            best_asks = sum(a[1] for a in order_book.get("asks", [])[:5])
            if best_bids > best_asks * 2:
                trap_probability += 15

    if df_5m is not None and len(df_5m) >= 5:
        c1 = df_5m.iloc[-5]
        c3 = df_5m.iloc[-3]
        c_now = df_5m.iloc[-1]

        drop = (c3["low"] - c1["low"]) / c1["low"]
        bounce = (c_now["close"] - c3["low"]) / c3["low"]

        if drop < -0.008 and bounce > 0.004:
            trap_probability += 20

    trap_probability = max(0.0, min(100.0, trap_probability))
    return trap_probability


def calculate_psychology(multi_tf_data, order_book) -> MarketPsychology:
    df_5m = multi_tf_data.get("tactical")
    df_1h = multi_tf_data.get("oversight")

    if df_5m is None or df_5m.empty:
        return MarketPsychology(
            sentiment=MarketSentiment.NEUTRAL,
            fear_greed_index=50.0,
            retail_positioning=50.0,
            smart_money_flow=0.0,
            volatility_regime="medium",
            manipulation_detected=False,
            trap_probability=0.0,
        )

    rsi = df_5m["RSI"].iloc[-1]
    atr_pct = (df_5m["ATR"].iloc[-1] / df_5m["close"].iloc[-1]) * 100

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

    if atr_pct > 3.5:
        vol_regime = "extreme"
    elif atr_pct > 1.8:
        vol_regime = "high"
    elif atr_pct > 0.6:
        vol_regime = "medium"
    else:
        vol_regime = "low"

    trap_probability = detect_trap(
        multi_tf_data.get("tactical"),
        multi_tf_data.get("strategic"),
        df_1h,
        order_book,
    )

    manipulation_detected = trap_probability >= 60

    order_flow = (
        df_5m["order_flow_delta"].iloc[-1] if "order_flow_delta" in df_5m.columns else 0.0
    )

    return MarketPsychology(
        sentiment=sentiment,
        fear_greed_index=fear_greed,
        retail_positioning=rsi,
        smart_money_flow=order_flow,
        volatility_regime=vol_regime,
        manipulation_detected=manipulation_detected,
        trap_probability=trap_probability,
    )

# ---------------------------------------------------------------------
# ENTRY SIGNAL (Yuichi style)
# ---------------------------------------------------------------------

def yuichi_entry_signal(multi_tf_data, psychology):
    df = multi_tf_data.get("tactical")
    if df is None or df.empty or len(df) < 80:
        return None, [], "Not enough data"

    confirmations = []
    price = df["close"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    ema9 = df["EMA_9"].iloc[-1]
    ema21 = df["EMA_21"].iloc[-1]
    vwap = df["VWAP"].iloc[-1]

    signal_type = None

    if rsi < config.rsi_extreme_oversold and price < vwap:
        signal_type = "buy"
        confirmations.append("oversold_rsi")
    elif rsi > config.rsi_extreme_overbought and price > vwap:
        signal_type = "sell"
        confirmations.append("overbought_rsi")

        # Smart money confirmation
    if psychology.smart_money_flow is not None:
        if psychology.smart_money_flow > 0 and signal_type == "buy":
            confirmations.append("smart_money_buy_support")
        elif psychology.smart_money_flow < 0 and signal_type == "sell":
            confirmations.append("smart_money_sell_support")

    if (price > ema9 > ema21 and signal_type == "buy") or (
        price < ema9 < ema21 and signal_type == "sell"
    ):
        confirmations.append("trend_alignment")

    if psychology.trap_probability >= 60:
        confirmations.append("trap_probability_high")

    if psychology.volatility_regime in ["high", "extreme"]:
        confirmations.append("favorable_volatility")

    if psychology.manipulation_detected:
        confirmations.append("manipulation_detected")

    if len(confirmations) < config.min_confirmations:
        return None, confirmations, "Not enough confirmations"

    logger.info(
        f"Signal: {signal_type.upper()} | Confirmations: {', '.join(confirmations)} | "
        f"TrapProb: {psychology.trap_probability:.1f}% | Vol: {psychology.volatility_regime}"
    )

    return signal_type, confirmations, "Valid signal"

# ---------------------------------------------------------------------
# POSITION / RISK
# ---------------------------------------------------------------------

def calculate_position_size(psychology, confirmations):
    base = config.martingale_steps[config.current_step]

    battle_mult = config.battle_steps[config.battle_step]
    size = base * battle_mult

    if psychology.volatility_regime == "extreme":
        size *= 0.7
    elif psychology.volatility_regime == "low":
        size *= 1.1

    if "trap_probability_high" in confirmations and psychology.trap_probability > 80:
        size *= 1.2

    size = max(10.0, size)
    return size


def calculate_sl_tp(entry_price, df, signal_type, psychology, confirmations):
    atr = df["ATR"].iloc[-1]

    rsi = df["RSI"].iloc[-1]
    shrink_factor = 0.0
    if rsi < 25 or rsi > 75:
        shrink_factor += 0.1
    if psychology.volatility_regime == "extreme":
        shrink_factor += 0.15
    if "trap_probability_high" in confirmations:
        shrink_factor += 0.15

    shrink_factor = min(config.max_sl_shrink_factor, shrink_factor)
    sl_mult = config.base_atr_multiplier_sl * (1 - shrink_factor)
    tp_mult = config.base_atr_multiplier_tp * (1 + shrink_factor * 0.3)

    if signal_type == "buy":
        stop_loss = entry_price - atr * sl_mult
        take_profit = entry_price + atr * tp_mult
    else:
        stop_loss = entry_price + atr * sl_mult
        take_profit = entry_price - atr * tp_mult

    return stop_loss, take_profit


def calculate_realistic_pnl(entry, exit, size, trade_type):
    fee_rate = 0.001
    slippage_rate = 0.0003

    if trade_type == "buy":
        actual_entry = entry * (1 + slippage_rate)
        actual_exit = exit * (1 - slippage_rate)
        qty = size / actual_entry
        gross_pnl = (actual_exit - actual_entry) * qty
    else:
        actual_entry = entry * (1 - slippage_rate)
        actual_exit = exit * (1 + slippage_rate)
        qty = size / actual_entry
        gross_pnl = (actual_entry - actual_exit) * qty

    total_fees = (size * fee_rate) * 2
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
        self.trailing_stop = None
        self.highest_price = None
        self.lowest_price = None
        self.confirmations = []
        self.psychology_snapshot = None

        self.original_stop_loss = None
        self.original_take_profit = None

        self.hedge_active = False
        self.hedge_type = None
        self.hedge_entry_price = None
        self.hedge_size = 0.0  # USDT

        # PnL cumulé sur la "bataille" (main + hedge) en USDT
        self.battle_pnl = 0.0
        self.main_size = 0.0


trade_state = TradeState()
price_history = []

# ---------------------------------------------------------------------
# STATUS EXPORT (for external monitoring / Windows app)
# ---------------------------------------------------------------------

def write_status(extra: dict | None = None):
    """
    Write current bot state to a JSON file so an external app (e.g. on Windows)
    can monitor status in real time.
    """
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
            "position_size_main": getattr(trade_state, "main_size", 0.0),
            "position_size_hedge": getattr(trade_state, "hedge_size", 0.0),
            "battle_pnl": getattr(trade_state, "battle_pnl", 0.0),

            "game_state": trade_state.psychology_snapshot.game_state.value if getattr(trade_state, "psychology_snapshot", None) else None,
            "trap_probability": trade_state.psychology_snapshot.trap_probability if getattr(trade_state, "psychology_snapshot", None) else None,

            "last_update": datetime.utcnow().isoformat() + "Z",
        }
        if extra:
            data.update(extra)

        path = os.path.join(config.status_dir, f"{config.bot_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"[STATUS] Write error: {e}")

# ---------------------------------------------------------------------
# GUI HELPERS (CLI: only file/console logging used)
# ---------------------------------------------------------------------

def log_result(msg, log_box):
    try:
        with open("yuichi_trades_v13_fixed.txt", "a") as f:
            f.write(f"[{datetime.now()}] {msg}\n")
        if log_box:
            log_box.config(state=tk.NORMAL)
            log_box.insert(tk.END, f"{msg}\n")
            log_box.see(tk.END)
            log_box.config(state=tk.DISABLED)
    except Exception as e:
        logger.error(f"Logging error: {e}")


def update_winnings_box(entry_var):
    try:
        entry_var.set(f"${config.cumulative_winnings:.2f}")
    except Exception:
        pass


def update_step_panel(box):
    try:
        if box:
            box.config(state=tk.NORMAL)
            box.delete(1.0, tk.END)
            box.insert(tk.END, "Martingale Base Steps\n" + "=" * 30 + "\n\n")
            for i, step in enumerate(config.martingale_steps):
                marker = ">" if i == config.current_step else " "
                box.insert(tk.END, f"{marker} Step {i + 1}: ${step:.0f}\n")
            box.insert(tk.END, "\nBattle Multipliers\n" + "=" * 30 + "\n\n")
            for i, mult in enumerate(config.battle_steps):
                marker = ">" if i == config.battle_step else " "
                box.insert(tk.END, f"{marker} Battle {i + 1}: x{mult:.2f}\n")
            box.config(state=tk.DISABLED)
    except Exception:
        pass


def update_stats_panel(box):
    try:
        if box:
            box.config(state=tk.NORMAL)
            box.delete(1.0, tk.END)
            box.insert(tk.END, "Performance\n" + "=" * 30 + "\n\n")
            box.insert(tk.END, f"Wins: {performance.wins}\n")
            box.insert(tk.END, f"Losses: {performance.losses}\n")
            box.insert(tk.END, f"Win Rate: {performance.win_rate:.1f}%\n\n")
            box.insert(tk.END, f"Total Profit: ${performance.total_profit:.2f}\n")
            box.insert(tk.END, f"Total Loss: ${performance.total_loss:.2f}\n")
            box.insert(tk.END, f"Largest Win: ${performance.largest_win:.2f}\n")
            box.insert(tk.END, f"Largest Loss: ${performance.largest_loss:.2f}\n\n")
            box.insert(tk.END, f"Psych Edges: {performance.psychological_edges}\n")
            box.insert(tk.END, f"Traps Avoided: {performance.trap_avoidances}\n")
            box.insert(tk.END, f"Whale Counters: {performance.manipulation_counters}\n\n")
            box.insert(tk.END, f"Best Streak: {performance.consecutive_wins}\n")
            box.insert(tk.END, f"Worst Streak: {performance.consecutive_losses}\n")
            box.insert(tk.END, f"Battles: {performance.battles}\n")
            box.config(state=tk.DISABLED)
    except Exception:
        pass


def update_psychology_panel(box, psychology):
    try:
        if box and psychology:
            box.config(state=tk.NORMAL)
            box.delete(1.0, tk.END)
            box.insert(tk.END, "Market Psychology\n" + "=" * 30 + "\n\n")
            box.insert(tk.END, f"Sentiment: {psychology.sentiment.value.upper()}\n")
            box.insert(tk.END, f"F&G Index: {psychology.fear_greed_index:.0f}/100\n")
            box.insert(tk.END, f"Retail Positioning (RSI): {psychology.retail_positioning:.1f}\n")
            box.insert(tk.END, f"Smart Money Flow: {psychology.smart_money_flow:.2f}\n")
            box.insert(tk.END, f"Volatility Regime: {psychology.volatility_regime}\n")
            box.insert(tk.END, f"Manipulation Detected: {psychology.manipulation_detected}\n")
            box.insert(tk.END, f"Trap Probability: {psychology.trap_probability:.1f}%\n")
            box.config(state=tk.DISABLED)
    except Exception:
        pass


def update_chart(ax, canvas):
    """CLI mode: chart disabled. Kept for compatibility."""
    return

# ---------------------------------------------------------------------
# BATTLE MANAGEMENT
# ---------------------------------------------------------------------

def reset_battle():
    trade_state.battle_pnl = 0.0
    config.battle_step = 0
    trade_state.hedge_active = False
    trade_state.hedge_type = None
    trade_state.hedge_entry_price = None
    trade_state.hedge_size = 0.0

    trade_state.battle_pnl = 0.0
    trade_state.main_size = 0.0


def on_battle_end(net_pnl):
    if net_pnl < 0:
        config.current_step = min(
            config.current_step + 1, config.max_martingale_steps - 1
        )
        if config.battle_step < len(config.battle_steps) - 1:
            config.battle_step += 1
    else:
        config.current_step = 0
        config.battle_step = 0

    reset_battle()

# ---------------------------------------------------------------------
# CLOSE TRADE
# ---------------------------------------------------------------------

def close_trade(reason, battle_pnl, exit_price, log_box, stats_box, step_box, winnings_var):
    config.capital += battle_pnl
    config.cumulative_winnings += battle_pnl

    on_battle_end(battle_pnl)

    result_label = "WIN" if battle_pnl > 0 else "LOSS"

    trade_data = {
        "timestamp": datetime.now(),
        "type": trade_state.type,
        "entry": trade_state.entry_price,
        "exit": exit_price,
        "profit_loss": battle_pnl,
        "result": result_label,
        "confirmations": trade_state.confirmations,
    }
    performance.log_trade(trade_data)

    log_msg = (
        f"\n{'=' * 70}\n"
        f"{reason.upper()} [{result_label}] - Battle PnL: ${battle_pnl:.2f}\n"
        f"{'-' * 70}\n"
        f"Entry: ${trade_state.entry_price:.2f} | Exit: ${exit_price:.2f}\n"
        f"Main Size: ${trade_state.main_size:.2f} | Hedge Size: ${trade_state.hedge_size:.2f}\n"
        f"Capital: ${config.capital:.2f} | Cumulative: ${config.cumulative_winnings:.2f}\n"
        f"Martingale Step: {config.current_step + 1}/{len(config.martingale_steps)}\n"
        f"Battle Step: {config.battle_step + 1}/{len(config.battle_steps)}\n"
        f"{'=' * 70}"
    )
    log_result(log_msg, log_box)

    trade_state.active = False
    trade_state.type = None
    trade_state.entry_price = None
    trade_state.stop_loss = None
    trade_state.take_profit = None
    trade_state.trailing_stop = None
    trade_state.highest_price = None
    trade_state.lowest_price = None
    trade_state.confirmations = []
    trade_state.psychology_snapshot = None
    trade_state.hedge_active = False
    trade_state.hedge_type = None
    trade_state.hedge_entry_price = None
    trade_state.hedge_size = 0.0

    config.trades_executed_today += 1

    update_stats_panel(stats_box)
    update_step_panel(step_box)
    update_winnings_box(winnings_var)
    write_status({"last_trade_reason": reason, "last_trade_pnl": battle_pnl, "last_trade_result": result_label})

# ---------------------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------------------

def fetch_multi_timeframe_data(symbol):
    data = {}
    for name, tf in config.timeframes.items():
        limit = 200
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        if ohlcv:
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            data[name] = df
    return data

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------

def execute_yuichi_strategy(
    log_box, ax, canvas, winnings_var, step_box, stats_box, psych_box
):
    logger.info("=" * 70)
    logger.info("YUICHI BOT ACTIVATED (v13 FIXED)")
    logger.info("Psychological warfare with optimized parameters")
    logger.info("FIXES: 6x position sizes, smarter hedging, gentler SL shrinkage")
    logger.info("=" * 70)

    while config.running:
        try:
            if (
                config.trades_executed_today >= config.max_trades_per_day
                or config.daily_loss >= (config.capital * config.max_daily_loss_pct)
                or config.current_step >= config.max_martingale_steps
            ):
                log_result("Daily limits reached. Resetting later.", log_box)
                time.sleep(60)
                continue

            multi_tf_data = fetch_multi_timeframe_data(config.symbols[0])
            df = multi_tf_data.get("tactical")
            if df is None or df.empty or len(df) < 80:
                time.sleep(5)
                continue

            for name in list(multi_tf_data.keys()):
                multi_tf_data[name] = calculate_all_indicators(multi_tf_data[name])

            df = multi_tf_data["tactical"]
            current_price = df["close"].iloc[-1]
            atr = df["ATR"].iloc[-1]

            price_history.append(current_price)
            if len(price_history) > 2500:
                price_history.pop(0)

            order_book = exchange.fetch_order_book(config.symbols[0], limit=50)
            psychology = calculate_psychology(multi_tf_data, order_book)
            update_psychology_panel(psych_box, psychology)

            # ------------------ NO ACTIVE POSITION ------------------
            if not trade_state.active:
                signal_type, confirmations, _ = yuichi_entry_signal(
                    multi_tf_data, psychology
                )

                if not signal_type:
                    time.sleep(5)
                    continue

                position_size = calculate_position_size(psychology, confirmations)
                if position_size <= 0:
                    time.sleep(5)
                    continue

                trade_state.active = True
                trade_state.type = signal_type
                trade_state.entry_price = current_price
                trade_state.entry_time = datetime.now()
                trade_state.confirmations = confirmations.copy()
                trade_state.psychology_snapshot = psychology
                trade_state.main_size = position_size

                performance.start_battle()

                sl, tp = calculate_sl_tp(
                    current_price, df, signal_type, psychology, confirmations
                )
                trade_state.stop_loss = sl
                trade_state.take_profit = tp
                trade_state.original_stop_loss = sl
                trade_state.original_take_profit = tp
                trade_state.highest_price = current_price
                trade_state.lowest_price = current_price

                log_msg = (
                    f"\n{'=' * 70}\n"
                    f"NEW BATTLE STARTED - {signal_type.upper()} | Size: ${position_size:.2f}\n"
                    f"{'-' * 70}\n"
                    f"Entry: ${current_price:.2f}\n"
                    f"SL: ${sl:.2f} | TP: ${tp:.2f}\n"
                    f"Martingale Step: {config.current_step + 1}/{len(config.martingale_steps)}\n"
                    f"Battle Step: {config.battle_step + 1}/{len(config.battle_steps)}\n"
                    f"Confirmations: {', '.join(confirmations)}\n"
                    f"TrapProb: {psychology.trap_probability:.1f}% | Vol: {psychology.volatility_regime}\n"
                    f"{'=' * 70}"
                )
                log_result(log_msg, log_box)

            # ------------------ ACTIVE POSITION ------------------
            else:
                new_atr = df["ATR"].iloc[-1]
                if trade_state.type == "buy":
                    if current_price > trade_state.highest_price:
                        trade_state.highest_price = current_price
                        distance = trade_state.highest_price - trade_state.entry_price
                        if distance > new_atr * config.base_atr_multiplier_sl:
                            new_trailing = trade_state.highest_price - new_atr * config.trailing_stop_factor
                            trade_state.trailing_stop = max(
                                trade_state.trailing_stop or trade_state.stop_loss,
                                new_trailing,
                            )
                    elif current_price < trade_state.lowest_price:
                        trade_state.lowest_price = current_price

                else:  # SELL
                    if current_price < trade_state.lowest_price:
                        trade_state.lowest_price = current_price
                        distance = trade_state.entry_price - trade_state.lowest_price
                        if distance > new_atr * config.base_atr_multiplier_sl:
                            new_trailing = trade_state.lowest_price + new_atr * config.trailing_stop_factor
                            trade_state.trailing_stop = min(
                                trade_state.trailing_stop or trade_state.stop_loss,
                                new_trailing,
                            )
                    elif current_price > trade_state.highest_price:
                        trade_state.highest_price = current_price

                # Hedge logic
                if not trade_state.hedge_active and psychology.manipulation_detected:
                    if trade_state.type == "buy":
                        adverse_move_pct = (trade_state.entry_price - current_price) / trade_state.entry_price
                    else:
                        adverse_move_pct = (current_price - trade_state.entry_price) / trade_state.entry_price

                    if (
                        adverse_move_pct > config.hedge_activation_threshold
                        and psychology.manipulation_detected
                    ):
                        trade_state.hedge_active = True
                        trade_state.hedge_type = "sell" if trade_state.type == "buy" else "buy"
                        trade_state.hedge_entry_price = current_price
                        hedge_size = trade_state.main_size * config.hedge_size_ratio
                        trade_state.hedge_size = hedge_size

                        log_result(
                            f"HEDGE BET ACTIVATED: {trade_state.hedge_type.upper()} at ${current_price:.2f} "
                            f"against main {trade_state.type.upper()} | Hedge Size: ${hedge_size:.2f}",
                            log_box,
                        )

                # Evaluate PnL
                main_pnl, _, _ = calculate_realistic_pnl(
                    trade_state.entry_price, current_price, trade_state.main_size, trade_state.type
                )
                hedge_pnl = 0.0
                if trade_state.hedge_active:
                    hedge_pnl, _, _ = calculate_realistic_pnl(
                        trade_state.hedge_entry_price,
                        current_price,
                        trade_state.hedge_size,
                        trade_state.hedge_type,
                    )

                trade_state.battle_pnl = main_pnl + hedge_pnl

                # Exit rules
                max_trade_duration = 60 * 60
                elapsed = (datetime.now() - trade_state.entry_time).total_seconds()
                if elapsed > max_trade_duration:
                    exit_price = current_price
                    if trade_state.main_size > 0:
                        main_qty = trade_state.main_size / trade_state.entry_price
                    else:
                        main_qty = 0.0

                    if trade_state.type == "buy":
                        price_diff = exit_price - trade_state.entry_price
                    else:
                        price_diff = trade_state.entry_price - exit_price

                    main_pnl_raw = price_diff * main_qty
                    battle_pnl = trade_state.battle_pnl + main_pnl_raw

                    close_trade(
                        "Time limit",
                        battle_pnl,
                        exit_price,
                        log_box,
                        stats_box,
                        step_box,
                        winnings_var,
                    )
                    update_chart(ax, canvas)
                    continue

                # TP / SL / trailing
                if trade_state.type == "buy":
                    if current_price >= trade_state.take_profit:
                        exit_price = trade_state.take_profit
                        if trade_state.main_size > 0:
                            main_qty = trade_state.main_size / trade_state.entry_price
                        else:
                            main_qty = 0.0
                        price_diff = exit_price - trade_state.entry_price
                        main_pnl_raw = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl_raw
                        close_trade(
                            "TAKE PROFIT",
                            battle_pnl,
                            exit_price,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif trade_state.trailing_stop and current_price <= trade_state.trailing_stop:
                        exit_price = trade_state.trailing_stop
                        if trade_state.main_size > 0:
                            main_qty = trade_state.main_size / trade_state.entry_price
                        else:
                            main_qty = 0.0
                        price_diff = exit_price - trade_state.entry_price
                        main_pnl_raw = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl_raw
                        close_trade(
                            "Trailing Stop",
                            battle_pnl,
                            exit_price,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif current_price <= trade_state.stop_loss:
                        exit_price = trade_state.stop_loss
                        if trade_state.main_size > 0:
                            main_qty = trade_state.main_size / trade_state.entry_price
                        else:
                            main_qty = 0.0
                        price_diff = exit_price - trade_state.entry_price
                        main_pnl_raw = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl_raw
                        close_trade(
                            "Stop Loss",
                            battle_pnl,
                            exit_price,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )

                else:  # SELL
                    if current_price <= trade_state.take_profit:
                        exit_price = trade_state.take_profit
                        if trade_state.main_size > 0:
                            main_qty = trade_state.main_size / trade_state.entry_price
                        else:
                            main_qty = 0.0
                        price_diff = trade_state.entry_price - exit_price
                        main_pnl_raw = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl_raw
                        close_trade(
                            "TAKE PROFIT",
                            battle_pnl,
                            exit_price,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif trade_state.trailing_stop and current_price >= trade_state.trailing_stop:
                        exit_price = trade_state.trailing_stop
                        if trade_state.main_size > 0:
                            main_qty = trade_state.main_size / trade_state.entry_price
                        else:
                            main_qty = 0.0
                        price_diff = trade_state.entry_price - exit_price
                        main_pnl_raw = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl_raw
                        close_trade(
                            "Trailing Stop",
                            battle_pnl,
                            exit_price,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif current_price >= trade_state.stop_loss:
                        exit_price = trade_state.stop_loss
                        if trade_state.main_size > 0:
                            main_qty = trade_state.main_size / trade_state.entry_price
                        else:
                            main_qty = 0.0
                        price_diff = trade_state.entry_price - exit_price
                        main_pnl_raw = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl_raw
                        close_trade(
                            "Stop Loss",
                            battle_pnl,
                            exit_price,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )

            update_stats_panel(stats_box)
            logger.info(f"[STATE] Step={config.current_step} BattleStep={config.battle_step} Cumulative=${config.cumulative_winnings:.2f} BattlePnL=${trade_state.battle_pnl:.2f}")
            write_status()
            update_step_panel(step_box)
            update_winnings_box(winnings_var)
            update_chart(ax, canvas)

            time.sleep(3)

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(5)

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Initializing Trading System v13 FIXED (CLI MODE)...")
    logger.info(f"Capital: ${config.capital}")
    logger.info("Mode: Psychological warfare with OPTIMIZED parameters (CLI only)")
    logger.info("Status JSON: {}/{}.json".format(config.status_dir, config.bot_name))
    # Run strategy in CLI mode (no GUI)
    execute_yuichi_strategy(
        log_box=None,
        ax=None,
        canvas=None,
        winnings_var=None,
        step_box=None,
        stats_box=None,
        psych_box=None,
    )
