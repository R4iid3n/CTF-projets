# Trading Bot v11 - "YUICHI KATAGIRI EDITION"
# Psychological Warfare: Outsmart the Elite Market Makers
#
# Core Philosophy: "The market thinks it is smarter than you. Let it."
# Strategy: Anticipate, Deceive, Exploit, Dominate

import tkinter as tk
from tkinter import scrolledtext
import threading
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import time
from datetime import datetime
from collections import deque
import sys

from dataclasses import dataclass
from typing import Dict
from enum import Enum

# Make sure console can at least use utf-8 (not strictly needed now that we are ASCII only)
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
        logging.FileHandler("yuichi_bot.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# MARKET PSYCHOLOGY
# ---------------------------------------------------------------------

class MarketSentiment(Enum):
    FEAR = "fear"
    GREED = "greed"
    UNCERTAINTY = "uncertainty"
    MANIPULATION = "manipulation"


class TrapType(Enum):
    BULL_TRAP = "bull_trap"
    BEAR_TRAP = "bear_trap"
    LIQUIDITY_GRAB = "liquidity_grab"
    WHALE_ACCUMULATION = "whale_accumulation"


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
# CONFIG
# ---------------------------------------------------------------------

class YuichiConfig:
    symbols = ["BTC/USDT"]
    timeframe = "1m"
    capital = 1000.0

    timeframes = {
        "micro": "1m",
        "tactical": "5m",
        "strategic": "15m",
        "oversight": "1h",
    }

    max_daily_loss_pct = 0.20
    max_trades_per_day = 25

    martingale_steps = [8, 12, 25, 50, 100, 200, 400, 800]
    max_martingale_steps = 6

    rsi_period = 14
    rsi_extreme_oversold = 25
    rsi_extreme_overbought = 75

    fake_signal_filter = True
    whale_detection = True
    retail_sentiment = True

    min_confirmation_layers = 4
    max_hold_time_minutes = 30

    market_regime_adaptation = True
    psychological_edge_mode = True
    counter_manipulation = True

    trades_executed_today = 0
    daily_loss = 0.0
    current_step = 0
    cumulative_winnings = 0.0
    running = True

    opponent_patterns: Dict = {}
    fake_signal_history = deque(maxlen=100)
    manipulation_attempts = 0


config = YuichiConfig()

# ---------------------------------------------------------------------
# EXCHANGE WRAPPER
# ---------------------------------------------------------------------

class EliteExchange:
    def __init__(self):
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        self.rate_limit_buffer = 1.2

    def fetch_with_retry(self, func, *args, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded:
                wait_time = (2 ** attempt) * self.rate_limit_buffer
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds")
                time.sleep(wait_time)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Exchange call failed after {max_retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        return self.fetch_with_retry(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)

    def fetch_order_book(self, symbol, limit=20):
        return self.fetch_with_retry(self.exchange.fetch_order_book, symbol, limit=limit)


exchange = EliteExchange()

# ---------------------------------------------------------------------
# PERFORMANCE TRACKING
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
        self.fake_signals_filtered = 0

        self.winning_patterns: Dict[str, int] = {}
        self.losing_patterns: Dict[str, int] = {}

    def log_trade(self, trade_data):
        self.trades.append(trade_data)

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
            self.consecutive_losses = max(self.consecutive_losses, abs(self.current_streak))

        total = self.wins + self.losses
        self.win_rate = (self.wins / total * 100.0) if total > 0 else 0.0

        if "pattern" in trade_data:
            pattern_dict = self.winning_patterns if trade_data["profit_loss"] > 0 else self.losing_patterns
            pat = trade_data["pattern"]
            pattern_dict[pat] = pattern_dict.get(pat, 0) + 1


performance = PerformanceTracker()

# ---------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
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


def calculate_fear_greed_index(df):
    rsi = df["RSI"].iloc[-1]
    volatility = df["ATR"].iloc[-1] / df["close"].iloc[-1] * 100.0
    rsi_score = (rsi / 100.0) * 50.0
    vol_score = max(0.0, 50.0 - (volatility * 5.0))
    fear_greed = rsi_score + vol_score
    return min(100.0, max(0.0, fear_greed))


# ---------------------------------------------------------------------
# PSYCHOLOGY / TRAPS
# ---------------------------------------------------------------------

def detect_bull_bear_trap(df):
    if len(df) < 20:
        return None, 0.0

    recent_high = df["high"].iloc[-20:-1].max()
    recent_low = df["low"].iloc[-20:-1].min()
    current_price = df["close"].iloc[-1]
    current_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].iloc[-20:-1].mean()

    if current_price > recent_high and current_volume < avg_volume * 0.8:
        trap_strength = (current_price - recent_high) / recent_high
        logger.info(f"BULL TRAP DETECTED: strength {trap_strength:.2%}")
        performance.trap_avoidances += 1
        return TrapType.BULL_TRAP, trap_strength

    if current_price < recent_low and current_volume < avg_volume * 0.8:
        trap_strength = (recent_low - current_price) / recent_low
        logger.info(f"BEAR TRAP DETECTED: strength {trap_strength:.2%}")
        performance.trap_avoidances += 1
        return TrapType.BEAR_TRAP, trap_strength

    return None, 0.0


def detect_liquidity_grab(df):
    if len(df) < 10:
        return False

    price_changes = df["close"].pct_change().iloc[-5:]
    volume_surge = df["volume"].iloc[-5:].max() > df["volume"].iloc[-20:-5].mean() * 2.0

    if price_changes.iloc[-3] < -0.01 and price_changes.iloc[-1] > 0.005 and volume_surge:
        logger.info("LIQUIDITY GRAB detected: stop loss hunt before reversal")
        return True

    return False


def analyze_whale_activity(order_book_data):
    if not order_book_data:
        return {"manipulation": False, "side": None, "strength": 0.0}

    bids = order_book_data.get("bids", [])
    asks = order_book_data.get("asks", [])

    if not bids or not asks:
        return {"manipulation": False, "side": None, "strength": 0.0}

    bid_volume = sum(b[1] for b in bids[:10])
    ask_volume = sum(a[1] for a in asks[:10])

    imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0

    if imbalance_ratio > 3.0:
        logger.info(f"WHALE BUY WALL: {imbalance_ratio:.1f}x imbalance")
        return {"manipulation": True, "side": "buy", "strength": imbalance_ratio}
    if imbalance_ratio < 0.33:
        logger.info(f"WHALE SELL WALL: {(1.0/imbalance_ratio):.1f}x imbalance")
        return {"manipulation": True, "side": "sell", "strength": 1.0 / imbalance_ratio}

    return {"manipulation": False, "side": None, "strength": 0.0}


def calculate_market_psychology(df, order_book_data=None):
    if df.empty or "RSI" not in df.columns:
        return MarketPsychology(
            sentiment=MarketSentiment.UNCERTAINTY,
            fear_greed_index=50.0,
            retail_positioning=50.0,
            smart_money_flow=0.0,
            volatility_regime="medium",
            manipulation_detected=False,
            trap_probability=0.0,
        )

    fear_greed = calculate_fear_greed_index(df)
    rsi = df["RSI"].iloc[-1]

    if fear_greed < 25.0:
        sentiment = MarketSentiment.FEAR
    elif fear_greed > 75.0:
        sentiment = MarketSentiment.GREED
    else:
        sentiment = MarketSentiment.UNCERTAINTY

    trap, trap_strength = detect_bull_bear_trap(df)
    liquidity_grab = detect_liquidity_grab(df)
    whale_info = analyze_whale_activity(order_book_data) if order_book_data else {"manipulation": False}

    manipulation_detected = trap is not None or liquidity_grab or whale_info["manipulation"]
    if manipulation_detected:
        sentiment = MarketSentiment.MANIPULATION

    atr_pct = (df["ATR"].iloc[-1] / df["close"].iloc[-1]) * 100.0
    if atr_pct < 0.5:
        vol_regime = "low"
    elif atr_pct < 1.5:
        vol_regime = "medium"
    elif atr_pct < 3.0:
        vol_regime = "high"
    else:
        vol_regime = "extreme"

    retail_long_pct = min(100.0, max(0.0, rsi))
    order_flow_delta = df["order_flow_delta"].iloc[-1] if "order_flow_delta" in df.columns else 0.0

    return MarketPsychology(
        sentiment=sentiment,
        fear_greed_index=fear_greed,
        retail_positioning=retail_long_pct,
        smart_money_flow=order_flow_delta,
        volatility_regime=vol_regime,
        manipulation_detected=manipulation_detected,
        trap_probability=trap_strength if trap else 0.0,
    )


# ---------------------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------------------

def fetch_multi_timeframe_data(symbol):
    data = {}
    for name, tf in config.timeframes.items():
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=200)
        if ohlcv:
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            data[name] = df
    return data


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

        exp12 = df["close"].ewm(span=12, adjust=False).mean()
        exp26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        return df
    except Exception as e:
        logger.error(f"Indicator calculation error: {e}")
        return df


# ---------------------------------------------------------------------
# ENTRY LOGIC
# ---------------------------------------------------------------------

def yuichi_entry_signal(multi_tf_data, psychology):
    if "tactical" not in multi_tf_data or multi_tf_data["tactical"].empty:
        return None, None, "No data"

    df = multi_tf_data["tactical"]
    strategic = multi_tf_data.get("strategic")
    oversight = multi_tf_data.get("oversight")

    if len(df) < 50:
        return None, None, "Insufficient data"

    price = df["close"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    vwap = df["VWAP"].iloc[-1]
    ema9 = df["EMA_9"].iloc[-1]
    ema21 = df["EMA_21"].iloc[-1]
    bb_upper = df["BB_Upper"].iloc[-1]
    bb_lower = df["BB_Lower"].iloc[-1]
    order_flow = df["order_flow_delta"].iloc[-1]

    confirmations = []
    signal_type = None

    if psychology.sentiment == MarketSentiment.FEAR and psychology.fear_greed_index < 30:
        confirmations.append("extreme_fear")
        signal_type = "buy"
    elif psychology.sentiment == MarketSentiment.GREED and psychology.fear_greed_index > 70:
        confirmations.append("extreme_greed")
        signal_type = "sell"

    if psychology.manipulation_detected:
        if psychology.trap_probability > 0 and price < bb_lower:
            confirmations.append("bear_trap_counter")
            signal_type = "buy"
            performance.manipulation_counters += 1
        elif psychology.trap_probability > 0 and price > bb_upper:
            confirmations.append("bull_trap_counter")
            signal_type = "sell"
            performance.manipulation_counters += 1

    if rsi < config.rsi_extreme_oversold:
        confirmations.append("rsi_extreme_oversold")
        signal_type = "buy"
    elif rsi > config.rsi_extreme_overbought:
        confirmations.append("rsi_extreme_overbought")
        signal_type = "sell"

    vwap_deviation = (price - vwap) / vwap
    if vwap_deviation < -0.01:
        confirmations.append("below_vwap")
        if signal_type != "sell":
            signal_type = "buy"
    elif vwap_deviation > 0.01:
        confirmations.append("above_vwap")
        if signal_type != "buy":
            signal_type = "sell"

    if ema9 > ema21 and price > ema9:
        confirmations.append("bullish_ema")
        if signal_type != "sell":
            signal_type = "buy"
    elif ema9 < ema21 and price < ema9:
        confirmations.append("bearish_ema")
        if signal_type != "buy":
            signal_type = "sell"

    if order_flow > 0 and confirmations:
        confirmations.append("positive_order_flow")
    elif order_flow < 0 and confirmations:
        confirmations.append("negative_order_flow")

    if strategic is not None and not strategic.empty:
        strategic_rsi = strategic["RSI"].iloc[-1]
        if signal_type == "buy" and strategic_rsi < 40:
            confirmations.append("strategic_alignment")
        elif signal_type == "sell" and strategic_rsi > 60:
            confirmations.append("strategic_alignment")

    if oversight is not None and not oversight.empty and len(oversight) >= 50:
        trend_up = False
        if "SMA_200" in oversight.columns:
            trend_up = oversight["SMA_50"].iloc[-1] > oversight["SMA_200"].iloc[-1]
        if signal_type == "buy" and trend_up:
            confirmations.append("uptrend_confirmed")
        elif signal_type == "sell" and not trend_up:
            confirmations.append("downtrend_confirmed")

    if len(confirmations) >= config.min_confirmation_layers and signal_type:
        logger.info(f"YUICHI SIGNAL: {signal_type.upper()} with {len(confirmations)} layers: {confirmations}")
        performance.psychological_edges += 1
        return signal_type, confirmations, psychology

    if confirmations:
        logger.debug(f"Signal building but not ready: {confirmations}")

    return None, None, "Insufficient confirmation"


# ---------------------------------------------------------------------
# RISK MANAGEMENT
# ---------------------------------------------------------------------

def calculate_position_size(psychology, step):
    base_size = config.martingale_steps[min(step, len(config.martingale_steps) - 1)]

    if psychology.volatility_regime == "extreme":
        return base_size * 0.5
    if psychology.volatility_regime == "high":
        return base_size * 0.7

    if psychology.manipulation_detected and performance.psychological_edges > 5:
        return base_size * 1.2

    return base_size


def calculate_dynamic_sl_tp(entry_price, atr, signal_type, psychology, confirmations):
    sl_mult = 1.0
    tp_mult = 1.5

    if psychology.volatility_regime == "low":
        sl_mult = 0.8
        tp_mult = 1.2
    elif psychology.volatility_regime == "high":
        sl_mult = 1.3
        tp_mult = 2.0
    elif psychology.volatility_regime == "extreme":
        sl_mult = 1.5
        tp_mult = 2.5

    if len(confirmations) >= 6:
        tp_mult *= 1.3

    if signal_type == "buy":
        stop_loss = entry_price - (atr * sl_mult)
        take_profit = entry_price + (atr * tp_mult)
    else:
        stop_loss = entry_price + (atr * sl_mult)
        take_profit = entry_price - (atr * tp_mult)

    return stop_loss, take_profit


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


trade_state = TradeState()

# price history for chart
price_history = []


# ---------------------------------------------------------------------
# GUI HELPERS
# ---------------------------------------------------------------------

def log_result(msg, log_box):
    try:
        with open("yuichi_trades.txt", "a") as f:
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
            box.insert(tk.END, "Martingale Steps\n" + "=" * 30 + "\n\n")
            for i, step in enumerate(config.martingale_steps):
                marker = ">" if i == config.current_step else " "
                box.insert(tk.END, f"{marker} Step {i + 1}: ${step:.0f}\n")
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
            box.insert(tk.END, f"Psych Edges: {performance.psychological_edges}\n")
            box.insert(tk.END, f"Traps Avoided: {performance.trap_avoidances}\n")
            box.insert(tk.END, f"Whale Counters: {performance.manipulation_counters}\n\n")
            box.insert(tk.END, f"Best Streak: {performance.consecutive_wins}\n")
            box.insert(tk.END, f"Worst Streak: {performance.consecutive_losses}\n")
            box.config(state=tk.DISABLED)
    except Exception:
        pass


def update_psychology_panel(box, psychology):
    try:
        if box:
            box.config(state=tk.NORMAL)
            box.delete(1.0, tk.END)
            box.insert(tk.END, "Market Psychology\n" + "=" * 30 + "\n\n")
            box.insert(tk.END, f"Sentiment: {psychology.sentiment.value.upper()}\n")
            box.insert(tk.END, f"F&G Index: {psychology.fear_greed_index:.0f}/100\n")
            box.insert(tk.END, f"Volatility: {psychology.volatility_regime}\n")
            box.insert(tk.END, f"Manipulation: {'YES' if psychology.manipulation_detected else 'NO'}\n")
            box.insert(tk.END, f"Trap Risk: {psychology.trap_probability:.1%}\n")
            box.config(state=tk.DISABLED)
    except Exception:
        pass


def update_chart(ax, canvas):
    if not price_history:
        return

    ax.clear()
    ax.plot(price_history, label="Price Evolution")

    if trade_state.entry_price is not None:
        ax.axhline(trade_state.entry_price, linestyle="--", label="Entry Price", color="blue")
    if trade_state.stop_loss is not None:
        ax.axhline(trade_state.stop_loss, linestyle="--", label="Stop Loss", color="red")
    if trade_state.take_profit is not None:
        ax.axhline(trade_state.take_profit, linestyle="--", label="Take Profit", color="green")

    ax.legend(loc="best")
    canvas.draw_idle()


# ---------------------------------------------------------------------
# CLOSE TRADE
# ---------------------------------------------------------------------

def close_trade(reason, profit_loss, exit_price, log_box, stats_box, step_box, winnings_var):
    config.capital += profit_loss
    config.cumulative_winnings += profit_loss

    if profit_loss > 0:
        config.current_step = 0
    else:
        config.current_step = min(config.current_step + 1, len(config.martingale_steps) - 1)
        config.daily_loss += abs(profit_loss)

    trade_data = {
        "timestamp": datetime.now(),
        "type": trade_state.type,
        "entry": trade_state.entry_price,
        "exit": exit_price,
        "profit_loss": profit_loss,
        "result": "WIN" if profit_loss > 0 else "LOSS",
        "confirmations": trade_state.confirmations,
        "psychology": trade_state.psychology_snapshot.sentiment.value
        if trade_state.psychology_snapshot
        else "unknown",
        "pattern": "_".join(trade_state.confirmations[:3]),
    }
    performance.log_trade(trade_data)

    log_msg = (
        f"\n{'='*70}\n"
        f"{reason}\n"
        f"{'='*70}\n"
        f"Entry: ${trade_state.entry_price:.2f} -> Exit: ${exit_price:.2f}\n"
        f"P/L: ${profit_loss:.2f} ({'WIN' if profit_loss > 0 else 'LOSS'})\n"
        f"Capital: ${config.capital:.2f}\n"
        f"Total Winnings: ${config.cumulative_winnings:.2f}\n"
        f"Win Rate: {performance.win_rate:.1f}%\n"
        f"Psych Edges: {performance.psychological_edges}\n"
        f"Traps Avoided: {performance.trap_avoidances}\n"
        f"{'='*70}\n"
    )
    log_result(log_msg, log_box)

    trade_state.active = False
    trade_state.type = None
    config.trades_executed_today += 1

    update_stats_panel(stats_box)
    update_step_panel(step_box)
    update_winnings_box(winnings_var)


# ---------------------------------------------------------------------
# MAIN TRADING LOOP
# ---------------------------------------------------------------------

def execute_yuichi_strategy(
    log_box, ax, canvas, winnings_var, step_box, stats_box, psych_box
):
    logger.info("=" * 70)
    logger.info("YUICHI KATAGIRI BOT ACTIVATED")
    logger.info("Philosophy: Outsmart the market makers, exploit their psychology")
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
            tactical_df = multi_tf_data.get("tactical")
            if tactical_df is None or tactical_df.empty:
                time.sleep(5)
                continue

            for tf_name in multi_tf_data:
                multi_tf_data[tf_name] = calculate_all_indicators(multi_tf_data[tf_name])

            df = multi_tf_data["tactical"]
            if "ATR" not in df.columns or df["ATR"].isna().all():
                time.sleep(5)
                continue

            order_book = exchange.fetch_order_book(config.symbols[0], limit=20)
            psychology = calculate_market_psychology(df, order_book)

            current_price = df["close"].iloc[-1]
            atr = df["ATR"].iloc[-1]

            price_history.append(current_price)
            if len(price_history) > 2500:
                price_history.pop(0)

            update_psychology_panel(psych_box, psychology)

            if not trade_state.active:
                signal_type, confirmations, _ = yuichi_entry_signal(
                    multi_tf_data, psychology
                )

                if signal_type and confirmations:
                    position_size = calculate_position_size(psychology, config.current_step)

                    trade_state.active = True
                    trade_state.type = signal_type
                    trade_state.entry_price = current_price
                    trade_state.entry_time = datetime.now()
                    trade_state.confirmations = confirmations
                    trade_state.psychology_snapshot = psychology

                    trade_state.stop_loss, trade_state.take_profit = calculate_dynamic_sl_tp(
                        current_price, atr, signal_type, psychology, confirmations
                    )

                    trade_state.highest_price = current_price
                    trade_state.lowest_price = current_price
                    trade_state.trailing_stop = None

                    log_msg = (
                        f"\n{'='*70}\n"
                        f"YUICHI MOVE: {signal_type.upper()}\n"
                        f"{'='*70}\n"
                        f"Price: ${current_price:.2f}\n"
                        f"Position: ${position_size:.2f}\n"
                        f"SL: ${trade_state.stop_loss:.2f} | "
                        f"TP: ${trade_state.take_profit:.2f}\n"
                        f"Psychology: {psychology.sentiment.value.upper()}\n"
                        f"F&G Index: {psychology.fear_greed_index:.0f}\n"
                        f"Confirmations ({len(confirmations)}): "
                        f"{', '.join(confirmations[:4])}\n"
                        f"Martingale Step: {config.current_step + 1}\n"
                        f"Manipulation Detected: "
                        f"{'YES' if psychology.manipulation_detected else 'NO'}\n"
                        f"{'='*70}"
                    )
                    log_result(log_msg, log_box)

            else:
                trade_duration = (
                    datetime.now() - trade_state.entry_time
                ).total_seconds() / 60.0

                if trade_duration > config.max_hold_time_minutes:
                    profit_loss = (
                        current_price - trade_state.entry_price
                        if trade_state.type == "buy"
                        else trade_state.entry_price - current_price
                    )
                    close_trade(
                        "Time limit",
                        profit_loss,
                        current_price,
                        log_box,
                        stats_box,
                        step_box,
                        winnings_var,
                    )
                    update_chart(ax, canvas)
                    continue

                if trade_state.type == "buy":
                    if current_price > trade_state.highest_price:
                        trade_state.highest_price = current_price
                        progress_to_tp = (
                            current_price - trade_state.entry_price
                        ) / (trade_state.take_profit - trade_state.entry_price)
                        if progress_to_tp >= 0.5:
                            trade_state.trailing_stop = (
                                trade_state.highest_price - (atr * 0.5)
                            )

                    if current_price >= trade_state.take_profit:
                        profit_loss = trade_state.take_profit - trade_state.entry_price
                        close_trade(
                            "TAKE PROFIT",
                            profit_loss,
                            trade_state.take_profit,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif trade_state.trailing_stop and current_price <= trade_state.trailing_stop:
                        profit_loss = trade_state.trailing_stop - trade_state.entry_price
                        close_trade(
                            "Trailing Stop (profit locked)",
                            profit_loss,
                            trade_state.trailing_stop,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif current_price <= trade_state.stop_loss:
                        profit_loss = trade_state.stop_loss - trade_state.entry_price
                        close_trade(
                            "Stop Loss",
                            profit_loss,
                            trade_state.stop_loss,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )

                else:
                    if current_price < trade_state.lowest_price:
                        trade_state.lowest_price = current_price
                        progress_to_tp = (
                            trade_state.entry_price - current_price
                        ) / (trade_state.entry_price - trade_state.take_profit)
                        if progress_to_tp >= 0.5:
                            trade_state.trailing_stop = (
                                trade_state.lowest_price + (atr * 0.5)
                            )

                    if current_price <= trade_state.take_profit:
                        profit_loss = trade_state.entry_price - trade_state.take_profit
                        close_trade(
                            "TAKE PROFIT",
                            profit_loss,
                            trade_state.take_profit,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif trade_state.trailing_stop and current_price >= trade_state.trailing_stop:
                        profit_loss = trade_state.entry_price - trade_state.trailing_stop
                        close_trade(
                            "Trailing Stop (profit locked)",
                            profit_loss,
                            trade_state.trailing_stop,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif current_price >= trade_state.stop_loss:
                        profit_loss = trade_state.entry_price - trade_state.stop_loss
                        close_trade(
                            "Stop Loss",
                            profit_loss,
                            trade_state.stop_loss,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )

            update_winnings_box(winnings_var)
            update_step_panel(step_box)
            update_stats_panel(stats_box)
            update_chart(ax, canvas)

            time.sleep(3)

        except Exception as e:
            logger.error(f"Trading loop error: {e}", exc_info=True)
            time.sleep(5)


# ---------------------------------------------------------------------
# GUI CREATION
# ---------------------------------------------------------------------

def create_gui():
    root = tk.Tk()
    root.title("Trading Bot v11: Yuichi Katagiri Psychological Warfare")
    root.geometry("1200x900")

    # Trade log
    tk.Label(root, text="Trade Log", font=("Arial", 12, "bold")).grid(
        row=0, column=0, sticky="w", padx=10, pady=(10, 0)
    )
    log_box = scrolledtext.ScrolledText(
        root,
        width=80,
        height=12,
        font=("Courier", 9),
        bg="#f0f0f0",
        fg="#000000",
    )
    log_box.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="nsew")
    log_box.config(state=tk.DISABLED)

    # Cumulative winnings
    tk.Label(root, text="Cumulative Winnings", font=("Arial", 12, "bold")).grid(
        row=2, column=0, sticky="w", padx=10, pady=(10, 0)
    )
    winnings_var = tk.StringVar()
    winnings_entry = tk.Entry(
        root,
        textvariable=winnings_var,
        font=("Arial", 14, "bold"),
        width=15,
        justify="center",
    )
    winnings_entry.grid(row=2, column=1, sticky="w", padx=10, pady=(10, 0))

    # Stats panel
    stats_frame = tk.LabelFrame(root, text="Stats", font=("Arial", 10, "bold"))
    stats_frame.grid(row=3, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")
    stats_box = tk.Text(
        stats_frame,
        height=15,
        width=30,
        font=("Courier", 9),
        bg="#f0f0f0",
        fg="#000000",
    )
    stats_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    stats_box.config(state=tk.DISABLED)

    # Martingale steps
    step_frame = tk.LabelFrame(root, text="Martingale Steps", font=("Arial", 10, "bold"))
    step_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
    step_box = tk.Text(
        step_frame,
        height=12,
        width=30,
        font=("Courier", 9),
        bg="#f0f0f0",
        fg="#000000",
    )
    step_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    step_box.config(state=tk.DISABLED)

    # Psychology panel
    psych_frame = tk.LabelFrame(root, text="Psychology", font=("Arial", 10, "bold"))
    psych_frame.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")
    psych_box = tk.Text(
        psych_frame,
        height=12,
        width=30,
        font=("Courier", 9),
        bg="#f0f0f0",
        fg="#000000",
    )
    psych_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    psych_box.config(state=tk.DISABLED)

    # Chart
    fig, ax = plt.subplots(figsize=(8, 4))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    tk.Label(root, text="Price Chart", font=("Arial", 12, "bold")).grid(
        row=5, column=0, columnspan=3, pady=(10, 0)
    )
    canvas_widget.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.grid(row=7, column=0, columnspan=3, pady=10)

    def stop():
        config.running = False
        log_result("Yuichi bot stopped by user.", log_box)

    tk.Button(
        btn_frame,
        text="Stop",
        command=stop,
        bg="#c0392b",
        fg="white",
        font=("Arial", 12, "bold"),
        width=12,
    ).pack(side=tk.LEFT, padx=10)

    tk.Button(
        btn_frame,
        text="Quit",
        command=root.destroy,
        bg="#7f8c8d",
        fg="white",
        font=("Arial", 12, "bold"),
        width=12,
    ).pack(side=tk.LEFT, padx=10)

    # Initialize panels
    update_winnings_box(winnings_var)
    update_step_panel(step_box)
    update_stats_panel(stats_box)

    # Start trading thread
    t = threading.Thread(
        target=execute_yuichi_strategy,
        args=(log_box, ax, canvas, winnings_var, step_box, stats_box, psych_box),
        daemon=True,
    )
    t.start()

    root.mainloop()


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Initializing Yuichi Katagiri Trading System...")
    logger.info(f"Capital: ${config.capital}")
    logger.info("Philosophy: Psychological warfare against market makers")
    logger.info("Strategy: Multi-layer deception and counter-manipulation")
    create_gui()
