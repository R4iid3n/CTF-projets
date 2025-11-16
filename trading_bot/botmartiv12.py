# Trading Bot v12 - "YUICHI KATAGIRI SPECIAL METHOD"
# Psychological Warfare with Hedge Bets and Battle-Level Martingale

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
        logging.FileHandler("yuichi_bot_v12.log"),
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

    # base size per martingale step (battle-level)
    martingale_steps = [8, 12, 25, 50, 100, 200, 400, 800]
    max_martingale_steps = 6
    current_step = 0  # index dans martingale_steps

    # battle-level multiplicateurs
    battle_steps = [1.0, 1.3, 1.6, 2.0]
    battle_step = 0  # index dans battle_steps

    rsi_period = 14
    rsi_extreme_oversold = 30
    rsi_extreme_overbought = 80

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
        self.fake_signals_filtered = 0

        self.winning_patterns: Dict[str, int] = {}
        self.losing_patterns: Dict[str, int] = {}

        self.battles = 0

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


def calculate_fear_greed_index(df):
    rsi = df["RSI"].iloc[-1]
    volatility = df["ATR"].iloc[-1] / df["close"].iloc[-1] * 100.0
    rsi_score = (rsi / 100.0) * 50.0
    vol_score = max(0.0, 50.0 - (volatility * 5.0))
    fear_greed = rsi_score + vol_score
    return min(100.0, max(0.0, fear_greed))


# ---------------------------------------------------------------------
# TRAPS / PSYCHO
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
    """
    Whale detection robuste :
    - ignore les carnets trop fins
    - ratio clampé pour éviter les 200x irréalistes
    """
    if not order_book_data:
        return {'manipulation': False, 'side': None, 'strength': 0.0}

    bids = order_book_data.get('bids', [])
    asks = order_book_data.get('asks', [])

    if not bids or not asks:
        return {'manipulation': False, 'side': None, 'strength': 0.0}

    bid_volume = sum(b[1] for b in bids[:10])
    ask_volume = sum(a[1] for a in asks[:10])

    total_depth = bid_volume + ask_volume
    min_depth = 5.0  # ex: 5 BTC sur les 10 premiers niveaux

    if total_depth < min_depth or bid_volume == 0 or ask_volume == 0:
        return {'manipulation': False, 'side': None, 'strength': 0.0}

    imbalance_ratio = bid_volume / ask_volume
    max_ratio = 10.0

    if imbalance_ratio > max_ratio:
        imbalance_ratio = max_ratio

    if imbalance_ratio > 3.0:
        logger.info(f"WHALE BUY WALL: {imbalance_ratio:.1f}x imbalance")
        return {'manipulation': True, 'side': 'buy', 'strength': imbalance_ratio}

    if imbalance_ratio < 1.0 / 3.0:
        strength = min(max_ratio, 1.0 / imbalance_ratio)
        logger.info(f"WHALE SELL WALL: {strength:.1f}x imbalance")
        return {'manipulation': True, 'side': 'sell', 'strength': strength}

    return {'manipulation': False, 'side': None, 'strength': 0.0}


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

    # 1) Psychologie globale
    if (
        psychology.sentiment == MarketSentiment.FEAR
        and psychology.fear_greed_index < 45
    ):
        confirmations.append("extreme_fear")
        signal_type = "buy"
    elif (
        psychology.sentiment == MarketSentiment.GREED
        and psychology.fear_greed_index > 85
    ):
        confirmations.append("extreme_greed")
        signal_type = "sell"

    # 2) Traps / manipulation
    if psychology.manipulation_detected:
        if psychology.trap_probability > 0.02 and price < bb_lower:
            confirmations.append("bear_trap_counter")
            signal_type = "buy"
            performance.manipulation_counters += 1
        elif psychology.trap_probability > 0.02 and price > bb_upper:
            confirmations.append("bull_trap_counter")
            signal_type = "sell"
            performance.manipulation_counters += 1

    # 3) RSI extrêmes
    if rsi < config.rsi_extreme_oversold:
        confirmations.append("rsi_extreme_oversold")
        if signal_type != "sell":
            signal_type = "buy"
    elif rsi > config.rsi_extreme_overbought:
        confirmations.append("rsi_extreme_overbought")
        if signal_type != "buy":
            signal_type = "sell"

    # 4) VWAP
    vwap_deviation = (price - vwap) / vwap
    if vwap_deviation < -0.01:
        confirmations.append("below_vwap")
        if signal_type != "sell":
            signal_type = "buy"
    elif vwap_deviation > 0.01:
        confirmations.append("above_vwap")
        if signal_type != "buy":
            signal_type = "sell"

    # 5) EMA trend local
    if ema9 > ema21 and price > ema9:
        confirmations.append("bullish_ema")
        if signal_type == "sell":
            if "extreme_greed" not in confirmations and "bull_trap_counter" not in confirmations:
                signal_type = "buy"
        elif signal_type is None:
            signal_type = "buy"
    elif ema9 < ema21 and price < ema9:
        confirmations.append("bearish_ema")
        if signal_type == "buy":
            if "extreme_fear" not in confirmations and "bear_trap_counter" not in confirmations:
                signal_type = "sell"
        elif signal_type is None:
            signal_type = "sell"

    # 6) Order flow
    if order_flow > 0 and confirmations:
        confirmations.append("positive_order_flow")
    elif order_flow < 0 and confirmations:
        confirmations.append("negative_order_flow")

    # 7) Alignement multi-timeframe
    trend_up_oversight = None
    if oversight is not None and not oversight.empty and "SMA_200" in oversight.columns:
        try:
            trend_up_oversight = oversight["SMA_50"].iloc[-1] > oversight["SMA_200"].iloc[-1]
        except Exception:
            trend_up_oversight = None

    if strategic is not None and not strategic.empty:
        strategic_rsi = strategic["RSI"].iloc[-1]
        if signal_type == "buy" and strategic_rsi < 55:
            confirmations.append("strategic_alignment")
        elif signal_type == "sell" and strategic_rsi > 45:
            confirmations.append("strategic_alignment")

    if trend_up_oversight is not None:
        if signal_type == "buy" and trend_up_oversight:
            confirmations.append("uptrend_confirmed")
        elif signal_type == "sell" and not trend_up_oversight:
            confirmations.append("downtrend_confirmed")

        # bloque les SELL contre tendance 1h sans trap fort
        if signal_type == "sell" and trend_up_oversight:
            strong_trap_short = (
                psychology.manipulation_detected
                and psychology.trap_probability > 0.05
                and price > bb_upper
            )
            if not strong_trap_short:
                return None, None, "Sell blocked by 1h uptrend"

    # 8) Validation finale
    if not signal_type or len(confirmations) < config.min_confirmation_layers:
        return None, None, "Insufficient confirmation"

    logger.info(
        f"YUICHI SIGNAL: {signal_type.upper()} with {len(confirmations)} layers: {confirmations}"
    )
    performance.psychological_edges += 1
    return signal_type, confirmations, psychology


# ---------------------------------------------------------------------
# RISK / SL-TP
# ---------------------------------------------------------------------

def calculate_position_size(psychology):
    base_index = min(config.current_step, len(config.martingale_steps) - 1)
    base_size = config.martingale_steps[base_index]

    battle_index = min(config.battle_step, len(config.battle_steps) - 1)
    battle_mult = config.battle_steps[battle_index]

    size = base_size * battle_mult

    if psychology.volatility_regime == "extreme":
        size *= 0.5
    elif psychology.volatility_regime == "high":
        size *= 0.7

    if psychology.manipulation_detected and performance.psychological_edges > 5:
        size *= 1.1

    # taille notionnelle (USDT)
    return size


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
# TRADE / BATTLE STATE
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

        # tailles notionnelles
        self.main_size = 0.0  # USDT

        # Hedge bet
        self.hedge_active = False
        self.hedge_type = None
        self.hedge_entry_price = None
        self.hedge_size = 0.0  # USDT

        # PnL cumulé sur la "bataille" (main + hedge) en USDT
        self.battle_pnl = 0.0


trade_state = TradeState()
price_history = []  # pour le graphique


# ---------------------------------------------------------------------
# GUI HELPERS
# ---------------------------------------------------------------------

def log_result(msg, log_box):
    try:
        with open("yuichi_trades_v12.txt", "a") as f:
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

    extra_levels = []

    if trade_state.entry_price is not None:
        ax.axhline(trade_state.entry_price, linestyle="--", label="Entry Price", color="blue")
        extra_levels.append(trade_state.entry_price)

    if trade_state.stop_loss is not None:
        ax.axhline(trade_state.stop_loss, linestyle="--", label="Stop Loss", color="red")
        extra_levels.append(trade_state.stop_loss)

    if trade_state.take_profit is not None:
        ax.axhline(trade_state.take_profit, linestyle="--", label="Take Profit", color="green")
        extra_levels.append(trade_state.take_profit)

    y_values = list(price_history) + extra_levels
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        padding = (y_max - y_min) * 0.1
        if padding == 0:
            padding = max(abs(y_min) * 0.001, 1)
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.legend(loc="best")
    canvas.draw_idle()


# ---------------------------------------------------------------------
# SL/TP SHRINK DANS LE TEMPS
# ---------------------------------------------------------------------

def tighten_sl_tp_over_time(trade_duration_minutes):
    if not trade_state.active:
        return
    if trade_state.original_stop_loss is None or trade_state.original_take_profit is None:
        return
    if config.max_hold_time_minutes <= 0:
        return

    progress = min(1.0, max(0.0, trade_duration_minutes / config.max_hold_time_minutes))
    shrink = 1.0 - 0.7 * progress
    shrink = max(0.3, shrink)

    entry = trade_state.entry_price

    if trade_state.type == "buy":
        dist_sl0 = entry - trade_state.original_stop_loss
        dist_tp0 = trade_state.original_take_profit - entry
        new_sl = entry - dist_sl0 * shrink
        new_tp = entry + dist_tp0 * shrink
        if new_sl < entry:
            trade_state.stop_loss = new_sl
        if new_tp > entry:
            trade_state.take_profit = new_tp

    elif trade_state.type == "sell":
        dist_sl0 = trade_state.original_stop_loss - entry
        dist_tp0 = entry - trade_state.original_take_profit
        new_sl = entry + dist_sl0 * shrink
        new_tp = entry - dist_tp0 * shrink
        if new_sl > entry:
            trade_state.stop_loss = new_sl
        if new_tp < entry:
            trade_state.take_profit = new_tp


# ---------------------------------------------------------------------
# BATTLE-LEVEL MARTINGALE
# ---------------------------------------------------------------------

def on_battle_end(battle_pnl):
    # battle-level martingale + base step
    if battle_pnl > 0:
        config.battle_step = 0
        config.current_step = 0
    else:
        config.battle_step = min(config.battle_step + 1, len(config.battle_steps) - 1)
        config.current_step = min(config.current_step + 1, len(config.martingale_steps) - 1)
        config.daily_loss += abs(battle_pnl)

    performance.battles += 1


# ---------------------------------------------------------------------
# CLOSE TRADE (MAIN BATTLE)
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
        "psychology": trade_state.psychology_snapshot.sentiment.value
        if trade_state.psychology_snapshot
        else "unknown",
        "pattern": "_".join(trade_state.confirmations[:3]) if trade_state.confirmations else "none",
    }
    performance.log_trade(trade_data)

    log_msg = (
        f"\n{'='*70}\n"
        f"{reason} [{result_label}]\n"
        f"{'='*70}\n"
        f"Entry: ${trade_state.entry_price:.2f} -> Exit: ${exit_price:.2f}\n"
        f"Battle P/L (USDT): ${battle_pnl:.2f}\n"
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

    trade_state.hedge_active = False
    trade_state.hedge_type = None
    trade_state.hedge_entry_price = None
    trade_state.hedge_size = 0.0

    trade_state.battle_pnl = 0.0
    trade_state.main_size = 0.0

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
    logger.info("YUICHI KATAGIRI BOT ACTIVATED (v12)")
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

            # ------------------ PAS DE POSITION ACTIVE ------------------
            if not trade_state.active:
                signal_type, confirmations, _ = yuichi_entry_signal(
                    multi_tf_data, psychology
                )

                if signal_type and confirmations:
                    position_size = calculate_position_size(psychology)  # notionnel USDT

                    trade_state.active = True
                    trade_state.type = signal_type
                    trade_state.entry_price = current_price
                    trade_state.entry_time = datetime.now()
                    trade_state.confirmations = confirmations
                    trade_state.psychology_snapshot = psychology

                    trade_state.stop_loss, trade_state.take_profit = calculate_dynamic_sl_tp(
                        current_price, atr, signal_type, psychology, confirmations
                    )
                    trade_state.original_stop_loss = trade_state.stop_loss
                    trade_state.original_take_profit = trade_state.take_profit

                    trade_state.highest_price = current_price
                    trade_state.lowest_price = current_price
                    trade_state.trailing_stop = None

                    trade_state.hedge_active = False
                    trade_state.hedge_type = None
                    trade_state.hedge_entry_price = None
                    trade_state.hedge_size = 0.0
                    trade_state.battle_pnl = 0.0

                    trade_state.main_size = position_size  # on stocke la taille notionnelle

                    log_msg = (
                        f"\n{'='*70}\n"
                        f"YUICHI MAIN MOVE: {signal_type.upper()}\n"
                        f"{'='*70}\n"
                        f"Price: ${current_price:.2f}\n"
                        f"Position Size (notional): ${position_size:.2f}\n"
                        f"SL: ${trade_state.stop_loss:.2f} | "
                        f"TP: ${trade_state.take_profit:.2f}\n"
                        f"Psychology: {psychology.sentiment.value.upper()}\n"
                        f"F&G Index: {psychology.fear_greed_index:.0f}\n"
                        f"Confirmations ({len(confirmations)}): "
                        f"{', '.join(confirmations[:6])}\n"
                        f"Martingale Step: {config.current_step + 1}\n"
                        f"Battle Step: {config.battle_step + 1}\n"
                        f"Manipulation Detected: "
                        f"{'YES' if psychology.manipulation_detected else 'NO'}\n"
                        f"{'='*70}"
                    )
                    log_result(log_msg, log_box)

            # ------------------ POSITION ACTIVE ------------------
            else:
                trade_duration = (
                    datetime.now() - trade_state.entry_time
                ).total_seconds() / 60.0

                # shrink dynamique SL/TP
                tighten_sl_tp_over_time(trade_duration)

                # HEDGE BET LOGIC
                if not trade_state.hedge_active:
                    if trade_state.type == "buy":
                        adverse_move_pct = (trade_state.entry_price - current_price) / trade_state.entry_price
                    else:
                        adverse_move_pct = (current_price - trade_state.entry_price) / trade_state.entry_price

                    if (
                        adverse_move_pct > 0.003  # 0.3% contre toi
                        and psychology.manipulation_detected
                    ):
                        trade_state.hedge_active = True
                        trade_state.hedge_type = "sell" if trade_state.type == "buy" else "buy"
                        trade_state.hedge_entry_price = current_price
                        # hedge = 50% de la taille main
                        hedge_size = trade_state.main_size * 0.5
                        trade_state.hedge_size = hedge_size

                        log_result(
                            f"HEDGE BET ACTIVATED: {trade_state.hedge_type.upper()} at ${current_price:.2f} "
                            f"against main {trade_state.type.upper()} entry ${trade_state.entry_price:.2f} "
                            f"(hedge notional: ${hedge_size:.2f})",
                            log_box
                        )

                # Gestion du hedge actif
                if trade_state.hedge_active and trade_state.hedge_entry_price:
                    if trade_state.hedge_type == "sell":
                        price_diff = trade_state.hedge_entry_price - current_price
                    else:
                        price_diff = current_price - trade_state.hedge_entry_price

                    # quantité en BTC pour le hedge
                    hedge_qty = trade_state.hedge_size / trade_state.hedge_entry_price if trade_state.hedge_entry_price > 0 else 0.0
                    hedge_pnl = price_diff * hedge_qty  # en USDT

                    # fermeture du hedge si le prix revient vers l'entry main
                    if trade_state.type == "buy":
                        close_hedge_condition = current_price >= trade_state.entry_price * 0.999
                    else:
                        close_hedge_condition = current_price <= trade_state.entry_price * 1.001

                    if close_hedge_condition:
                        trade_state.battle_pnl += hedge_pnl
                        log_result(
                            f"HEDGE CLOSED at ${current_price:.2f} | Hedge PnL (USDT): ${hedge_pnl:.2f} "
                            f"(Battle PnL so far: ${trade_state.battle_pnl:.2f})",
                            log_box
                        )
                        trade_state.hedge_active = False
                        trade_state.hedge_type = None
                        trade_state.hedge_entry_price = None
                        trade_state.hedge_size = 0.0

                # gestion du temps max
                if trade_duration > config.max_hold_time_minutes:
                    exit_price = current_price
                    if trade_state.entry_price and trade_state.main_size > 0:
                        main_qty = trade_state.main_size / trade_state.entry_price
                    else:
                        main_qty = 0.0

                    if trade_state.type == "buy":
                        price_diff = exit_price - trade_state.entry_price
                    else:
                        price_diff = trade_state.entry_price - exit_price

                    main_pnl = price_diff * main_qty
                    battle_pnl = trade_state.battle_pnl + main_pnl

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

                # gestion main TP / SL / trailing
                if trade_state.entry_price and trade_state.main_size > 0:
                    main_qty = trade_state.main_size / trade_state.entry_price
                else:
                    main_qty = 0.0

                if trade_state.type == "buy":
                    if current_price > trade_state.highest_price:
                        trade_state.highest_price = current_price
                        progress_to_tp = (
                            current_price - trade_state.entry_price
                        ) / max(1e-9, (trade_state.take_profit - trade_state.entry_price))
                        if progress_to_tp >= 0.5:
                            trade_state.trailing_stop = (
                                trade_state.highest_price - (atr * 0.5)
                            )

                    if current_price >= trade_state.take_profit:
                        exit_price = trade_state.take_profit
                        price_diff = exit_price - trade_state.entry_price
                        main_pnl = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl
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
                        price_diff = exit_price - trade_state.entry_price
                        main_pnl = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl
                        close_trade(
                            "Trailing Stop (profit locked)",
                            battle_pnl,
                            exit_price,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif current_price <= trade_state.stop_loss:
                        exit_price = trade_state.stop_loss
                        price_diff = exit_price - trade_state.entry_price
                        main_pnl = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl
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
                    if current_price < trade_state.lowest_price:
                        trade_state.lowest_price = current_price
                        progress_to_tp = (
                            trade_state.entry_price - current_price
                        ) / max(1e-9, (trade_state.entry_price - trade_state.take_profit))
                        if progress_to_tp >= 0.5:
                            trade_state.trailing_stop = (
                                trade_state.lowest_price + (atr * 0.5)
                            )

                    if current_price <= trade_state.take_profit:
                        exit_price = trade_state.take_profit
                        price_diff = trade_state.entry_price - exit_price
                        main_pnl = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl
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
                        price_diff = trade_state.entry_price - exit_price
                        main_pnl = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl
                        close_trade(
                            "Trailing Stop (profit locked)",
                            battle_pnl,
                            exit_price,
                            log_box,
                            stats_box,
                            step_box,
                            winnings_var,
                        )
                    elif current_price >= trade_state.stop_loss:
                        exit_price = trade_state.stop_loss
                        price_diff = trade_state.entry_price - exit_price
                        main_pnl = price_diff * main_qty
                        battle_pnl = trade_state.battle_pnl + main_pnl
                        close_trade(
                            "Stop Loss",
                            battle_pnl,
                            exit_price,
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
# GUI
# ---------------------------------------------------------------------

def create_gui():
    root = tk.Tk()
    root.title("Trading Bot v12: Yuichi Katagiri Special Method")
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

    # Martingale / battle steps
    step_frame = tk.LabelFrame(root, text="Martingale & Battle Steps", font=("Arial", 10, "bold"))
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

    # Init panels
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
    logger.info("Initializing Yuichi Katagiri Trading System v12...")
    logger.info(f"Capital: ${config.capital}")
    logger.info("Philosophy: Psychological warfare against market makers")
    logger.info("Strategy: Multi-layer deception, hedge bets and battle-level martingale")
    create_gui()
