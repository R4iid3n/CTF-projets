# Trading Bot v13.5 - "Enhanced Method"
# v13 AMÉLIORÉE avec tous les axes d'amélioration
# GARDE la philosophie multi-confirmation MAIS optimise tout le reste

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
from typing import Dict, Optional
from enum import Enum

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - [ENHANCED] - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("yuichi_bot_v13_5.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ENUMS
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


class MarketCondition(Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    CHOPPY = "choppy"
    VOLATILE_EXTREME = "volatile_extreme"


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
# CONFIG - OPTIMISÉ
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

    # ✅ NOUVEAU: Frais réels
    trading_fee_rate = 0.001   # 0.1% Binance
    slippage_rate = 0.0003     # 0.03% slippage
    min_profit_threshold = 0.004  # 0.4% min pour couvrir frais

    max_daily_loss_pct = 0.20
    max_trades_per_day = 20  # Réduit de 25 à 20

    # Position sizing - Augmenté mais pas trop
    martingale_steps = [50, 75, 125, 200, 350, 600, 1000, 1600]
    max_martingale_steps = 6
    current_step = 0

    battle_steps = [1.0, 1.5, 2.25, 3.5]
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
                logger.warning(f"Rate limit - wait {wait_time}s")
                time.sleep(wait_time)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed: {e}")
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
        
        # Nouveau: tracking par type de setup
        self.gold_setups = 0  # Score 6.0+
        self.gold_wins = 0

    def log_trade(self, trade_data):
        self.trades.append(trade_data)
        
        if trade_data.get("score", 0) >= 6.0:
            self.gold_setups += 1
            if trade_data["profit_loss"] > 0:
                self.gold_wins += 1

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
        logger.error(f"Indicator error: {e}")
        return df

# ---------------------------------------------------------------------
# ✅ NOUVEAU: MARKET CONDITION DETECTION
# ---------------------------------------------------------------------

def detect_market_condition(df_1h, df_15m, df_5m) -> MarketCondition:
    """
    Détecte si le marché est tradeable
    """
    if df_1h is None or len(df_1h) < 50:
        return MarketCondition.UNCERTAIN
    
    # ATR sur 1h
    atr_1h_pct = (df_1h["ATR"].iloc[-1] / df_1h["close"].iloc[-1]) * 100
    
    # Trend 1h
    sma50 = df_1h["SMA_50"].iloc[-1]
    sma200 = df_1h["SMA_200"].iloc[-1]
    price = df_1h["close"].iloc[-1]
    
    # Wicks sur 5m (choppy detection)
    recent_wicks = 0
    if df_5m is not None and len(df_5m) >= 20:
        for i in range(-20, 0):
            candle_body = abs(df_5m["close"].iloc[i] - df_5m["open"].iloc[i])
            candle_range = df_5m["high"].iloc[i] - df_5m["low"].iloc[i]
            if candle_range > 0 and candle_body < candle_range * 0.3:
                recent_wicks += 1
    
    # FILTRE 1: Volatilité extrême
    if atr_1h_pct > 3.5:
        logger.warning(f"⚠️ Volatilité EXTREME: {atr_1h_pct:.2f}% - Pas de trades")
        return MarketCondition.VOLATILE_EXTREME
    
    # FILTRE 2: Marché choppy
    if recent_wicks > 12:  # >60% wicks
        logger.warning(f"⚠️ Marché CHOPPY: {recent_wicks}/20 wicks - Pas de trades")
        return MarketCondition.CHOPPY
    
    # Trending bull
    if sma50 > sma200 and price > sma50:
        return MarketCondition.TRENDING_BULL
    
    # Trending bear
    elif sma50 < sma200 and price < sma50:
        return MarketCondition.TRENDING_BEAR
    
    # Range
    else:
        return MarketCondition.RANGING


# ---------------------------------------------------------------------
# TRAP DETECTION
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
        logger.info(f"🎭 BULL TRAP: {trap_strength:.2%}")
        performance.trap_avoidances += 1
        return TrapType.BULL_TRAP, trap_strength

    if current_price < recent_low and current_volume < avg_volume * 0.8:
        trap_strength = (recent_low - current_price) / recent_low
        logger.info(f"🎭 BEAR TRAP: {trap_strength:.2%}")
        performance.trap_avoidances += 1
        return TrapType.BEAR_TRAP, trap_strength

    return None, 0.0


def detect_liquidity_grab(df):
    if len(df) < 10:
        return False

    price_changes = df["close"].pct_change().iloc[-5:]
    volume_surge = df["volume"].iloc[-5:].max() > df["volume"].iloc[-20:-5].mean() * 2.0

    if price_changes.iloc[-3] < -0.01 and price_changes.iloc[-1] > 0.005 and volume_surge:
        logger.info("🎭 LIQUIDITY GRAB detected")
        return True

    return False


def analyze_whale_activity(order_book_data):
    if not order_book_data:
        return {'manipulation': False, 'side': None, 'strength': 0.0}

    bids = order_book_data.get('bids', [])
    asks = order_book_data.get('asks', [])

    if not bids or not asks:
        return {'manipulation': False, 'side': None, 'strength': 0.0}

    bid_volume = sum(b[1] for b in bids[:10])
    ask_volume = sum(a[1] for a in asks[:10])

    total_depth = bid_volume + ask_volume
    min_depth = 10.0

    if total_depth < min_depth or bid_volume == 0 or ask_volume == 0:
        return {'manipulation': False, 'side': None, 'strength': 0.0}

    imbalance_ratio = bid_volume / ask_volume
    max_ratio = 5.0

    if imbalance_ratio > max_ratio:
        imbalance_ratio = max_ratio

    if imbalance_ratio > 3.0:
        logger.info(f"🐋 WHALE BUY WALL: {imbalance_ratio:.1f}x")
        return {'manipulation': True, 'side': 'buy', 'strength': imbalance_ratio}

    if imbalance_ratio < 1.0 / 3.0:
        strength = min(max_ratio, 1.0 / imbalance_ratio)
        logger.info(f"🐋 WHALE SELL WALL: {strength:.1f}x")
        return {'manipulation': True, 'side': 'sell', 'strength': strength}

    return {'manipulation': False, 'side': None, 'strength': 0.0}


def calculate_fear_greed_index(df):
    rsi = df["RSI"].iloc[-1]
    volatility = df["ATR"].iloc[-1] / df["close"].iloc[-1] * 100.0
    rsi_score = (rsi / 100.0) * 50.0
    vol_score = max(0.0, 50.0 - (volatility * 5.0))
    fear_greed = rsi_score + vol_score
    return min(100.0, max(0.0, fear_greed))


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
# ✅ NOUVEAU: SIGNAL SCORING SYSTEM
# ---------------------------------------------------------------------

def calculate_signal_score(confirmations, psychology):
    """
    Système de scoring pour qualifier les setups
    """
    score = 0.0
    
    # Confirmations PREMIUM (valent 2 points)
    premium_signals = [
        "bear_trap_counter",
        "bull_trap_counter",
        "rsi_extreme_oversold",
        "rsi_extreme_overbought"
    ]
    
    # Confirmations SOLIDES (valent 1 point)
    solid_signals = [
        "uptrend_confirmed",
        "downtrend_confirmed",
        "strategic_support_long",
        "strategic_support_short"
    ]
    
    # Confirmations FAIBLES (valent 0.5 points)
    weak_signals = [
        "fear_bias_long",
        "greed_bias_short",
        "positive_order_flow",
        "negative_order_flow",
        "bullish_ema_trend",
        "bearish_ema_trend",
        "below_vwap_value_zone",
        "above_vwap_premium_zone"
    ]
    
    for conf in confirmations:
        if conf in premium_signals:
            score += 2.0
        elif conf in solid_signals:
            score += 1.0
        elif conf in weak_signals:
            score += 0.5
    
    # BONUS: Manipulation + trap = GOLD setup
    if psychology.manipulation_detected and any(x in confirmations for x in premium_signals):
        score += 1.5
        logger.info("💎 GOLD SETUP BONUS: +1.5 points")
    
    return score


# ---------------------------------------------------------------------
# ENTRY LOGIC avec SCORING
# ---------------------------------------------------------------------

def yuichi_entry_signal(multi_tf_data, psychology, market_condition):
    if "tactical" not in multi_tf_data or multi_tf_data["tactical"].empty:
        return None, 0.0, None, "No data"

    df = multi_tf_data["tactical"]
    strategic = multi_tf_data.get("strategic")
    oversight = multi_tf_data.get("oversight")

    if len(df) < 50:
        return None, 0.0, None, "Insufficient data"

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

    # Psychology bias
    if psychology.sentiment == MarketSentiment.FEAR and psychology.fear_greed_index < 40:
        confirmations.append("fear_bias_long")
        signal_type = "buy"
    elif psychology.sentiment == MarketSentiment.GREED and psychology.fear_greed_index > 60:
        confirmations.append("greed_bias_short")
        signal_type = "sell"

    # Traps (contrarian)
    if psychology.manipulation_detected:
        if psychology.trap_probability > 0.02 and price < bb_lower:
            confirmations.append("bear_trap_counter")
            signal_type = "buy"
            performance.manipulation_counters += 1
        elif psychology.trap_probability > 0.02 and price > bb_upper:
            confirmations.append("bull_trap_counter")
            signal_type = "sell"
            performance.manipulation_counters += 1

    # RSI extremes
    if rsi < config.rsi_extreme_oversold:
        confirmations.append("rsi_extreme_oversold")
        if signal_type != "sell":
            signal_type = "buy"
    elif rsi > config.rsi_extreme_overbought:
        confirmations.append("rsi_extreme_overbought")
        if signal_type != "buy":
            signal_type = "sell"

    # VWAP
    vwap_deviation = (price - vwap) / vwap
    if vwap_deviation < -0.01:
        confirmations.append("below_vwap_value_zone")
        if signal_type is None:
            signal_type = "buy"
    elif vwap_deviation > 0.01:
        confirmations.append("above_vwap_premium_zone")
        if signal_type is None:
            signal_type = "sell"

    # EMA trend
    if ema9 > ema21 and price > ema9:
        confirmations.append("bullish_ema_trend")
        if signal_type == "sell":
            if "bull_trap_counter" not in confirmations and "greed_bias_short" not in confirmations:
                signal_type = "buy"
        elif signal_type is None:
            signal_type = "buy"
    elif ema9 < ema21 and price < ema9:
        confirmations.append("bearish_ema_trend")
        if signal_type == "buy":
            if "bear_trap_counter" not in confirmations and "fear_bias_long" not in confirmations:
                signal_type = "sell"
        elif signal_type is None:
            signal_type = "sell"

    # Order flow
    if order_flow > 0:
        confirmations.append("positive_order_flow")
    elif order_flow < 0:
        confirmations.append("negative_order_flow")

    # Multi-timeframe
    if strategic is not None and not strategic.empty:
        strategic_rsi = strategic["RSI"].iloc[-1]
        if signal_type == "buy" and strategic_rsi < 60:
            confirmations.append("strategic_support_long")
        elif signal_type == "sell" and strategic_rsi > 40:
            confirmations.append("strategic_support_short")

    if oversight is not None and not oversight.empty and "SMA_200" in oversight.columns:
        try:
            trend_up = oversight["SMA_50"].iloc[-1] > oversight["SMA_200"].iloc[-1]
            if trend_up and signal_type == "buy":
                confirmations.append("uptrend_confirmed")
            elif (not trend_up) and signal_type == "sell":
                confirmations.append("downtrend_confirmed")
            else:
                confirmations.append("countertrend_play")
        except Exception:
            pass

    # Dernier recours
    if signal_type is None:
        if rsi < 50 and "bullish_ema_trend" in confirmations:
            signal_type = "buy"
        elif rsi > 50 and "bearish_ema_trend" in confirmations:
            signal_type = "sell"

    # ✅ CALCUL DU SCORE
    score = calculate_signal_score(confirmations, psychology)

    # ✅ Validation avec score
    if not signal_type or score < config.min_setup_score:
        return None, score, confirmations, f"Score {score:.1f} < {config.min_setup_score}"
    
    # ✅ Respect market condition
    if market_condition == MarketCondition.TRENDING_BULL and signal_type == "sell":
        logger.warning(f"⚠️ SKIP: Trending BULL but signal SHORT (score {score:.1f})")
        return None, score, confirmations, "Against trend"
    
    if market_condition == MarketCondition.TRENDING_BEAR and signal_type == "buy":
        logger.warning(f"⚠️ SKIP: Trending BEAR but signal LONG (score {score:.1f})")
        return None, score, confirmations, "Against trend"

    logger.info(f"✅ SIGNAL: {signal_type.upper()} | Score: {score:.1f} | Confs: {confirmations}")
    performance.psychological_edges += 1
    
    return signal_type, score, confirmations, psychology


# ---------------------------------------------------------------------
# ✅ NOUVEAU: TP ADAPTATIF
# ---------------------------------------------------------------------

def calculate_adaptive_tp_mult(confirmations, psychology, score):
    """
    TP adapté selon la qualité du setup
    """
    tp_mult = 1.5  # Base
    
    # Setup GOLD (manipulation + trap)
    if psychology.manipulation_detected:
        premium_traps = ["bear_trap_counter", "bull_trap_counter"]
        if any(x in confirmations for x in premium_traps):
            tp_mult = 3.5
            logger.info("💎 GOLD SETUP: TP = 3.5 ATR")
            return tp_mult
    
    # Setup PREMIUM (score élevé)
    if score >= 6.0:
        tp_mult = 3.0
        logger.info("⭐ PREMIUM SETUP: TP = 3.0 ATR")
    elif score >= 5.0:
        tp_mult = 2.5
    
    # Multi-timeframe aligné
    if "uptrend_confirmed" in confirmations or "downtrend_confirmed" in confirmations:
        tp_mult += 0.5
    
    return tp_mult


# ---------------------------------------------------------------------
# ✅ NOUVEAU: SL STRUCTUREL
# ---------------------------------------------------------------------

def calculate_smart_sl_tp(entry_price, df, signal_type, psychology, score, confirmations):
    """
    SL basé sur structure + TP adaptatif
    """
    atr = df["ATR"].iloc[-1]
    
    # Support/Resistance
    recent_lows = df["low"].iloc[-30:].nsmallest(3).mean()
    recent_highs = df["high"].iloc[-30:].nlargest(3).mean()
    
    if signal_type == "buy":
        # SL sous support
        structural_sl = recent_lows * 0.997
        atr_sl = entry_price - (atr * 1.0)
        stop_loss = max(structural_sl, atr_sl)
        
        # TP adaptatif
        tp_mult = calculate_adaptive_tp_mult(confirmations, psychology, score)
        take_profit = entry_price + (atr * tp_mult)
    
    else:  # sell
        structural_sl = recent_highs * 1.003
        atr_sl = entry_price + (atr * 1.0)
        stop_loss = min(structural_sl, atr_sl)
        
        tp_mult = calculate_adaptive_tp_mult(confirmations, psychology, score)
        take_profit = entry_price - (atr * tp_mult)
    
    return stop_loss, take_profit


# ---------------------------------------------------------------------
# POSITION SIZING
# ---------------------------------------------------------------------

def calculate_position_size(psychology, score):
    base_index = min(config.current_step, len(config.martingale_steps) - 1)
    base_size = config.martingale_steps[base_index]

    battle_index = min(config.battle_step, len(config.battle_steps) - 1)
    battle_mult = config.battle_steps[battle_index]

    size = base_size * battle_mult

    # Ajustement volatilité
    if psychology.volatility_regime == "extreme":
        size *= 0.5
    elif psychology.volatility_regime == "high":
        size *= 0.7
    elif psychology.volatility_regime == "low":
        size *= 1.1

    # ✅ Bonus pour excellent setup
    if score >= 6.0:
        size *= 1.2
        logger.info(f"💎 Score {score:.1f}: Size +20%")

    return size


# ---------------------------------------------------------------------
# ✅ REALISTIC PNL
# ---------------------------------------------------------------------

def calculate_realistic_pnl(entry, exit, size, trade_type):
    """
    PnL avec frais et slippage
    """
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


# ---------------------------------------------------------------------
# GUI HELPERS
# ---------------------------------------------------------------------

def log_result(msg, log_box):
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


def update_winnings_box(entry_var):
    try:
        entry_var.set(f"${config.cumulative_winnings:.2f}")
    except Exception:
        pass


def update_stats_panel(box):
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
    if not price_history:
        return
    
    ax.clear()
    ax.plot(price_history, label="Price", color="blue")
    
    if trade_state.entry_price:
        ax.axhline(trade_state.entry_price, linestyle="--", label="Entry", color="yellow")
    if trade_state.stop_loss:
        ax.axhline(trade_state.stop_loss, linestyle="--", label="SL", color="red")
    if trade_state.take_profit:
        ax.axhline(trade_state.take_profit, linestyle="--", label="TP", color="green")
    
    ax.legend(loc="best")
    canvas.draw_idle()


# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------

def execute_yuichi_strategy(log_box, ax, canvas, winnings_var, stats_box, psych_box):
    logger.info("=" * 70)
    logger.info("💎 ENHANCED METHOD v13.5 💎")
    logger.info("Scoring + Market Conditions + Adaptive TP + No Hedge")
    logger.info("=" * 70)
    
    while config.running:
        try:
            # Limits
            if (config.trades_executed_today >= config.max_trades_per_day or
                config.daily_loss >= config.capital * config.max_daily_loss_pct or
                config.current_step >= config.max_martingale_steps):
                log_result("Daily limits reached.", log_box)
                time.sleep(60)
                continue
            
            # ✅ Cooldown
            if config.last_trade_time:
                elapsed = (datetime.now() - config.last_trade_time).total_seconds() / 60.0
                if elapsed < config.min_time_between_trades:
                    time.sleep(10)
                    continue
            
            # Fetch data
            multi_tf_data = fetch_multi_timeframe_data(config.symbols[0])
            df = multi_tf_data.get("tactical")
            if not df or len(df) < 50:
                time.sleep(5)
                continue
            
            # Indicators
            for name in multi_tf_data:
                multi_tf_data[name] = calculate_all_indicators(multi_tf_data[name])
            
            df = multi_tf_data["tactical"]
            current_price = df["close"].iloc[-1]
            
            price_history.append(current_price)
            if len(price_history) > 500:
                price_history.pop(0)
            
            # Order book
            order_book = exchange.fetch_order_book(config.symbols[0], limit=20)
            
            # Psychology
            psychology = calculate_market_psychology(df, order_book)
            
            # ✅ Market condition
            market_condition = detect_market_condition(
                multi_tf_data.get("oversight"),
                multi_tf_data.get("strategic"),
                multi_tf_data.get("tactical")
            )
            
            update_psychology_panel(psych_box, psychology, market_condition)
            
            # ✅ Skip bad markets
            if market_condition in [MarketCondition.CHOPPY, MarketCondition.VOLATILE_EXTREME]:
                time.sleep(10)
                continue
            
            # ==================== NO POSITION ====================
            if not trade_state.active:
                signal_type, score, confirmations, reason = yuichi_entry_signal(
                    multi_tf_data, psychology, market_condition
                )
                
                if not signal_type:
                    time.sleep(5)
                    continue
                
                # ✅ Check min profit threshold
                atr = df["ATR"].iloc[-1]
                tp_mult = calculate_adaptive_tp_mult(confirmations, psychology, score)
                expected_profit_pct = (atr * tp_mult) / current_price
                
                if expected_profit_pct < config.min_profit_threshold:
                    logger.info(f"⚠️ Expected profit {expected_profit_pct:.2%} < {config.min_profit_threshold:.2%}, skip")
                    time.sleep(5)
                    continue
                
                # Position size
                position_size = calculate_position_size(psychology, score)
                
                # Enter
                trade_state.active = True
                trade_state.type = signal_type
                trade_state.entry_price = current_price
                trade_state.entry_time = datetime.now()
                trade_state.position_size = position_size
                trade_state.score = score
                trade_state.confirmations = confirmations
                trade_state.psychology_snapshot = psychology
                
                sl, tp = calculate_smart_sl_tp(
                    current_price, df, signal_type, psychology, score, confirmations
                )
                trade_state.stop_loss = sl
                trade_state.take_profit = tp
                
                rr_ratio = abs((tp - current_price) / (current_price - sl)) if signal_type == "buy" else abs((current_price - tp) / (sl - current_price))
                
                log_msg = (
                    f"\n{'='*70}\n"
                    f"💎 ENTER: {signal_type.upper()}\n"
                    f"{'='*70}\n"
                    f"Score: {score:.1f} | Market: {market_condition.value}\n"
                    f"Entry: ${current_price:.2f}\n"
                    f"Position: ${position_size:.2f}\n"
                    f"SL: ${sl:.2f} | TP: ${tp:.2f}\n"
                    f"R:R: {rr_ratio:.2f}:1\n"
                    f"Confirmations: {', '.join(confirmations[:5])}\n"
                    f"{'='*70}"
                )
                log_result(log_msg, log_box)
            
            # ==================== ACTIVE POSITION ====================
            else:
                # TP
                if trade_state.type == "buy" and current_price >= trade_state.take_profit:
                    close_trade("TAKE PROFIT", trade_state.take_profit, log_box, winnings_var, stats_box)
                elif trade_state.type == "sell" and current_price <= trade_state.take_profit:
                    close_trade("TAKE PROFIT", trade_state.take_profit, log_box, winnings_var, stats_box)
                
                # SL
                elif trade_state.type == "buy" and current_price <= trade_state.stop_loss:
                    close_trade("STOP LOSS", trade_state.stop_loss, log_box, winnings_var, stats_box)
                elif trade_state.type == "sell" and current_price >= trade_state.stop_loss:
                    close_trade("STOP LOSS", trade_state.stop_loss, log_box, winnings_var, stats_box)
            
            update_stats_panel(stats_box)
            update_chart(ax, canvas)
            time.sleep(3)
        
        except Exception as e:
            logger.error(f"Loop error: {e}", exc_info=True)
            time.sleep(5)


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------

def create_gui():
    root = tk.Tk()
    root.title("💎 Enhanced Method v13.5 💎")
    root.geometry("1200x900")
    
    tk.Label(root, text="Trade Log", font=("Arial", 12, "bold")).grid(
        row=0, column=0, sticky="w", padx=10, pady=(10,0)
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
    
    tk.Button(btn_frame, text="Stop", command=stop, bg="#c0392b", fg="white", font=("Arial", 12, "bold"), width=12).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Quit", command=root.destroy, bg="#7f8c8d", fg="white", font=("Arial", 12, "bold"), width=12).pack(side=tk.LEFT, padx=10)
    
    update_winnings_box(winnings_var)
    
    t = threading.Thread(target=execute_yuichi_strategy, args=(log_box, ax, canvas, winnings_var, stats_box, psych_box), daemon=True)
    t.start()
    
    root.mainloop()


if __name__ == "__main__":
    logger.info("💎 Initializing Enhanced Method v13.5 💎")
    logger.info(f"Capital: ${config.capital}")
    logger.info("Improvements: Scoring, Market Conditions, Adaptive TP, No Hedge, Fees")
    create_gui()
