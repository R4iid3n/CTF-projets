"""
Trading Bots Dashboard - Flask Web Application
Runs v13, v13.5, and v14 bots in parallel with live monitoring
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import json
import time
from datetime import datetime
from collections import deque
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yuichi-trading-secret-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------------------------------------------------------------------
# BOT STATE MANAGEMENT
# ---------------------------------------------------------------------

class BotState:
    def __init__(self, bot_id, name, description, philosophy):
        self.bot_id = bot_id
        self.name = name
        self.description = description
        self.philosophy = philosophy
        
        # Performance metrics
        self.capital = 1000.0
        self.cumulative_winnings = 0.0
        self.trades_executed = 0
        self.wins = 0
        self.losses = 0
        self.win_rate = 0.0
        
        # Current trade
        self.active_trade = None
        self.current_price = 0.0
        
        # History
        self.trade_history = deque(maxlen=100)
        self.equity_curve = deque(maxlen=500)
        self.log_messages = deque(maxlen=50)
        
        # Status
        self.running = False
        self.last_update = None
        
    def to_dict(self):
        return {
            'bot_id': self.bot_id,
            'name': self.name,
            'description': self.description,
            'philosophy': self.philosophy,
            'capital': round(self.capital, 2),
            'cumulative_winnings': round(self.cumulative_winnings, 2),
            'trades_executed': self.trades_executed,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': round(self.win_rate, 1),
            'active_trade': self.active_trade,
            'current_price': round(self.current_price, 2),
            'running': self.running,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'trade_history': list(self.trade_history),
            'equity_curve': list(self.equity_curve),
            'log_messages': list(self.log_messages)
        }


# Initialize bot states
bots = {
    'v13': BotState(
        bot_id='v13',
        name='v13 FIXED',
        description='Multi-confirmation with hedge betting',
        philosophy=[
            'Trades 80-100 times per month',
            'Uses 2-3 confirmation layers',
            'Hedge betting at 1% adverse move',
            'Conservative but consistent',
            'Expected: 60% win rate, +$1-3/month'
        ]
    ),
    'v13_5': BotState(
        bot_id='v13_5',
        name='v13.5 ENHANCED',
        description='Optimized multi-confirmation with scoring',
        philosophy=[
            'Trades 40-50 times per month',
            'Scoring system (4.0+ required)',
            'Market condition filtering',
            'Adaptive take profit (up to 3.5 ATR)',
            'No hedge - keeps 100% profit',
            'Expected: 67% win rate, +$9-10/month'
        ]
    ),
    'v14': BotState(
        bot_id='v14',
        name='v14 TRUE METHOD',
        description='Yuichi psychological warfare - traps only',
        philosophy=[
            'Trades 10-15 times per month',
            'Only enters on confirmed traps (75%+ confidence)',
            'Game States: OBSERVING → GAME OVER',
            '7 trap types detected',
            'All-in on perfect setups ($150+)',
            'Expected: 75% win rate, +$9-12/month'
        ]
    )
}

# Global comparison stats
comparison_stats = {
    'total_capital': 3000.0,
    'total_winnings': 0.0,
    'best_performer': None,
    'total_trades': 0
}

# ---------------------------------------------------------------------
# SIMULATED BOT RUNNERS (Replace with real bot logic)
# ---------------------------------------------------------------------

def simulate_bot_v13(bot_state):
    """Simulates v13 bot behavior"""
    while bot_state.running:
        try:
            # Simulate price movement
            bot_state.current_price = 90000 + (time.time() % 1000) * 10
            
            # Simulate occasional trades (higher frequency)
            if len(bot_state.trade_history) < bot_state.trades_executed:
                # Trade already logged
                pass
            elif bot_state.trades_executed < 100 and time.time() % 30 < 1:
                # Simulate trade every ~30 seconds
                is_win = (time.time() % 10) > 4  # 60% win rate
                profit = 0.35 if is_win else -0.21
                
                bot_state.cumulative_winnings += profit
                bot_state.capital += profit
                bot_state.trades_executed += 1
                
                if is_win:
                    bot_state.wins += 1
                else:
                    bot_state.losses += 1
                
                bot_state.win_rate = (bot_state.wins / bot_state.trades_executed * 100) if bot_state.trades_executed > 0 else 0
                
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'buy' if time.time() % 2 < 1 else 'sell',
                    'entry': round(bot_state.current_price, 2),
                    'exit': round(bot_state.current_price + (profit * 100), 2),
                    'profit': round(profit, 2),
                    'result': 'WIN' if is_win else 'LOSS'
                }
                bot_state.trade_history.append(trade)
                bot_state.log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Trade closed: {trade['result']} ${profit:.2f}")
            
            # Update equity curve
            bot_state.equity_curve.append({
                'timestamp': datetime.now().isoformat(),
                'value': round(bot_state.capital, 2)
            })
            
            bot_state.last_update = datetime.now()
            
            # Emit update via WebSocket
            socketio.emit('bot_update', {'bot_id': 'v13', 'data': bot_state.to_dict()})
            
            time.sleep(2)
            
        except Exception as e:
            bot_state.log_messages.append(f"[ERROR] {str(e)}")
            time.sleep(5)


def simulate_bot_v13_5(bot_state):
    """Simulates v13.5 bot behavior"""
    while bot_state.running:
        try:
            bot_state.current_price = 90000 + (time.time() % 1000) * 10
            
            # Medium frequency trades
            if bot_state.trades_executed < 50 and time.time() % 50 < 1:
                is_win = (time.time() % 10) > 3.3  # 67% win rate
                profit = 0.80 if is_win else -0.40
                
                bot_state.cumulative_winnings += profit
                bot_state.capital += profit
                bot_state.trades_executed += 1
                
                if is_win:
                    bot_state.wins += 1
                else:
                    bot_state.losses += 1
                
                bot_state.win_rate = (bot_state.wins / bot_state.trades_executed * 100) if bot_state.trades_executed > 0 else 0
                
                score = round(4.0 + (time.time() % 3), 1)
                
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'buy' if time.time() % 2 < 1 else 'sell',
                    'entry': round(bot_state.current_price, 2),
                    'exit': round(bot_state.current_price + (profit * 100), 2),
                    'profit': round(profit, 2),
                    'result': 'WIN' if is_win else 'LOSS',
                    'score': score
                }
                bot_state.trade_history.append(trade)
                bot_state.log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Score {score} - {trade['result']} ${profit:.2f}")
            
            bot_state.equity_curve.append({
                'timestamp': datetime.now().isoformat(),
                'value': round(bot_state.capital, 2)
            })
            
            bot_state.last_update = datetime.now()
            socketio.emit('bot_update', {'bot_id': 'v13_5', 'data': bot_state.to_dict()})
            
            time.sleep(2)
            
        except Exception as e:
            bot_state.log_messages.append(f"[ERROR] {str(e)}")
            time.sleep(5)


def simulate_bot_v14(bot_state):
    """Simulates v14 bot behavior"""
    while bot_state.running:
        try:
            bot_state.current_price = 90000 + (time.time() % 1000) * 10
            
            # Low frequency, high quality trades
            if bot_state.trades_executed < 15 and time.time() % 120 < 1:
                is_win = (time.time() % 10) > 2.5  # 75% win rate
                profit = 1.56 if is_win else -0.60
                
                bot_state.cumulative_winnings += profit
                bot_state.capital += profit
                bot_state.trades_executed += 1
                
                if is_win:
                    bot_state.wins += 1
                else:
                    bot_state.losses += 1
                
                bot_state.win_rate = (bot_state.wins / bot_state.trades_executed * 100) if bot_state.trades_executed > 0 else 0
                
                game_state = ['OBSERVING', 'TRAP_DETECTED', 'SETUP_FORMING', 'GAME_OVER'][int(time.time()) % 4]
                
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'buy' if time.time() % 2 < 1 else 'sell',
                    'entry': round(bot_state.current_price, 2),
                    'exit': round(bot_state.current_price + (profit * 100), 2),
                    'profit': round(profit, 2),
                    'result': 'WIN' if is_win else 'LOSS',
                    'game_state': game_state
                }
                bot_state.trade_history.append(trade)
                bot_state.log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {game_state} - {trade['result']} ${profit:.2f}")
            
            bot_state.equity_curve.append({
                'timestamp': datetime.now().isoformat(),
                'value': round(bot_state.capital, 2)
            })
            
            bot_state.last_update = datetime.now()
            socketio.emit('bot_update', {'bot_id': 'v14', 'data': bot_state.to_dict()})
            
            time.sleep(2)
            
        except Exception as e:
            bot_state.log_messages.append(f"[ERROR] {str(e)}")
            time.sleep(5)


# Bot runner mapping
bot_runners = {
    'v13': simulate_bot_v13,
    'v13_5': simulate_bot_v13_5,
    'v14': simulate_bot_v14
}

# ---------------------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/bots')
def get_bots():
    """Get all bot states"""
    return jsonify({
        'bots': {bot_id: bot.to_dict() for bot_id, bot in bots.items()},
        'comparison': update_comparison_stats()
    })

@app.route('/api/bot/<bot_id>')
def get_bot(bot_id):
    """Get specific bot state"""
    if bot_id not in bots:
        return jsonify({'error': 'Bot not found'}), 404
    return jsonify(bots[bot_id].to_dict())

@app.route('/api/bot/<bot_id>/start', methods=['POST'])
def start_bot(bot_id):
    """Start a bot"""
    if bot_id not in bots:
        return jsonify({'error': 'Bot not found'}), 404
    
    bot = bots[bot_id]
    if not bot.running:
        bot.running = True
        thread = threading.Thread(target=bot_runners[bot_id], args=(bot,), daemon=True)
        thread.start()
        return jsonify({'status': 'started', 'bot_id': bot_id})
    
    return jsonify({'status': 'already_running', 'bot_id': bot_id})

@app.route('/api/bot/<bot_id>/stop', methods=['POST'])
def stop_bot(bot_id):
    """Stop a bot"""
    if bot_id not in bots:
        return jsonify({'error': 'Bot not found'}), 404
    
    bot = bots[bot_id]
    bot.running = False
    return jsonify({'status': 'stopped', 'bot_id': bot_id})

@app.route('/api/bots/start_all', methods=['POST'])
def start_all_bots():
    """Start all bots"""
    for bot_id, bot in bots.items():
        if not bot.running:
            bot.running = True
            thread = threading.Thread(target=bot_runners[bot_id], args=(bot,), daemon=True)
            thread.start()
    return jsonify({'status': 'all_started'})

@app.route('/api/bots/stop_all', methods=['POST'])
def stop_all_bots():
    """Stop all bots"""
    for bot in bots.values():
        bot.running = False
    return jsonify({'status': 'all_stopped'})

@app.route('/api/bot/<bot_id>/code')
def get_bot_code(bot_id):
    """Get bot source code"""
    code_files = {
        'v13': 'yuichi_bot_v13_fixed.py',
        'v13_5': 'yuichi_bot_v13_5_enhanced.py',
        'v14': 'yuichi_bot_v14_true_method.py'
    }
    
    if bot_id not in code_files:
        return jsonify({'error': 'Bot not found'}), 404
    
    # Try to read from outputs directory
    file_path = f'/mnt/user-data/outputs/{code_files[bot_id]}'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return jsonify({'code': code, 'filename': code_files[bot_id]})
    except FileNotFoundError:
        return jsonify({'error': 'Code file not found', 'path': file_path}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def update_comparison_stats():
    """Update global comparison statistics"""
    comparison_stats['total_capital'] = sum(bot.capital for bot in bots.values())
    comparison_stats['total_winnings'] = sum(bot.cumulative_winnings for bot in bots.values())
    comparison_stats['total_trades'] = sum(bot.trades_executed for bot in bots.values())
    
    # Find best performer
    if bots:
        best = max(bots.values(), key=lambda b: b.cumulative_winnings)
        comparison_stats['best_performer'] = {
            'bot_id': best.bot_id,
            'name': best.name,
            'winnings': round(best.cumulative_winnings, 2)
        }
    
    return comparison_stats

# ---------------------------------------------------------------------
# WEBSOCKET EVENTS
# ---------------------------------------------------------------------

@socketio.on('connect')
def handle_connect():
    emit('connected', {'message': 'Connected to Trading Bots Dashboard'})

@socketio.on('request_update')
def handle_update_request():
    """Send current state of all bots"""
    emit('full_update', {
        'bots': {bot_id: bot.to_dict() for bot_id, bot in bots.items()},
        'comparison': update_comparison_stats()
    })

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Trading Bots Dashboard Starting")
    print("=" * 70)
    print("Access at: http://localhost:5000")
    print("=" * 70)
    
    # Start Flask-SocketIO server
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
