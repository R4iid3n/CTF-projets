"""
VPS Trading Bots Dashboard - Real-time monitoring via SSH
Connects to your Debian VPS and displays live bot status
"""

from flask import Flask, render_template, jsonify
import paramiko
import json
import time
from datetime import datetime
from threading import Thread, Lock
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# VPS CONFIG
# ---------------------------------------------------------------------

VPS_CONFIG = {
    'host': '46.224.11.203',  # Change to your VPS IP
    'port': 22,
    'username': 'aml',
    'key_path': None,  # Path to SSH key or None
    'password': "9XrjobfAZLCPdPqno2h2CCHkY",  # Password or None if using key
}

# Bot configurations - match your VPS paths
BOTS_CONFIG = {
    'v13': {
        'name': 'v13 FIXED',
        'status_path': '/home/aml/trading-bot/cli/status/yuichi_v13_cli.json',
        'log_path': '/home/aml/trading-bot/cli/yuichi_bot_v13_cli.log',
        'color': '#ff4757'
    },
    'v13_5': {
        'name': 'v13.5 ENHANCED',
        'status_path': '/home/aml/trading-bot/cli/status/yuichi_v13_5_cli.json',
        'log_path': '/home/aml/trading-bot/cli/yuichi_bot_v13_5_cli.log',
        'color': '#00d4ff'
    },
    'v14': {
        'name': 'v14 TRUE METHOD',
        'status_path': '/home/aml/trading-bot/cli/status/yuichi_v14_cli.json',
        'log_path': '/home/aml/trading-bot/cli/yuichi_bot_v14_cli.log',
        'color': '#00ff88'
    },
    'v14_aggressive': {
        'name': 'v14 AGGRESSIVE',
        'status_path': '/home/aml/trading-bot/cli/status/yuichi_v14_aggressive_cli.json',
        'log_path': '/home/aml/trading-bot/cli/yuichi_bot_v14_aggressive_cli.log',
        'color': '#ffa502'
    }
}

# ---------------------------------------------------------------------
# SSH CLIENT
# ---------------------------------------------------------------------

class VPSClient:
    def __init__(self):
        self.client = None
        self.connected = False
        self.lock = Lock()
        
    def connect(self):
        """Connect to VPS via SSH"""
        with self.lock:
            if self.connected:
                return True
                
            try:
                self.client = paramiko.SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                if VPS_CONFIG['key_path']:
                    self.client.connect(
                        VPS_CONFIG['host'],
                        port=VPS_CONFIG['port'],
                        username=VPS_CONFIG['username'],
                        key_filename=VPS_CONFIG['key_path'],
                        timeout=10
                    )
                else:
                    self.client.connect(
                        VPS_CONFIG['host'],
                        port=VPS_CONFIG['port'],
                        username=VPS_CONFIG['username'],
                        password=VPS_CONFIG['password'],
                        timeout=10
                    )
                
                self.connected = True
                logger.info("✅ Connected to VPS")
                return True
                
            except Exception as e:
                logger.error(f"❌ VPS connection failed: {e}")
                self.connected = False
                return False
    
    def read_json_file(self, path):
        """Read and parse JSON file from VPS"""
        try:
            if not self.connect():
                return None
                
            sftp = self.client.open_sftp()
            with sftp.open(path, 'r') as f:
                content = f.read().decode('utf-8')
                return json.loads(content)
                
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {path}")
            return None
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            self.connected = False
            return None
    
    def tail_log(self, path, lines=50):
        """Get last N lines of log file"""
        try:
            if not self.connect():
                return ""
                
            cmd = f"tail -n {lines} {path} 2>/dev/null || echo 'Log not found'"
            stdin, stdout, stderr = self.client.exec_command(cmd)
            return stdout.read().decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"Error tailing {path}: {e}")
            return f"ERROR: {str(e)}"
    
    def disconnect(self):
        """Disconnect from VPS"""
        with self.lock:
            if self.client:
                self.client.close()
                self.connected = False
                logger.info("Disconnected from VPS")


# Global VPS client
vps = VPSClient()

# Bot data cache
bots_data = {}
data_lock = Lock()

# ---------------------------------------------------------------------
# DATA UPDATER (Background Thread)
# ---------------------------------------------------------------------

def update_bot_data():
    """Background thread to update bot data every 3 seconds"""
    while True:
        try:
            for bot_id, config in BOTS_CONFIG.items():
                status = vps.read_json_file(config['status_path'])
                
                if status:
                    # Enrich with config
                    status['bot_id'] = bot_id
                    status['display_name'] = config['name']
                    status['color'] = config['color']
                    
                    # Handle different JSON formats
                    # v14 bots don't have "running" key, check timestamp instead
                    if 'running' not in status:
                        # Check if timestamp is recent (within last 30 seconds)
                        if 'timestamp' in status or 'last_update' in status:
                            status['running'] = True  # If file exists and recent, bot is running
                        else:
                            status['running'] = False
                    
                    # Normalize field names for v14 bots
                    if 'active_trade' in status and 'position_active' not in status:
                        status['position_active'] = status['active_trade']
                    
                    if 'trade_type' in status and 'side' not in status:
                        status['side'] = status['trade_type']
                    
                    if 'position_size' in status and 'position_size_main' not in status:
                        status['position_size_main'] = status.get('position_size', 0.0)
                    
                    if 'martingale_level' in status and 'martingale_step' not in status:
                        status['martingale_step'] = status['martingale_level']
                    
                    # Get recent logs
                    status['recent_logs'] = vps.tail_log(config['log_path'], lines=20)
                    
                    with data_lock:
                        bots_data[bot_id] = status
                else:
                    # Bot not running or file not found
                    with data_lock:
                        if bot_id in bots_data:
                            bots_data[bot_id]['running'] = False
                            bots_data[bot_id]['last_error'] = 'Status file not found'
            
            time.sleep(3)  # Update every 3 seconds
            
        except Exception as e:
            logger.error(f"Update error: {e}")
            time.sleep(5)


# Start background updater
updater_thread = Thread(target=update_bot_data, daemon=True)
updater_thread.start()

# ---------------------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('vps_dashboard.html')

@app.route('/api/bots')
def get_all_bots():
    """Get all bot statuses"""
    with data_lock:
        bots_list = list(bots_data.values())
    
    # Calculate totals
    total_capital = sum(b.get('capital', 0) for b in bots_list)
    total_winnings = sum(b.get('cumulative_winnings', 0) for b in bots_list)
    total_trades = sum(b.get('trades_executed_today', 0) for b in bots_list)
    
    # Find best performer
    best = max(bots_list, key=lambda b: b.get('cumulative_winnings', -999999)) if bots_list else None
    
    return jsonify({
        'bots': bots_list,
        'summary': {
            'total_capital': round(total_capital, 2),
            'total_winnings': round(total_winnings, 2),
            'total_trades': total_trades,
            'best_performer': {
                'name': best.get('display_name', 'N/A'),
                'winnings': round(best.get('cumulative_winnings', 0), 2)
            } if best else None,
            'vps_connected': vps.connected
        }
    })

@app.route('/api/bot/<bot_id>')
def get_bot(bot_id):
    """Get specific bot status"""
    with data_lock:
        bot = bots_data.get(bot_id)
    
    if not bot:
        return jsonify({'error': 'Bot not found'}), 404
    
    return jsonify(bot)

@app.route('/api/vps/status')
def vps_status():
    """Check VPS connection status"""
    return jsonify({
        'connected': vps.connected,
        'host': VPS_CONFIG['host'],
        'bots_monitored': len(BOTS_CONFIG)
    })

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 VPS Trading Bots Dashboard Starting")
    print("=" * 70)
    print(f"VPS: {VPS_CONFIG['host']}")
    print(f"Monitoring {len(BOTS_CONFIG)} bots")
    print("=" * 70)
    print("Access at: http://localhost:5000")
    print("=" * 70)
    
    app.run(debug=False, host='0.0.0.0', port=5000)










