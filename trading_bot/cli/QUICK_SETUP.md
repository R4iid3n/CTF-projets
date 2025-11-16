# 🚀 Quick Setup - VPS Trading Dashboard

## Installation (5 minutes)

### 1. Extract & Install

```bash
# Extract archive
tar -xzf vps_trading_dashboard.tar.gz
cd vps_dashboard

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure VPS Connection

Open `app.py` and edit lines 18-24:

```python
VPS_CONFIG = {
    'host': '46.XXX.XXX.XXX',     # ← Change to your VPS IP
    'port': 22,
    'username': 'aml',             # ← Your VPS username
    'key_path': None,              # ← Path to SSH key OR None
    'password': None,              # ← Password OR None if using key
}
```

### 3. Configure Bot Paths (Lines 28-50)

Update to match YOUR VPS paths:

```python
BOTS_CONFIG = {
    'v13': {
        'name': 'v13 FIXED',
        'status_path': '/home/aml/bot-status/yuichi_v13.json',  # ← Change paths
        'log_path': '/home/aml/trading-bot/yuichi_bot_v13_cli.log',
        'color': '#ff4757'
    },
    # Add your other bots...
}
```

### 4. On Your VPS (Debian)

Make sure status directory exists:

```bash
ssh aml@46.XXX.XXX.XXX
mkdir -p /home/aml/bot-status
```

Your bots already write status files (they have `write_status()` function).

### 5. Run Dashboard

```bash
python app.py
```

Open browser: `http://localhost:5000`

---

## SSH Authentication Setup

### Option A: SSH Key (Recommended) ⭐

**Generate key (if you don't have one):**

```bash
# On your machine (Windows/Mac/Linux)
ssh-keygen -t rsa -b 4096
# Save to: /home/you/.ssh/id_rsa
```

**Copy to VPS:**

```bash
ssh-copy-id aml@46.XXX.XXX.XXX
```

**In app.py:**

```python
VPS_CONFIG = {
    'host': '46.XXX.XXX.XXX',
    'username': 'aml',
    'key_path': '/home/you/.ssh/id_rsa',  # ← Path to your key
    'password': None
}
```

### Option B: Password

**In app.py:**

```python
VPS_CONFIG = {
    'host': '46.XXX.XXX.XXX',
    'username': 'aml',
    'key_path': None,
    'password': 'your_vps_password'  # ← Your password
}
```

⚠️ **Security:** SSH key is more secure than password.

---

## What You'll See

### Dashboard Features:

**Global Stats:**
- Total Capital: Sum of all bots
- Total Winnings: Green if positive
- Total Trades Today
- Best Performer

**Bot Cards (for each bot):**
- 🟢 Running / ⚫ Stopped status
- Capital & Cumulative Winnings
- Trades & Battle Steps
- **Active Position** (if in trade):
  - Entry Price
  - Stop Loss (red)
  - Take Profit (green)
  - Position size
  - Game State badge (v14)
  - Battle P&L
- 📋 View Logs button

**Position Chart:**
- Bar chart of active positions
- Real-time updates every 3 seconds

---

## Testing Connection

**Before running dashboard, test SSH manually:**

```bash
# Test connection
ssh aml@46.XXX.XXX.XXX

# Check if status files exist
ls -la /home/aml/bot-status/

# View a status file
cat /home/aml/bot-status/yuichi_v13.json
```

Should see JSON like:
```json
{
  "bot_name": "yuichi_v13",
  "capital": 1025.50,
  "cumulative_winnings": 25.50,
  "position_active": true,
  "entry_price": 90245.50,
  ...
}
```

---

## Troubleshooting

### "Connection Failed"

1. Check VPS IP is correct
2. Check port 22 is open
3. Test: `ssh aml@46.XXX.XXX.XXX`
4. Check firewall allows SSH

### "File Not Found"

1. Bots must be running on VPS
2. Check paths in `BOTS_CONFIG`
3. On VPS: `ls /home/aml/bot-status/`

### "Dashboard Shows Nothing"

1. Bots must write status files (they do via `write_status()`)
2. Check bot is actually running: `ps aux | grep yuichi`
3. Wait 3 seconds for first update

---

## Running in Background

### Option 1: Screen (Simple)

```bash
screen -S dashboard
python app.py
# Press Ctrl+A then D to detach
```

Re-attach: `screen -r dashboard`

### Option 2: tmux

```bash
tmux new -s dashboard
python app.py
# Press Ctrl+B then D to detach
```

Re-attach: `tmux attach -t dashboard`

### Option 3: Systemd Service (Production)

See full README.md for systemd setup.

---

## Access from Other Devices

**Currently:** Dashboard runs on `localhost:5000`

**To access from other devices:**

Change in `app.py` (line 228):
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

Then access via: `http://YOUR_MACHINE_IP:5000`

⚠️ **Security:** Only do this on trusted networks.

---

## Next Steps

1. ✅ Configure VPS connection
2. ✅ Update bot paths
3. ✅ Test SSH connection
4. ✅ Run dashboard
5. ✅ Check browser at localhost:5000
6. ✅ Enjoy real-time monitoring! 🎮

**Need help?** Check README.md for detailed docs.

---

🎮 **Your bots, visualized!** 🚀
