# 🎮 VPS Trading Bots Dashboard

Monitor your trading bots running on a Debian VPS in real-time via SSH.

## Features

✅ **Real-time SSH Monitoring**
- Connects to your VPS every 3 seconds
- Reads JSON status files from your bots
- Displays live metrics and positions

✅ **Live Position Tracking**
- Entry price, Stop Loss, Take Profit
- Position size (main + hedge)
- Game state (v14 bots)
- Battle P&L

✅ **Performance Metrics**
- Capital tracking
- Cumulative winnings
- Trades executed today
- Martingale & Battle steps

✅ **Log Viewer**
- View real-time logs from each bot
- Tail last 50 lines
- Modal popup interface

✅ **Visual Charts**
- Active positions bar chart
- Color-coded by bot
- Real-time updates

## Setup

### 1. Install Dependencies

```bash
cd vps_dashboard
pip install -r requirements.txt
```

### 2. Configure VPS Connection

Edit `app.py` and update VPS_CONFIG:

```python
VPS_CONFIG = {
    'host': '46.XXX.XXX.XXX',  # Your VPS IP
    'port': 22,
    'username': 'aml',
    'key_path': '/path/to/ssh/key',  # Or None if using password
    'password': None,  # Or your password if not using key
}
```

### 3. Configure Bot Paths

Update BOTS_CONFIG in `app.py` to match your VPS paths:

```python
BOTS_CONFIG = {
    'v13': {
        'name': 'v13 FIXED',
        'status_path': '/home/aml/bot-status/yuichi_v13.json',
        'log_path': '/home/aml/trading-bot/yuichi_bot_v13_cli.log',
        'color': '#ff4757'
    },
    # Add more bots...
}
```

### 4. Ensure Bots Write Status Files

Your bots must write JSON status files. They already do this via `write_status()` function.

Make sure your VPS has the status directory:

```bash
# On your VPS
mkdir -p /home/aml/bot-status
```

### 5. Run Dashboard

```bash
python app.py
```

Access at: `http://localhost:5000`

## SSH Authentication Options

### Option A: SSH Key (Recommended)

```python
VPS_CONFIG = {
    'host': '46.XXX.XXX.XXX',
    'username': 'aml',
    'key_path': '/home/you/.ssh/id_rsa',  # Path to private key
    'password': None
}
```

### Option B: Password

```python
VPS_CONFIG = {
    'host': '46.XXX.XXX.XXX',
    'username': 'aml',
    'key_path': None,
    'password': 'your_password'
}
```

## Status JSON Format

Your bots write status files like this:

```json
{
  "bot_name": "yuichi_v13",
  "capital": 1025.50,
  "cumulative_winnings": 25.50,
  "trades_executed_today": 15,
  "battle_step": 2,
  "martingale_step": 1,
  "running": true,
  
  "position_active": true,
  "side": "buy",
  "entry_price": 90245.50,
  "stop_loss": 89800.00,
  "take_profit": 90850.00,
  "position_size_main": 150.00,
  "position_size_hedge": 45.00,
  "battle_pnl": 12.50,
  
  "game_state": "GAME_OVER",
  "trap_type": "bear_trap",
  "trap_probability": 0.85,
  
  "last_update": "2025-01-16T18:30:45Z"
}
```

## Dashboard Features

### Global Stats Bar
- Total Capital across all bots
- Total Winnings (green if positive, red if negative)
- Total Trades Today
- Best Performer

### Bot Cards
Each bot shows:
- Running status (🟢/⚫)
- Capital & Winnings
- Trades & Steps
- **Active Position Display:**
  - Entry / SL / TP prices
  - Position sizes
  - Game state badge
  - Battle P&L
- View Logs button

### Position Chart
- Bar chart of active positions
- Shows total position size per bot
- Color-coded
- Real-time updates

## Deployment

### Local (Windows/Mac/Linux)

```bash
python app.py
# Access: http://localhost:5000
```

### Deploy as Web Service

The dashboard can run anywhere that can reach your VPS via SSH:

**Option 1: Run on your local machine**
- Keep it running in background
- Access via `localhost:5000`

**Option 2: Deploy on a server**
- Any server with SSH access to your VPS
- Use systemd service or screen session
- Access via server IP

**Option 3: Run on the VPS itself**
- Install on the same VPS as bots
- Use `0.0.0.0` as host
- Access via VPS IP

### Systemd Service (Linux)

Create `/etc/systemd/system/trading-dashboard.service`:

```ini
[Unit]
Description=Trading Bots Dashboard
After=network.target

[Service]
Type=simple
User=aml
WorkingDirectory=/home/aml/vps_dashboard
ExecStart=/usr/bin/python3 /home/aml/vps_dashboard/app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable trading-dashboard
sudo systemctl start trading-dashboard
```

## Troubleshooting

### SSH Connection Failed

**Check:**
1. VPS IP is correct
2. Port 22 is open
3. SSH credentials are valid
4. Firewall allows connection

**Test manually:**
```bash
ssh -i /path/to/key aml@46.XXX.XXX.XXX
```

### Status Files Not Found

**Check:**
1. Bots are writing status files
2. Paths in `BOTS_CONFIG` match VPS
3. Directory `/home/aml/bot-status` exists

**On VPS:**
```bash
ls -la /home/aml/bot-status/
cat /home/aml/bot-status/yuichi_v13.json
```

### Logs Not Showing

**Check:**
1. Log file paths are correct
2. Log files exist on VPS
3. User has read permission

### Dashboard Shows Old Data

- Status files update every trade
- Dashboard refreshes every 3 seconds
- Check bot is actually running on VPS

## Security Notes

⚠️ **For Production:**

1. **Use SSH Keys** (not passwords)
2. **Restrict SSH access** via firewall
3. **Use VPN** if dashboard is public
4. **Don't expose passwords** in code
5. **Use environment variables** for secrets

Example with env vars:

```python
import os

VPS_CONFIG = {
    'host': os.environ.get('VPS_HOST'),
    'username': os.environ.get('VPS_USER'),
    'password': os.environ.get('VPS_PASSWORD')
}
```

## API Endpoints

### GET /api/bots
Returns all bot statuses + summary

### GET /api/bot/<bot_id>
Returns specific bot status with logs

### GET /api/vps/status
Returns VPS connection status

## Customization

### Add More Bots

In `app.py`, add to `BOTS_CONFIG`:

```python
'my_new_bot': {
    'name': 'My Strategy',
    'status_path': '/path/to/status.json',
    'log_path': '/path/to/bot.log',
    'color': '#ff00ff'  # Choose a color
}
```

### Change Colors

Edit `static/css/style.css`:

```css
:root {
    --accent-primary: #00d4ff;  /* Change colors */
    --accent-success: #00ff88;
    --accent-danger: #ff4757;
}
```

### Adjust Update Frequency

In `app.py`:

```python
time.sleep(3)  # Change to 5 for slower updates
```

In `static/js/app.js`:

```javascript
setInterval(updateData, 3000);  # Change to 5000 for 5 seconds
```

## Support

**Dashboard Issues:**
- Check browser console (F12)
- Check `app.py` terminal output
- Verify SSH connection manually

**Bot Issues:**
- Check bot logs on VPS
- Verify status JSON is being written
- Test bot independently

---

🎮 **Monitor your bots in style!** 🚀
