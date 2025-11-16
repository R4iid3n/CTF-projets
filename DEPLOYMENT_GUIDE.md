# 🚀 Deployment Guide - Trading Bots Dashboard

## What You'll Get

A beautiful web dashboard that runs all 3 bots in parallel with:
- ✅ Real-time updates (WebSocket)
- ✅ Live equity curves
- ✅ Performance comparison
- ✅ View & download bot code
- ✅ Start/stop individual or all bots
- ✅ Dark theme optimized for trading

## Dashboard Features

### 1. **Global Stats Bar**
```
┌─────────────────────────────────────────────────┐
│  Total Capital    Total Winnings    Trades    │
│    $3,000.00         $0.00            0       │
└─────────────────────────────────────────────────┘
```

### 2. **Bot Cards (3 columns)**

Each card shows:
- Bot name & status (🟢 Running / ⚫ Stopped)
- Philosophy bullet points
- Real-time metrics:
  - Capital
  - Cumulative Winnings (green if +, red if -)
  - Total Trades
  - Win Rate %
- Control buttons:
  - ▶️ Start
  - ⏸️ Stop
  - 📄 View Code
- Recent activity log (last 10 messages)

### 3. **Performance Chart**
Live equity curve comparing all 3 bots:
- v13 FIXED (red line)
- v13.5 ENHANCED (blue line)
- v14 TRUE METHOD (green line)

### 4. **Code Viewer Modal**
Click "View Code" on any bot to:
- See full source code
- Download .py file
- Review implementation

---

## Deployment Options

### 🥇 RECOMMENDED: Render.com (100% Free)

**Why Render:**
- ✅ Free tier (no credit card needed)
- ✅ Auto-deploy from GitHub
- ✅ WebSocket support
- ✅ SSL certificate included
- ✅ Easy to use

**Steps:**

1. **Push to GitHub**
```bash
cd trading_dashboard
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/trading-dashboard.git
git push -u origin main
```

2. **Deploy on Render**
   - Go to https://render.com
   - Sign up (free)
   - Click "New +" → "Web Service"
   - Connect GitHub repository
   - Settings:
     ```
     Name: trading-bots-dashboard
     Environment: Python 3
     Build Command: pip install -r requirements.txt
     Start Command: python app.py
     ```
   - Click "Create Web Service"

3. **Access Your Dashboard**
   - URL: `https://trading-bots-dashboard.onrender.com`
   - Wait 2-3 minutes for first deployment

**Note:** Free tier sleeps after 15min inactivity. First request may take 30s.

---

### 🥈 Railway.app (Free Trial)

**Why Railway:**
- ✅ Very fast deployment
- ✅ Auto-detects everything
- ✅ Great developer experience

**Steps:**

1. **Push to GitHub** (same as above)

2. **Deploy on Railway**
   - Go to https://railway.app
   - Sign up with GitHub
   - "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Python and deploys
   - Get URL from deployment

**Note:** Free trial includes $5 credit (~500 hours)

---

### 🥉 PythonAnywhere (Always Free)

**Why PythonAnywhere:**
- ✅ Truly free forever
- ✅ No credit card needed
- ✅ Good for learning

**Steps:**

1. **Create Account**
   - https://www.pythonanywhere.com
   - Sign up for free "Beginner" account

2. **Upload Files**
   - Go to "Files" tab
   - Upload `trading_dashboard.tar.gz`
   - Extract: `tar -xzf trading_dashboard.tar.gz`

3. **Install Dependencies**
   - Go to "Consoles" tab
   - Start Bash console
   ```bash
   pip install --user -r requirements.txt
   ```

4. **Create Web App**
   - "Web" tab → "Add a new web app"
   - Choose "Flask"
   - Python version: 3.10
   - Path: `/home/yourusername/trading_dashboard/app.py`

5. **Configure WSGI**
   Edit the WSGI file:
   ```python
   import sys
   path = '/home/yourusername/trading_dashboard'
   if path not in sys.path:
       sys.path.append(path)
   
   from app import app as application
   ```

6. **Reload**
   - Click "Reload" button
   - Access: `https://yourusername.pythonanywhere.com`

**Limitations:**
- No WebSocket on free tier (real-time updates won't work)
- Use manual refresh instead

---

## Local Testing First

Before deploying, test locally:

```bash
cd trading_dashboard

# Install dependencies
pip install -r requirements.txt

# Run
python app.py

# Open browser
http://localhost:5000
```

You should see:
- Dashboard loads
- 3 bot cards displayed
- Charts rendering
- Buttons functional

Test:
1. Click "Start All Bots"
2. Watch metrics update in real-time
3. Check equity curve updates
4. Click "View Code" on a bot
5. Try downloading code

---

## Connecting Real Bots

Currently, bots are simulated. To use real bots:

### Option 1: Replace Simulation Functions

In `app.py`, find:
```python
def simulate_bot_v13(bot_state):
    # Replace this entire function
```

Replace with your real bot logic:
```python
def run_real_bot_v13(bot_state):
    # Import your bot
    from yuichi_bot_v13_fixed import execute_yuichi_strategy
    
    # Run it
    execute_yuichi_strategy(
        bot_state=bot_state,
        socketio=socketio
    )
```

### Option 2: Modify Bot Files

Edit your bot files to integrate with dashboard:

```python
# In your bot file
def execute_yuichi_strategy(bot_state, socketio):
    while bot_state.running:
        # Your trading logic
        
        # Update dashboard
        bot_state.cumulative_winnings = config.cumulative_winnings
        bot_state.trades_executed = config.trades_executed
        bot_state.capital = config.capital
        
        # Emit update
        socketio.emit('bot_update', {
            'bot_id': 'v13',
            'data': bot_state.to_dict()
        })
        
        time.sleep(3)
```

---

## Environment Variables (Production)

For security, set these:

**On Render:**
- Dashboard → Environment
- Add:
  ```
  SECRET_KEY=your-random-secret-key-here
  BINANCE_API_KEY=your-api-key
  BINANCE_SECRET=your-secret
  ```

**In app.py:**
```python
import os

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key')
```

---

## Troubleshooting

### Bots Not Starting
- Check browser console (F12)
- Verify WebSocket connection
- Check server logs

### Charts Not Updating
- Ensure Chart.js CDN loads
- Check data format in equity_curve
- Verify WebSocket events

### Code Viewer Shows Error
- Ensure bot files are in correct path
- Check file permissions
- Verify `/mnt/user-data/outputs/` exists

### Real-time Updates Slow
- WebSocket might be sleeping (Render free tier)
- Increase update interval
- Use manual refresh button

---

## Cost Comparison

| Platform | Free Tier | Limitations | Best For |
|----------|-----------|-------------|----------|
| **Render** | ✅ Free | Sleeps after 15min | Demos, testing |
| **Railway** | $5 credit | ~500 hours | Development |
| **PythonAnywhere** | ✅ Free | No WebSocket | Learning |
| **Heroku** | ❌ No free | $7/month | Production |
| **AWS/GCP** | Credits | Complex | Enterprise |

---

## Next Steps

1. ✅ Test locally
2. ✅ Choose deployment platform
3. ✅ Deploy dashboard
4. ✅ Test with simulated bots
5. ✅ Integrate real bot logic
6. ✅ Add authentication (optional)
7. ✅ Monitor performance

---

## Support

**Dashboard Issues:**
- Check `app.py` logs
- Browser console (F12)
- Network tab for API calls

**Bot Issues:**
- Check individual bot logs
- Verify API connections
- Test bots independently first

---

🎮 **Enjoy your Trading Dashboard!** 🚀
