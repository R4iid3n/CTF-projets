# 🎮 Trading Bots Dashboard

Multi-strategy trading bots platform with real-time monitoring and parallel execution.

## Features

✅ **3 Bots Running in Parallel**
- v13 FIXED - Multi-confirmation with hedge
- v13.5 ENHANCED - Optimized with scoring system ⭐
- v14 TRUE METHOD - Yuichi psychological warfare

✅ **Real-time Updates**
- WebSocket-powered live data
- Performance metrics
- Trade history
- Equity curves

✅ **Full Transparency**
- View source code of each bot
- Download bot files
- Detailed philosophy explanations

✅ **Modern UI**
- Dark theme optimized for traders
- Responsive design
- Real-time charts (Chart.js)

## Quick Start (Local)

### 1. Install Dependencies

```bash
cd trading_dashboard
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Open Browser

Navigate to: `http://localhost:5000`

## Deployment Options

### Option 1: Render.com (Free)

1. **Create account** at https://render.com
2. **New Web Service** → Connect your GitHub repo
3. **Settings:**
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
4. **Deploy!**

Your app will be live at: `https://your-app-name.onrender.com`

### Option 2: Railway.app (Free)

1. **Create account** at https://railway.app
2. **New Project** → Deploy from GitHub
3. Railway auto-detects Python and requirements.txt
4. **Deploy!**

### Option 3: PythonAnywhere (Free)

1. **Create account** at https://www.pythonanywhere.com
2. **Upload files** via Files tab
3. **Web tab** → Add new web app
4. **Configure WSGI file**:

```python
import sys
path = '/home/yourusername/trading_dashboard'
if path not in sys.path:
    sys.path.append(path)

from app import app as application
```

5. **Reload web app**

### Option 4: Heroku

1. **Create** `Procfile`:
```
web: python app.py
```

2. **Deploy:**
```bash
heroku create your-app-name
git push heroku main
```

## Project Structure

```
trading_dashboard/
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html             # Main dashboard
├── static/
│   ├── css/
│   │   └── style.css          # Dark theme styles
│   └── js/
│       └── app.js             # Client-side logic
└── README.md                  # This file
```

## API Endpoints

### Get All Bots
```
GET /api/bots
```

### Get Specific Bot
```
GET /api/bot/<bot_id>
```

### Start Bot
```
POST /api/bot/<bot_id>/start
```

### Stop Bot
```
POST /api/bot/<bot_id>/stop
```

### Start All Bots
```
POST /api/bots/start_all
```

### Stop All Bots
```
POST /api/bots/stop_all
```

### View Bot Code
```
GET /api/bot/<bot_id>/code
```

## WebSocket Events

### Client → Server
- `connect` - Initial connection
- `request_update` - Request full state update

### Server → Client
- `connected` - Connection confirmed
- `bot_update` - Single bot update
- `full_update` - Complete state update

## Bot Integration

To connect real bot logic, replace the `simulate_bot_*` functions in `app.py` with your actual trading bot code.

Each bot should:
1. Update `bot_state` attributes
2. Append to `trade_history`
3. Add log messages
4. Emit updates via SocketIO

Example:
```python
def run_real_bot_v13(bot_state):
    while bot_state.running:
        # Your trading logic here
        
        # Update state
        bot_state.current_price = get_current_price()
        bot_state.cumulative_winnings += pnl
        
        # Log trade
        bot_state.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'profit': pnl,
            'result': 'WIN' if pnl > 0 else 'LOSS'
        })
        
        # Emit update
        socketio.emit('bot_update', {
            'bot_id': 'v13',
            'data': bot_state.to_dict()
        })
        
        time.sleep(3)
```

## Customization

### Change Colors

Edit `static/css/style.css`:
```css
:root {
    --accent-primary: #00d4ff;  /* Main accent */
    --accent-success: #00ff88;  /* Success/profit */
    --accent-danger: #ff4757;   /* Danger/loss */
}
```

### Add More Bots

In `app.py`, add to `bots` dictionary:
```python
bots['v15'] = BotState(
    bot_id='v15',
    name='My Custom Bot',
    description='Description here',
    philosophy=['Point 1', 'Point 2']
)
```

## Security Notes

⚠️ **For Production:**
1. Add authentication (Flask-Login)
2. Use HTTPS
3. Set SECRET_KEY from environment variable
4. Add rate limiting
5. Implement proper error handling

## Troubleshooting

**Port already in use:**
```bash
# Change port in app.py
socketio.run(app, port=5001)
```

**WebSocket not connecting:**
- Check firewall settings
- Ensure `cors_allowed_origins="*"` for development
- For production, specify exact origin

**Charts not showing:**
- Check browser console for errors
- Ensure Chart.js CDN is accessible
- Verify data format in equity_curve

## License

MIT License - Feel free to modify and use for your projects!

## Support

For issues or questions:
1. Check browser console (F12)
2. Check server logs
3. Verify all dependencies installed

## Credits

Built with:
- Flask (Web framework)
- Socket.IO (Real-time communication)
- Chart.js (Data visualization)
- Modern CSS Grid & Flexbox

---

🎮 **Happy Trading!** 🚀
