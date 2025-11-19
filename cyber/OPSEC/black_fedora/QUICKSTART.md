# Black Fedora - Quick Start Guide

## 🚀 Installation (5 minutes)

### 1. Install the CLI Tool
```bash
sudo ./install.sh
```

### 2. Run Full Installation
```bash
sudo black-fedora install
```
This installs all packages and creates security scripts (~5-10 minutes)

### 3. Import VPN Config
```bash
# Download your Mullvad WireGuard config first
sudo nmcli connection import type wireguard file ~/Downloads/mullvad-us.conf
```

### 4. Configure
```bash
black-fedora configure
```
- VPN Interface: `wg-mullvad` (or your VPN interface name)
- Tor Port: `9050` (default)
- Primary Interface: Auto-detected (usually `eth0` or `wlan0`)

## 🎯 Daily Usage Workflow

### Launch Black Fedora
```bash
black-fedora
```

### Pre-Work Checklist (First Time Each Session)

**Menu Navigation:**
```
1. Select "Operational Mode"
2. Run through these options in order:
```

| # | Action | Purpose |
|---|--------|---------|
| 1 | Pre-flight checks | Verify all systems operational |
| 2 | Randomize MAC | Prevent hardware tracking |
| 3 | Change hostname | Prevent network identification |
| 4 | Disable services | Stop telemetry leaks |
| 5 | Enable firewall | Activate killswitch |
| 6 | Start watchdog | Monitor VPN continuously |
| 7 | Launch ghost session | Start anonymous browsing |

### After Checklist Complete

✅ Your system is now in full ghost mode:
- VPN active with killswitch
- MAC address randomized
- Hostname changed
- Telemetry disabled
- Watchdog monitoring
- Tor Browser ready

## 📋 Quick Commands

```bash
# Check status
black-fedora status

# Start operational mode directly
black-fedora start

# Reconfigure settings
black-fedora configure

# Reinstall everything
sudo black-fedora install
```

## 🔍 Verify Your Anonymity

After setup, test with:
1. https://dnsleaktest.com - Check for DNS leaks
2. https://browserleaks.com - Full privacy test
3. https://check.torproject.org - Verify Tor connection

## ⚠️ Important Notes

### VPN Watchdog Behavior
- Monitors VPN every 5 seconds
- **Kills all browsers** if VPN drops
- **Disables networking** completely
- This is intentional for security!

### To Resume After Watchdog Kill
1. Fix your VPN connection
2. Re-enable networking: `sudo nmcli networking on`
3. Restart Black Fedora

### Firewall Killswitch
- Blocks **ALL** traffic except through VPN
- If VPN is down, you have **NO** internet access
- This prevents IP leaks!

## 🎮 Interactive Menu Guide

```
╔══════════════════════════════════════════════════════════╗
║  BLACK FEDORA - Advanced OPSEC & Anonymity Manager      ║
╚══════════════════════════════════════════════════════════╝

Main Menu:
  1. Operational Mode    ← Your main interface
  2. Configuration       ← Update settings
  3. Reinstall           ← Fresh install
  q. Quit

Operational Mode Menu:
  1. Pre-flight checks        ← Run first!
  2. Randomize MAC address    ← Hardware anonymity
  3. Change hostname          ← Network anonymity
  4. Disable telemetry        ← Stop leaks
  5. Enable firewall          ← Activate killswitch
  6. Start VPN watchdog       ← Auto-protection
  7. Launch ghost session     ← Start browsing
  8. Cleanup logs             ← Clear traces
  9. Run OSINT checks         ← Verify privacy
  s. Show full status         ← Detailed info
  q. Quit
```

## 🛠️ Troubleshooting

### "VPN is not active"
```bash
# List connections
nmcli connection show

# Activate VPN
sudo nmcli connection up wg-mullvad
```

### "Tor is not running"
```bash
# Start Tor
sudo systemctl start tor

# Check status
sudo systemctl status tor
```

### "Cannot reach internet"
Your firewall killswitch is active! This is correct.
- Make sure VPN is connected
- Verify VPN interface name in config matches reality

### Watchdog keeps killing browsers
Your VPN connection is unstable:
1. Check VPN service status
2. Try different VPN server
3. Temporarily disable watchdog: `sudo systemctl stop black-fedora-watchdog`

## 🎓 Pro Tips

### Auto-Start on Boot
```bash
# Enable anonymization on boot
sudo systemctl enable black-fedora-anon.service

# Enable watchdog on boot
sudo systemctl enable black-fedora-watchdog.service
```

### Quick MAC Randomization
```bash
sudo macchanger -r eth0
```

### Quick Status Check
```bash
# Check VPN
ip route show default

# Check Tor
ss -lnt | grep 9050

# Check firewall
sudo nft list ruleset
```

### Multiple VPN Profiles
```bash
# Switch between VPN servers
sudo nmcli connection up wg-mullvad-us
sudo nmcli connection up wg-mullvad-se
sudo nmcli connection up wg-mullvad-jp
```

## 📱 Threat Model

Black Fedora protects against:
- ✅ ISP surveillance
- ✅ Government mass surveillance
- ✅ IP address tracking
- ✅ DNS leaks
- ✅ VPN disconnection leaks
- ✅ Browser fingerprinting (with Tor)
- ✅ MAC address tracking
- ✅ Hostname identification
- ✅ System telemetry

Black Fedora does NOT protect against:
- ❌ Targeted nation-state attacks
- ❌ Malicious browser extensions
- ❌ Compromised Tor exit nodes
- ❌ Physical access to device
- ❌ Social engineering
- ❌ Application-level leaks

## 🔐 Best Practices

1. **Always run pre-flight checks** before sensitive work
2. **Enable watchdog** for automatic protection
3. **Use Tor Browser** for maximum anonymity
4. **Clear logs regularly** (option 8)
5. **Verify with OSINT checks** (option 9)
6. **Randomize MAC** on new networks
7. **Change hostname** regularly
8. **Use VPN + Tor** for layered protection

## 📊 Status Indicators

```
Current Status:

  VPN Active:        ✓  (Good - Protected)
  Tor Running:       ✓  (Good - Anonymous)
  Firewall Active:   ✓  (Good - Killswitch on)
  Watchdog Running:  ✓  (Good - Monitoring)
```

Green ✓ = All good
Red ✗ = Needs attention

## 🚨 Emergency Shutdown

If you need to quickly disable everything:
```bash
# Stop watchdog
sudo systemctl stop black-fedora-watchdog

# Disable firewall
sudo nft flush ruleset

# Disconnect VPN
sudo nmcli connection down wg-mullvad

# Stop Tor
sudo systemctl stop tor
```

## 📞 Getting Help

1. Check detailed status: `black-fedora` → option 's'
2. Review logs: `sudo journalctl -xe`
3. Test VPN: `curl ifconfig.me`
4. Test Tor: Visit https://check.torproject.org

---

**Stay safe, stay anonymous! 🎭**
