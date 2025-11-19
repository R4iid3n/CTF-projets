# Black Fedora - Practical Usage Examples

## 📖 Real-World Scenarios

### Scenario 1: First Time Setup

**Goal**: Install and configure Black Fedora on a fresh Fedora system

```bash
# Step 1: Download and install
cd ~/Downloads
git clone <repo-url> black-fedora
cd black-fedora
sudo ./install.sh

# Step 2: Run installation
sudo black-fedora install
# Wait 5-10 minutes for package installation

# Step 3: Import your VPN
# Download Mullvad config first from mullvad.net
sudo nmcli connection import type wireguard file ~/Downloads/mullvad-us-nyc-001.conf

# Step 4: Configure Black Fedora
black-fedora configure
# Enter: wg-mullvad (VPN interface)
# Enter: 9050 (Tor port)
# Enter: eth0 (your network interface)

# Step 5: Test VPN connection
sudo nmcli connection up wg-mullvad
curl ifconfig.me  # Should show VPN IP

# Step 6: Start Tor
sudo systemctl start tor

# Step 7: Ready to use!
black-fedora start
```

---

### Scenario 2: Daily Work Session

**Goal**: Secure anonymous browsing session for research

```bash
# Morning routine
black-fedora start

# Select: 1 (Operational Mode)

# Run pre-flight checks
Select: 1
# Output:
#   ✓ VPN Connection
#   ✓ Tor Service
#   ✓ Firewall Active
#   ✓ Watchdog Running
# All checks passed!

# Randomize MAC (if on new network)
Select: 2
# MAC address changed

# Change hostname (for extra paranoia)
Select: 3
# Hostname changed to host-a8f3d2e1

# Enable killswitch
Select: 5
# Firewall killswitch active

# Start watchdog (auto-kill if VPN drops)
Select: 6
# Watchdog monitoring started

# Launch ghost session
Select: 7
# Tor Browser launched in sandbox

# Now browse anonymously!
# Tor Browser is running with:
# - VPN tunnel
# - Tor anonymity
# - Firejail sandbox
# - MAC randomized
# - Hostname changed
# - Watchdog monitoring

# When done, verify your privacy
Select: 9
# Visit the OSINT check sites

# Clean up before shutdown
Select: 8
# Logs cleaned
```

---

### Scenario 3: CTF Competition

**Goal**: Anonymous reconnaissance without leaving traces

```bash
# Pre-competition setup
black-fedora start

# Full anonymization sequence
Select: 1 (Operational Mode)
# Then run: 1, 2, 3, 4, 5, 6 in sequence

# Launch tools
Select: 7  # Tor Browser for OSINT

# Parallel terminal for recon tools
# In another terminal:
firejail --private nmap -sn <target-range>
firejail --private --net=none maltego

# Monitor your connection
watch -n 1 'ip route show default'

# If VPN drops, watchdog will:
# - Kill all browsers
# - Disable networking
# - Protect your IP

# After competition
Select: 8  # Clean all logs
```

---

### Scenario 4: Investigating Suspicious Activity

**Goal**: Safely investigate potentially malicious websites

```bash
# Setup isolated environment
black-fedora start
Select: 1 (Operational Mode)

# Full security protocol
Select: 1  # Pre-flight checks
Select: 2  # Randomize MAC
Select: 3  # Change hostname
Select: 5  # Enable killswitch
Select: 6  # Start watchdog
Select: 7  # Launch ghost session

# In Tor Browser:
# 1. Visit site through Tor
# 2. Take screenshots
# 3. Analyze behavior
# 4. Document findings

# Additional isolation
# Create throwaway Firefox profile:
firejail --private=/tmp/investigation-$(date +%s) \
         --private-tmp \
         --nodbus \
         --nosound \
         firefox --new-instance

# After investigation
Select: 8  # Clean logs
sudo shred -vfz -n 10 /tmp/investigation-*
```

---

### Scenario 5: Public WiFi Usage

**Goal**: Secure connection on untrusted network

```bash
# Connect to public WiFi
nmcli device wifi connect "Coffee Shop WiFi"

# Immediately secure your connection
black-fedora start
Select: 1 (Operational Mode)

# CRITICAL: Randomize MAC first!
Select: 2
# This prevents WiFi tracking

# Change hostname
Select: 3
# Prevents network identification

# Disable telemetry
Select: 4
# Stops location/service broadcasts

# Connect VPN
sudo nmcli connection up wg-mullvad

# Enable killswitch
Select: 5
# Blocks all non-VPN traffic

# Start watchdog
Select: 6
# Auto-protection if WiFi drops

# Pre-flight check
Select: 1
# Verify everything is secure

# Launch secure session
Select: 7

# Additional hardening
# Disable IPv6 to prevent leaks:
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1

# Now safe to work on public WiFi!
```

---

### Scenario 6: Whistleblowing / Sensitive Communications

**Goal**: Maximum anonymity for sensitive information sharing

```bash
# EXTREME PARANOIA MODE

# Step 1: Physical security
# - Use live USB/DVD (Tails OS recommended)
# - Or fresh VM with no identifying info
# - Public computer (library, internet cafe)
# - Public WiFi (not your home/work)

# Step 2: Black Fedora full setup
black-fedora start

# Step 3: Complete anonymization
Select: 1, then run each:
2 - Randomize MAC
3 - Change hostname  
4 - Disable services
5 - Enable killswitch
6 - Start watchdog
7 - Launch ghost session

# Step 4: Additional layers
# In terminal:
sudo systemctl stop NetworkManager-wait-online
sudo rm -rf ~/.bash_history
history -c

# Step 5: Use Tor Browser only
# - Don't login to any accounts
# - Don't reveal identifying info
# - Use onion services when possible
# - Consider Tails OS for maximum security

# Step 6: Upload through Tor
# Use SecureDrop or similar:
# - https://securedrop.org/directory
# - Access via .onion address only
# - Never use clearnet

# Step 7: Complete cleanup
Select: 8  # Clean logs
sudo shred -vfz -n 10 ~/.bash_history
sudo shred -vfz -n 10 /var/log/*

# Step 8: Reboot (destroys RAM)
sudo reboot
```

---

### Scenario 7: Testing Your Setup

**Goal**: Verify Black Fedora is working correctly

```bash
# Test 1: Check installation
black-fedora status
# Should show all green checkmarks

# Test 2: Verify VPN killswitch
# Terminal 1:
black-fedora start
Select: 5  # Enable killswitch
Select: 6  # Start watchdog

# Terminal 2:
curl ifconfig.me  # Note VPN IP

# Terminal 3:
sudo nmcli connection down wg-mullvad
# Watchdog should kill browsers and disable network

# Test 3: Verify MAC randomization
ip link show eth0 | grep link/ether  # Note MAC
black-fedora start
Select: 2  # Randomize
ip link show eth0 | grep link/ether  # Should be different

# Test 4: Verify Tor
black-fedora start
Select: 7  # Launch ghost session
# In Tor Browser visit:
https://check.torproject.org
# Should say "Congratulations. This browser is configured to use Tor."

# Test 5: Verify DNS
# Visit: https://dnsleaktest.com
# Should show VPN DNS only, no ISP DNS

# Test 6: Verify fingerprinting protection
# Visit: https://browserleaks.com
# Check Canvas, WebGL, Audio, Fonts
# Tor Browser should block most fingerprinting

# Test 7: Verify firewall
sudo nft list ruleset | grep -A 20 "chain output"
# Should see policy drop and VPN interface

# Test 8: Stress test watchdog
# While browsing, repeatedly disconnect/reconnect VPN
# Watchdog should consistently kill browsers
```

---

### Scenario 8: Emergency Shutdown

**Goal**: Quickly disable all anonymization if needed

```bash
# Quick shutdown script
cat > /tmp/emergency-shutdown.sh << 'EOF'
#!/bin/bash
# Emergency shutdown - disables all Black Fedora protections

# Stop watchdog
sudo systemctl stop black-fedora-watchdog

# Flush firewall
sudo nft flush ruleset

# Disconnect VPN
sudo nmcli connection down wg-mullvad

# Stop Tor
sudo systemctl stop tor

# Kill browsers
pkill -9 torbrowser firefox librewolf chromium

# Restore original MAC (if you know it)
# sudo ip link set eth0 address XX:XX:XX:XX:XX:XX

# Re-enable networking
sudo nmcli networking on

echo "Emergency shutdown complete"
EOF

chmod +x /tmp/emergency-shutdown.sh

# Use when needed:
/tmp/emergency-shutdown.sh
```

---

### Scenario 9: Automated Daily Workflow

**Goal**: Script your daily security routine

```bash
# Create startup script
cat > ~/start-ghost-mode.sh << 'EOF'
#!/bin/bash

echo "Starting Ghost Mode..."

# Connect VPN
sudo nmcli connection up wg-mullvad
sleep 3

# Start Tor
sudo systemctl start tor
sleep 2

# Run Black Fedora setup (non-interactive)
sudo macchanger -r eth0
sudo hostnamectl set-hostname "host-$(openssl rand -hex 4)"
sudo systemctl disable --now avahi-daemon cups bluetooth geoclue

# Enable firewall
sudo nft -f /etc/nftables/black-fedora-killswitch.nft

# Start watchdog
sudo systemctl start black-fedora-watchdog

# Verify
if ip link show wg-mullvad up &>/dev/null && \
   ss -lnt | grep -q ":9050" && \
   sudo nft list ruleset | grep -q "wg-mullvad"; then
    echo "✓ Ghost Mode Active"
    
    # Launch Tor Browser
    firejail --profile=/etc/firejail/torbrowser-launcher.local torbrowser-launcher &
else
    echo "✗ Ghost Mode Failed - Check configuration"
    exit 1
fi
EOF

chmod +x ~/start-ghost-mode.sh

# Use daily:
~/start-ghost-mode.sh
```

---

### Scenario 10: Multi-Hop VPN + Tor

**Goal**: Maximum anonymity with layered protection

```bash
# Setup: VPN → VPN → Tor

# Step 1: First VPN (Mullvad)
sudo nmcli connection up wg-mullvad

# Step 2: Second VPN over first (ProtonVPN)
sudo nmcli connection up protonvpn
# Edit routing to go through first VPN

# Step 3: Configure Tor to use VPN
sudo nano /etc/tor/torrc
# Add:
# SocksPort 9050
# HTTPTunnelPort 9053

# Step 4: Black Fedora
black-fedora start
Select: 1 (Operational)
Select: 5 (Firewall)
Select: 6 (Watchdog)
Select: 7 (Ghost session)

# Now traffic flow:
# You → VPN1 → VPN2 → Tor → Internet

# Verify each hop:
curl --proxy socks5://127.0.0.1:9050 ifconfig.me
```

---

## 🎯 Best Practices Summary

### Always Do:
1. ✅ Run pre-flight checks before sensitive work
2. ✅ Enable VPN watchdog
3. ✅ Randomize MAC on new networks
4. ✅ Use Tor Browser for anonymity
5. ✅ Verify with OSINT checks
6. ✅ Clean logs regularly
7. ✅ Keep software updated
8. ✅ Use strong VPN provider

### Never Do:
1. ❌ Login to personal accounts on Tor
2. ❌ Disable killswitch without VPN backup
3. ❌ Trust single anonymization layer
4. ❌ Use same identity across sessions
5. ❌ Ignore watchdog warnings
6. ❌ Skip verification tests
7. ❌ Use cracked/pirated VPN
8. ❌ Assume perfect anonymity

---

## 📞 Quick Reference Commands

```bash
# Installation
sudo black-fedora install

# Configuration
black-fedora configure

# Start session
black-fedora start

# Quick status
black-fedora status

# Emergency stop
sudo systemctl stop black-fedora-watchdog
sudo nft flush ruleset

# Manual VPN
sudo nmcli connection up wg-mullvad
sudo nmcli connection down wg-mullvad

# Check VPN
ip route show default

# Check Tor
ss -lnt | grep 9050

# Check MAC
ip link show eth0 | grep link/ether

# Clean logs
sudo journalctl --vacuum-time=3d

# Test anonymity
curl ifconfig.me
curl --proxy socks5://127.0.0.1:9050 ifconfig.me
```

---

Stay safe, stay anonymous! 🎭
