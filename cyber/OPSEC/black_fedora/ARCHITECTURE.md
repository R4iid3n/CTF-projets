# Black Fedora - Project Overview

## 📋 Project Structure

```
black-fedora/
├── black-fedora.py          # Main CLI application (5000+ lines)
├── install.sh               # Installation script
├── demo.py                  # Interactive demo (no root required)
├── README.md                # Full documentation
├── QUICKSTART.md            # Quick start guide
└── ARCHITECTURE.md          # This file
```

## 🏗️ Architecture

### Core Components

#### 1. Main Application (`black-fedora.py`)

**BlackFedora Class** - Central application controller
- Configuration management (JSON-based)
- State persistence
- Color-coded terminal UI
- Command execution framework

**Key Methods:**
- `installation_menu()` - Handles full system installation
- `operational_menu()` - Main user interface loop
- `pre_flight_checks()` - Security verification system
- `create_*_script()` - Script generation methods

#### 2. Installation System

**Package Installation:**
- nftables (firewall)
- tor (anonymity network)
- torsocks (Tor wrapper)
- macchanger (MAC randomization)
- proxychains-ng (proxy chains)
- firejail (sandboxing)
- firetools (sandbox manager)
- torbrowser-launcher (Tor Browser)
- wireguard-tools (VPN)

**Script Creation:**
```
/usr/local/sbin/black-fedora/
├── fedora-anon-setup.sh      # MAC + hostname + services
├── ghost-watchdog.sh          # VPN monitoring
├── ghost-session.sh           # Secure browser launcher
└── cleanup-logs.sh            # Log sanitization
```

**Systemd Services:**
```
/etc/systemd/system/
├── black-fedora-anon.service     # Boot-time anonymization
└── black-fedora-watchdog.service # VPN watchdog daemon
```

#### 3. Security Components

**Firewall Killswitch** (`/etc/nftables/black-fedora-killswitch.nft`)
```
- Default: DROP all traffic
- Allow: VPN interface only
- Allow: Loopback
- Allow: Established connections
- Block: Everything else
```

**Firejail Profiles** (`/etc/firejail/*.local`)
```
- Private home directory
- Isolated /tmp
- No D-Bus access
- Blacklisted directories
- No execution in /tmp or /dev/shm
```

**Journald Configuration** (`/etc/systemd/journald.conf.d/black-fedora.conf`)
```
- 100MB max storage
- 3-day retention
- Compressed logs
- No syslog forwarding
```

#### 4. Configuration System

**Main Config** (`~/.config/black-fedora/config.json`)
```json
{
  "vpn_interface": "wg-mullvad",
  "tor_port": "9050",
  "primary_interface": "eth0",
  "installed": true
}
```

**State File** (`~/.config/black-fedora/state.json`)
```json
{
  "mac_randomized": false,
  "hostname_changed": false,
  "services_disabled": false,
  "firewall_active": false,
  "vpn_active": false,
  "tor_active": false
}
```

## 🔄 Operational Flow

### Installation Flow
```
1. User runs: sudo black-fedora install
2. Check root privileges
3. Install system packages (dnf)
4. Create /usr/local/sbin/black-fedora/
5. Generate security scripts
6. Create systemd service files
7. Configure nftables killswitch
8. Create firejail profiles
9. Configure journald
10. Mark as installed in config
```

### Pre-Work Flow (Recommended Daily Use)
```
1. Launch: black-fedora start
2. Run pre-flight checks
   ├── Verify VPN active
   ├── Confirm Tor running
   ├── Check firewall enabled
   └── Validate watchdog running
3. Randomize MAC address
4. Change hostname
5. Disable telemetry services
6. Enable firewall killswitch
7. Start VPN watchdog
8. Launch ghost session
   ├── Verify VPN connection
   ├── Confirm default route
   ├── Check Tor status
   └── Start sandboxed Tor Browser
```

### VPN Watchdog Loop
```
while true:
  1. Check VPN interface is UP
  2. Verify default route through VPN
  3. If failure detected:
     ├── Kill all browsers (pkill -9)
     ├── Disable networking (nmcli)
     └── Exit with error
  4. Sleep 5 seconds
  5. Repeat
```

## 🔒 Security Model

### Threat Protection

**Protected Against:**
- ✅ IP address leaks (killswitch)
- ✅ DNS leaks (VPN routing)
- ✅ VPN disconnection leaks (watchdog)
- ✅ MAC address tracking (randomization)
- ✅ Hostname identification (rotation)
- ✅ ISP surveillance (VPN + Tor)
- ✅ Browser fingerprinting (Tor Browser)
- ✅ Application sandboxing (Firejail)
- ✅ System telemetry (service disabling)

**NOT Protected Against:**
- ❌ Nation-state targeted attacks
- ❌ Physical device access
- ❌ Compromised VPN provider
- ❌ Malicious Tor exit nodes
- ❌ Browser exploits
- ❌ Social engineering
- ❌ Side-channel attacks

### Defense in Depth

**Layer 1: Network**
- VPN tunnel (WireGuard)
- Tor network (optional overlay)
- Firewall killswitch

**Layer 2: System**
- MAC randomization
- Hostname rotation
- Service disabling
- Log minimization

**Layer 3: Application**
- Sandboxed browsers (Firejail)
- Tor Browser (fingerprint resistance)
- Isolated home directories

**Layer 4: Monitoring**
- VPN watchdog (continuous)
- Pre-flight checks (manual)
- Status monitoring (on-demand)

## 🎯 Design Decisions

### Why Python?
- Cross-platform compatibility
- Rich standard library
- Easy subprocess management
- JSON configuration handling
- Colored terminal output
- Excellent error handling

### Why Modular Scripts?
- Easy troubleshooting
- Manual execution possible
- Systemd integration
- Independent testing
- Clear separation of concerns

### Why nftables over iptables?
- Modern replacement for iptables
- Better performance
- Cleaner syntax
- Atomic ruleset updates
- Better IPv6 support

### Why Firejail?
- Lightweight sandboxing
- No kernel changes required
- Easy profile customization
- Wide application support
- Active development

### Why JSON Config?
- Human readable
- Easy parsing
- Native Python support
- Extensible structure
- Version control friendly

## 🚀 Future Enhancements

### Planned Features
- [ ] Multiple VPN profile management
- [ ] Tor bridge configuration
- [ ] DNS-over-HTTPS integration
- [ ] Encrypted DNS
- [ ] USB device control
- [ ] Webcam/microphone blocking
- [ ] Network namespace isolation
- [ ] Container-based isolation
- [ ] Automatic updates
- [ ] Threat intelligence feeds

### Enhancement Ideas
- [ ] GUI version (GTK or Qt)
- [ ] Web dashboard
- [ ] Mobile companion app
- [ ] Remote management
- [ ] Scheduled operations
- [ ] Automated OSINT checks
- [ ] Integration with Qubes OS
- [ ] Hardware key support (YubiKey)
- [ ] Encrypted storage management
- [ ] Secure communication channels

## 🧪 Testing Strategy

### Installation Testing
```bash
# Test in VM first
sudo black-fedora install

# Verify files created
ls -la /usr/local/sbin/black-fedora/
ls -la /etc/systemd/system/black-fedora*
ls -la /etc/nftables/
ls -la /etc/firejail/*.local

# Check services
systemctl status black-fedora-anon
systemctl status black-fedora-watchdog
```

### Functional Testing
```bash
# Test VPN detection
nmcli connection up wg-mullvad
black-fedora status

# Test MAC randomization
ip link show eth0 | grep link/ether

# Test firewall
sudo nft list ruleset

# Test Tor
ss -lnt | grep 9050

# Test watchdog (disconnect VPN and observe)
```

### Security Testing
```bash
# DNS leak test
curl https://dnsleaktest.com

# IP address check
curl ifconfig.me

# Tor check
curl https://check.torproject.org

# Browser fingerprinting
# Visit: https://browserleaks.com
```

## 📊 Performance Considerations

### Resource Usage
- **Memory**: ~50MB (Python + scripts)
- **Disk**: ~200MB (with all packages)
- **CPU**: Minimal (<1% idle)
- **Network**: Adds ~5-10% latency (VPN + Tor)

### Optimization Tips
- Use systemd for startup (faster than cron)
- Cache DNS queries
- Use persistent Tor circuits
- Minimize log writes
- Use tmpfs for temporary files

## 🛠️ Maintenance

### Regular Updates
```bash
# Update packages
sudo dnf update

# Update Black Fedora
cd black-fedora
git pull
sudo ./install.sh
```

### Log Management
```bash
# View journald logs
sudo journalctl -u black-fedora-watchdog -f

# Clean old logs
sudo black-fedora (option 8)

# Or manually
sudo journalctl --vacuum-time=3d
```

### Backup Configuration
```bash
# Backup config
cp ~/.config/black-fedora/config.json ~/backup/

# Backup scripts
sudo cp -r /usr/local/sbin/black-fedora ~/backup/
```

## 📚 References

### Documentation Sources
- Whonix Documentation: https://www.whonix.org/wiki/Documentation
- Tails Documentation: https://tails.boum.org/doc/
- Tor Project: https://www.torproject.org/docs/
- Privacy Guides: https://www.privacyguides.org/
- Fedora Security Guide: https://docs.fedoraproject.org/en-US/security-guide/

### Technical References
- nftables Wiki: https://wiki.nftables.org/
- Firejail Documentation: https://firejail.wordpress.com/
- WireGuard Documentation: https://www.wireguard.com/
- Mullvad Guides: https://mullvad.net/en/help/
- systemd Documentation: https://www.freedesktop.org/wiki/Software/systemd/

## 🤝 Contributing

### Code Style
- PEP 8 compliant Python
- Descriptive variable names
- Comprehensive docstrings
- Type hints where appropriate
- Error handling with try/except

### Script Standards
- Bash strict mode (set -e)
- Descriptive comments
- Error messages to stderr
- Exit codes: 0=success, 1=failure
- Use functions for reusability

### Documentation
- Update README for new features
- Add to QUICKSTART for user features
- Update ARCHITECTURE for structural changes
- Include examples in docs
- Keep changelog updated

## 📝 License & Legal

### Usage Terms
- For legitimate privacy protection only
- Users responsible for legal compliance
- No warranty provided
- Use at your own risk
- Check local laws regarding VPN/Tor usage

### Ethical Considerations
- Respect others' privacy
- Don't facilitate illegal activities
- Report security vulnerabilities responsibly
- Contribute improvements back
- Help others learn about privacy

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready  
**Python Version**: 3.6+  
**OS**: Fedora Linux 38+

**Built with 🖤 for privacy enthusiasts and security researchers**
