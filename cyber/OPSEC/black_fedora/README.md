# Black Fedora - Advanced OPSEC & Anonymity CLI Manager

A comprehensive security hardening and anonymization toolkit for Fedora Linux, inspired by hardcore OPSEC practices. Black Fedora provides a professional CLI interface for managing VPN killswitches, Tor integration, MAC randomization, hostile service disabling, and complete anonymization workflows.

## 🎯 Features

### Installation Mode
- **Automated Package Installation**: Installs all necessary security tools (Tor, Firejail, nftables, macchanger, etc.)
- **Script Deployment**: Creates all anonymization, watchdog, and cleanup scripts
- **Systemd Service Creation**: Sets up automatic services for anonymization and VPN monitoring
- **Firewall Killswitch**: Configures strict nftables rules to prevent non-VPN traffic
- **Firejail Profiles**: Creates hardened sandbox profiles for browsers
- **Log Minimization**: Configures journald for minimal logging

### Operational Mode
- **Pre-flight Checks**: Verify all security components before starting work
- **MAC Randomization**: Randomize network interface MAC addresses
- **Hostname Rotation**: Change system hostname to random values
- **Service Disabling**: Disable telemetry and tracking services
- **Firewall Control**: Enable/disable killswitch firewall
- **VPN Watchdog**: Monitor VPN connection and kill browsers if it drops
- **Ghost Session Launcher**: Start fully anonymous browsing sessions
- **Log Cleanup**: Secure deletion of system logs
- **OSINT Checks**: Quick access to privacy testing websites

## 📋 Requirements

- Fedora Linux (tested on Fedora 38+)
- Root/sudo access
- Python 3.6+
- Mullvad VPN account (or any WireGuard VPN)

## 🚀 Installation

### Step 1: Download Black Fedora

```bash
git clone <repository-url>
cd black-fedora
```

### Step 2: Install the CLI Tool

```bash
sudo ./install.sh
```

This will:
- Install the `black-fedora` command globally
- Create necessary directories
- Set up the environment

### Step 3: Run Full Installation

```bash
sudo black-fedora install
```

This will:
- Install all required packages (Tor, Firejail, nftables, etc.)
- Create anonymization scripts
- Set up systemd services
- Configure firewall killswitch
- Create Firejail sandboxing profiles
- Configure minimal logging

### Step 4: Import Your VPN Configuration

Import your Mullvad WireGuard configuration:

```bash
sudo nmcli connection import type wireguard file ~/path/to/mullvad-config.conf
```

### Step 5: Configure Black Fedora

```bash
black-fedora configure
```

Update the VPN interface name, Tor port, and primary network interface.

## 📖 Usage

### Interactive Mode

Simply run:
```bash
black-fedora
```

This launches the interactive menu system.

### Direct Commands

```bash
# Run installation
black-fedora install

# Configure settings
black-fedora configure

# Start operational mode
black-fedora start

# Show status
black-fedora status
```

## 🎮 Operational Workflow

### Recommended Pre-Work Checklist

1. **Start Black Fedora**
   ```bash
   black-fedora start
   ```

2. **Run Pre-flight Checks** (Option 1)
   - Verifies VPN is active
   - Checks Tor is running
   - Confirms firewall is active
   - Validates watchdog is monitoring

3. **Randomize MAC Address** (Option 2)
   - Changes your network interface MAC address
   - Prevents hardware identification

4. **Change Hostname** (Option 3)
   - Randomizes your system hostname
   - Prevents network identification

5. **Disable Telemetry Services** (Option 4)
   - Stops avahi-daemon (mDNS leaks)
   - Stops cups (printer discovery)
   - Stops bluetooth
   - Stops geoclue (location services)

6. **Enable Firewall Killswitch** (Option 5)
   - Activates strict nftables rules
   - Blocks all non-VPN traffic
   - Prevents IP leaks

7. **Start VPN Watchdog** (Option 6)
   - Monitors VPN connection continuously
   - Kills browsers if VPN drops
   - Disables networking on failure

8. **Launch Ghost Session** (Option 7)
   - Verifies all security checks
   - Launches sandboxed Tor Browser
   - Ready for anonymous work

## 🔒 Security Components

### Firewall Killswitch

Black Fedora uses nftables to create a strict killswitch that:
- Drops all traffic by default
- Only allows traffic through VPN interface
- Allows loopback and established connections
- Blocks all traffic if VPN drops

Configuration file: `/etc/nftables/black-fedora-killswitch.nft`

### VPN Watchdog

Continuously monitors your VPN connection:
- Checks VPN interface status every 5 seconds
- Verifies default route goes through VPN
- Kills all browsers immediately if VPN fails
- Disables all networking on failure

Service: `black-fedora-watchdog.service`

### Ghost Session

Secure browser launch with multiple checks:
1. Verifies VPN is active
2. Confirms default route through VPN
3. Checks Tor is running
4. Launches Tor Browser in Firejail sandbox

Script: `/usr/local/sbin/black-fedora/ghost-session.sh`

### Firejail Sandboxing

All browsers run in hardened Firejail profiles:
- Private home directory
- No access to documents/downloads
- Disabled sound/video (optional)
- No D-Bus access
- Temporary filesystem
- Non-executable /tmp and /dev/shm

Profiles: `/etc/firejail/*.local`

### Log Minimization

- Journald limited to 100MB storage
- 3-day log retention
- System logs cleared on demand
- Audit logs disabled

## 🛡️ OSINT Verification

After setup, verify your anonymity:

1. **DNS Leak Test**: https://dnsleaktest.com
2. **Browser Fingerprinting**: https://browserleaks.com
3. **Tracking Protection**: https://panopticlick.eff.org
4. **Cover Your Tracks**: https://coveryourtracks.eff.org
5. **Tor Check**: https://check.torproject.org

## 📁 File Locations

### Configuration
- Main config: `~/.config/black-fedora/config.json`
- State file: `~/.config/black-fedora/state.json`

### Scripts
- Scripts directory: `/usr/local/sbin/black-fedora/`
- Anonymization: `fedora-anon-setup.sh`
- Watchdog: `ghost-watchdog.sh`
- Ghost session: `ghost-session.sh`
- Log cleanup: `cleanup-logs.sh`

### Systemd Services
- Anonymization: `black-fedora-anon.service`
- Watchdog: `black-fedora-watchdog.service`

### Firewall
- Killswitch rules: `/etc/nftables/black-fedora-killswitch.nft`

### Sandboxing
- Firejail profiles: `/etc/firejail/*.local`

## 🔧 Configuration

### Default Configuration

```json
{
  "vpn_interface": "wg-mullvad",
  "tor_port": "9050",
  "primary_interface": "eth0",
  "installed": true
}
```

### Customization

Edit configuration interactively:
```bash
black-fedora configure
```

Or manually edit:
```bash
nano ~/.config/black-fedora/config.json
```

## 🚨 Troubleshooting

### VPN Won't Start

1. Check WireGuard configuration:
   ```bash
   sudo nmcli connection show
   ```

2. Verify interface name matches config:
   ```bash
   ip link show
   ```

3. Test connection manually:
   ```bash
   sudo nmcli connection up wg-mullvad
   ```

### Tor Won't Start

1. Check Tor service:
   ```bash
   sudo systemctl status tor
   ```

2. View Tor logs:
   ```bash
   sudo journalctl -u tor -f
   ```

3. Restart Tor:
   ```bash
   sudo systemctl restart tor
   ```

### Firewall Blocks Everything

1. Check current rules:
   ```bash
   sudo nft list ruleset
   ```

2. Temporarily disable:
   ```bash
   sudo nft flush ruleset
   ```

3. Verify VPN interface name in killswitch config:
   ```bash
   sudo nano /etc/nftables/black-fedora-killswitch.nft
   ```

### Network Disabled After Watchdog

This is intentional! The watchdog detected VPN failure.

1. Fix VPN connection
2. Re-enable networking:
   ```bash
   sudo nmcli networking on
   ```

## ⚠️ Important Warnings

### Legal Usage Only

Black Fedora is designed for legitimate privacy protection and security research. Users are responsible for complying with all applicable laws and regulations.

### VPN Required

Black Fedora assumes you have a VPN service (Mullvad recommended). Without VPN, the killswitch will block all internet access.

### Aggressive Protection

The watchdog will immediately kill browsers and disable networking if VPN drops. This is intentional for maximum security but may disrupt work.

### Root Access

Many operations require root/sudo access. Only run Black Fedora on systems you control.

## 🎓 Advanced Usage

### Auto-Start Anonymization on Boot

```bash
sudo systemctl enable black-fedora-anon.service
```

### Persistent Watchdog

```bash
sudo systemctl enable black-fedora-watchdog.service
```

### Custom Firejail Profiles

Edit profiles in `/etc/firejail/*.local` and reload:
```bash
firejail --list
```

### Multiple VPN Profiles

Create multiple configs and switch between them:
```bash
sudo nmcli connection up wg-mullvad-us
sudo nmcli connection up wg-mullvad-se
```

## 📊 Status Monitoring

### Quick Status

```bash
black-fedora status
```

### Detailed Status

Run Black Fedora and select option 's' in operational mode for:
- Network interfaces
- Routing table
- Active connections
- Firewall rules
- Service status

### Watch VPN

```bash
watch -n 1 'ip route show default'
```

### Monitor Tor

```bash
sudo journalctl -u tor -f
```

## 🤝 Contributing

Contributions are welcome! Focus areas:
- Additional anonymization techniques
- More browser profiles
- Enhanced monitoring
- Better error handling
- Documentation improvements

## 📜 License

This project is provided as-is for educational and legitimate privacy protection purposes.

## 🙏 Credits

Based on security practices from:
- Whonix Project
- Tails OS
- Qubes OS
- Tor Project
- Privacy Guides

## 📞 Support

For issues, questions, or suggestions:
- Create an issue in the repository
- Check existing documentation
- Review Fedora and Tor documentation

---

**Remember**: Perfect anonymity doesn't exist. Black Fedora significantly improves your privacy and security posture, but no tool is 100% foolproof. Stay informed, stay vigilant, and adapt your practices to your specific threat model.

**Stay Anonymous. Stay Safe. 🎭**
