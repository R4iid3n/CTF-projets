#!/usr/bin/env python3
"""
Black Fedora - Advanced OPSEC & Anonymity CLI Manager
A comprehensive security hardening and anonymization toolkit for Fedora Linux
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# Configuration
CONFIG_DIR = Path.home() / ".config" / "black-fedora"
CONFIG_FILE = CONFIG_DIR / "config.json"
SCRIPTS_DIR = "/usr/local/sbin/black-fedora"
STATE_FILE = CONFIG_DIR / "state.json"

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class BlackFedora:
    def __init__(self):
        self.config = self.load_config()
        self.state = self.load_state()
        
    def load_config(self) -> dict:
        """Load configuration from file"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {
            "vpn_interface": "wg-mullvad",
            "tor_port": "9050",
            "primary_interface": self.get_primary_interface(),
            "installed": False
        }
    
    def save_config(self):
        """Save configuration to file"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_state(self) -> dict:
        """Load operational state"""
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {
            "mac_randomized": False,
            "hostname_changed": False,
            "services_disabled": False,
            "firewall_active": False,
            "vpn_active": False,
            "tor_active": False
        }
    
    def save_state(self):
        """Save operational state"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    @staticmethod
    def get_primary_interface() -> str:
        """Get the primary network interface"""
        try:
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True, check=True
            )
            for line in result.stdout.split('\n'):
                if 'default' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        return parts[4]
        except:
            pass
        return "eth0"
    
    def print_header(self):
        """Print application header"""
        print(f"\n{Colors.BOLD}{Colors.RED}╔══════════════════════════════════════════════════════════╗{Colors.END}")
        print(f"{Colors.BOLD}{Colors.RED}║{Colors.END}  {Colors.BOLD}{Colors.WHITE}BLACK FEDORA{Colors.END} - Advanced OPSEC & Anonymity Manager  {Colors.BOLD}{Colors.RED}║{Colors.END}")
        print(f"{Colors.BOLD}{Colors.RED}╚══════════════════════════════════════════════════════════╝{Colors.END}\n")
    
    def print_banner(self, text: str, color=Colors.CYAN):
        """Print a section banner"""
        print(f"\n{color}{Colors.BOLD}{'='*60}{Colors.END}")
        print(f"{color}{Colors.BOLD}{text.center(60)}{Colors.END}")
        print(f"{color}{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    def run_command(self, cmd: List[str], description: str = "", check=True, sudo=False) -> Tuple[bool, str]:
        """Run a shell command with optional sudo"""
        if sudo and os.geteuid() != 0:
            cmd = ["sudo"] + cmd
        
        if description:
            print(f"{Colors.CYAN}→{Colors.END} {description}...", end=" ")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check
            )
            if description:
                print(f"{Colors.GREEN}✓{Colors.END}")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            if description:
                print(f"{Colors.RED}✗{Colors.END}")
            return False, e.stderr
    
    def check_root(self):
        """Check if running as root"""
        if os.geteuid() != 0:
            print(f"{Colors.RED}✗ This operation requires root privileges. Please run with sudo.{Colors.END}")
            sys.exit(1)
    
    def installation_menu(self):
        """Show installation menu and perform installation"""
        self.print_banner("INSTALLATION MODE", Colors.MAGENTA)
        
        print(f"{Colors.YELLOW}This will install Black Fedora components:{Colors.END}\n")
        print(f"  • System packages (tor, firejail, nftables, etc.)")
        print(f"  • Security scripts")
        print(f"  • Systemd services")
        print(f"  • Firewall killswitch")
        print(f"  • Watchdog services\n")
        
        response = input(f"{Colors.CYAN}Continue with installation? [y/N]: {Colors.END}").lower()
        if response != 'y':
            print(f"{Colors.YELLOW}Installation cancelled.{Colors.END}")
            return
        
        self.check_root()
        self.perform_installation()
    
    def perform_installation(self):
        """Perform the actual installation"""
        self.print_banner("INSTALLING PACKAGES", Colors.BLUE)
        
        # Install packages
        packages = [
            "nftables", "tor", "torsocks", "macchanger", "proxychains-ng",
            "firejail", "firetools", "torbrowser-launcher", "wireguard-tools",
            "NetworkManager-tui", "bind-utils", "tcpdump"
        ]
        
        self.run_command(
            ["dnf", "install", "-y"] + packages,
            "Installing security packages",
            sudo=True
        )
        
        # Create scripts directory
        self.print_banner("CREATING SCRIPTS", Colors.BLUE)
        Path(SCRIPTS_DIR).mkdir(parents=True, exist_ok=True)
        
        # Create anonymization script
        self.create_anon_script()
        
        # Create watchdog script
        self.create_watchdog_script()
        
        # Create ghost session script
        self.create_ghost_session_script()
        
        # Create log cleanup script
        self.create_log_cleanup_script()
        
        # Create systemd services
        self.print_banner("CREATING SYSTEMD SERVICES", Colors.BLUE)
        self.create_systemd_services()
        
        # Create nftables killswitch
        self.print_banner("CONFIGURING FIREWALL", Colors.BLUE)
        self.create_nftables_killswitch()
        
        # Create firejail profiles
        self.print_banner("CREATING FIREJAIL PROFILES", Colors.BLUE)
        self.create_firejail_profiles()
        
        # Configure journald
        self.print_banner("CONFIGURING LOGGING", Colors.BLUE)
        self.configure_journald()
        
        # Mark as installed
        self.config["installed"] = True
        self.save_config()
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Installation completed successfully!{Colors.END}\n")
        print(f"{Colors.YELLOW}Next steps:{Colors.END}")
        print(f"  1. Import your Mullvad WireGuard config")
        print(f"  2. Run: black-fedora configure")
        print(f"  3. Start using: black-fedora start\n")
    
    def create_anon_script(self):
        """Create the main anonymization script"""
        script_path = f"{SCRIPTS_DIR}/fedora-anon-setup.sh"
        script_content = f'''#!/usr/bin/env bash
set -e

# Black Fedora Anonymization Script
IFACE=$(ip route | awk '/default/ {{print $5; exit}}')

# Randomize MAC address
ip link set "$IFACE" down || true
macchanger -r "$IFACE" 2>/dev/null || true
ip link set "$IFACE" up || true

# Change hostname
NEW_HOST="host-$(tr -dc 'a-z0-9' </dev/urandom | head -c8)"
hostnamectl set-hostname "$NEW_HOST"

# Disable telemetry services
systemctl disable --now avahi-daemon 2>/dev/null || true
systemctl disable --now cups 2>/dev/null || true
systemctl disable --now bluetooth 2>/dev/null || true
systemctl disable --now geoclue 2>/dev/null || true

echo "Anonymization setup complete - MAC randomized, hostname: $NEW_HOST"
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"{Colors.GREEN}✓{Colors.END} Created anonymization script")
    
    def create_watchdog_script(self):
        """Create the VPN watchdog script"""
        script_path = f"{SCRIPTS_DIR}/ghost-watchdog.sh"
        vpn_iface = self.config.get("vpn_interface", "wg-mullvad")
        
        script_content = f'''#!/usr/bin/env bash

# Black Fedora VPN Watchdog
VPN_IFACE="{vpn_iface}"

while true; do
    # Check if VPN interface is up
    if ! ip link show "$VPN_IFACE" up &>/dev/null; then
        echo "VPN interface down - killing browsers and disabling network"
        pkill -9 torbrowser firefox librewolf chromium 2>/dev/null || true
        nmcli networking off
        exit 1
    fi
    
    # Check if default route goes through VPN
    DEF=$(ip route show default | awk '/default/ {{print $5}}')
    if [[ "$DEF" != "$VPN_IFACE" ]]; then
        echo "Default route not through VPN - killing browsers and disabling network"
        pkill -9 torbrowser firefox librewolf chromium 2>/dev/null || true
        nmcli networking off
        exit 1
    fi
    
    sleep 5
done
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"{Colors.GREEN}✓{Colors.END} Created VPN watchdog script")
    
    def create_ghost_session_script(self):
        """Create the ghost session launcher"""
        script_path = f"{SCRIPTS_DIR}/ghost-session.sh"
        vpn_iface = self.config.get("vpn_interface", "wg-mullvad")
        tor_port = self.config.get("tor_port", "9050")
        
        script_content = f'''#!/usr/bin/env bash

# Black Fedora Ghost Session Launcher
VPN_IFACE="{vpn_iface}"
TOR_PORT="{tor_port}"

# Check VPN is up
if ! ip link show "$VPN_IFACE" up &>/dev/null; then
    echo "ERROR: VPN interface not up"
    nmcli networking off
    exit 1
fi

# Check default route through VPN
DEF=$(ip route show default | awk '/default/ {{print $5}}')
if [[ "$DEF" != "$VPN_IFACE" ]]; then
    echo "ERROR: Default route not through VPN"
    nmcli networking off
    exit 1
fi

# Check Tor is running
if ! ss -lnt | grep -q ":{tor_port}"; then
    echo "ERROR: Tor is not running"
    nmcli networking off
    exit 1
fi

echo "All checks passed - launching ghost session"

# Launch Tor Browser in firejail
firejail --profile=/etc/firejail/torbrowser-launcher.local torbrowser-launcher &

echo "Ghost session active"
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"{Colors.GREEN}✓{Colors.END} Created ghost session script")
    
    def create_log_cleanup_script(self):
        """Create log cleanup script"""
        script_path = f"{SCRIPTS_DIR}/cleanup-logs.sh"
        
        script_content = '''#!/usr/bin/env bash
set -e

# Black Fedora Log Cleanup Script
journalctl --vacuum-size=50M --vacuum-time=3days

# Clear system logs
: > /var/log/wtmp
: > /var/log/btmp

# Clear other logs
for f in messages secure maillog cron dnf.log; do
    if [ -f "/var/log/$f" ]; then
        : > "/var/log/$f"
    fi
done

echo "Log cleanup complete"
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"{Colors.GREEN}✓{Colors.END} Created log cleanup script")
    
    def create_systemd_services(self):
        """Create systemd service files"""
        # Anonymization service
        anon_service = '''[Unit]
Description=Black Fedora Anonymization Setup
Before=network-pre.target
Wants=network-pre.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/black-fedora/fedora-anon-setup.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
'''
        
        with open('/etc/systemd/system/black-fedora-anon.service', 'w') as f:
            f.write(anon_service)
        
        # Watchdog service
        watchdog_service = '''[Unit]
Description=Black Fedora VPN Watchdog
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/sbin/black-fedora/ghost-watchdog.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
        
        with open('/etc/systemd/system/black-fedora-watchdog.service', 'w') as f:
            f.write(watchdog_service)
        
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        print(f"{Colors.GREEN}✓{Colors.END} Created systemd services")
    
    def create_nftables_killswitch(self):
        """Create nftables killswitch configuration"""
        vpn_iface = self.config.get("vpn_interface", "wg-mullvad")
        
        nftables_config = f'''#!/usr/sbin/nft -f

# Black Fedora Killswitch
flush ruleset

table inet filter {{
    chain input {{
        type filter hook input priority 0; policy drop;
        
        # Allow established connections
        ct state established,related accept
        
        # Allow loopback
        iif "lo" accept
        
        # Allow ICMP
        ip protocol icmp accept
        ip6 nexthdr icmpv6 accept
        
        # Allow SSH from local network (adjust as needed)
        tcp dport 22 ip saddr 192.168.0.0/16 accept
    }}
    
    chain output {{
        type filter hook output priority 0; policy drop;
        
        # Allow VPN traffic
        oif "{vpn_iface}" accept
        
        # Allow Tor (if using tun0)
        oif "tun0" accept
        
        # Allow established connections
        ct state established,related accept
        
        # Allow loopback
        oif "lo" accept
    }}
    
    chain forward {{
        type filter hook forward priority 0; policy drop;
    }}
}}
'''
        
        with open('/etc/nftables/black-fedora-killswitch.nft', 'w') as f:
            f.write(nftables_config)
        
        print(f"{Colors.GREEN}✓{Colors.END} Created nftables killswitch")
    
    def create_firejail_profiles(self):
        """Create firejail override profiles"""
        # LibreWolf profile
        librewolf_profile = '''# Black Fedora LibreWolf Profile
include /etc/firejail/librewolf.profile

private
private-tmp
private-cache
nonewprivs
nodbus
nosound
notv
novideo
noexec /tmp
noexec /dev/shm
blacklist /media
blacklist /mnt
blacklist /run/media
'''
        
        Path('/etc/firejail').mkdir(parents=True, exist_ok=True)
        with open('/etc/firejail/librewolf.local', 'w') as f:
            f.write(librewolf_profile)
        
        # Tor Browser profile
        torbrowser_profile = '''# Black Fedora Tor Browser Profile
include /etc/firejail/torbrowser-launcher.profile

private
private-tmp
private-cache
nodbus
notv
novideo
noexec /tmp
noexec /dev/shm
blacklist ${HOME}/Documents
blacklist ${HOME}/Downloads
blacklist ${HOME}/Pictures
blacklist ${HOME}/Videos
'''
        
        with open('/etc/firejail/torbrowser-launcher.local', 'w') as f:
            f.write(torbrowser_profile)
        
        print(f"{Colors.GREEN}✓{Colors.END} Created firejail profiles")
    
    def configure_journald(self):
        """Configure journald for minimal logging"""
        journald_config = '''[Journal]
Storage=persistent
SystemMaxUse=100M
RuntimeMaxUse=50M
MaxRetentionSec=3day
MaxFileSec=1day
Compress=yes
ForwardToSyslog=no
'''
        
        config_dir = Path('/etc/systemd/journald.conf.d')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_dir / 'black-fedora.conf', 'w') as f:
            f.write(journald_config)
        
        print(f"{Colors.GREEN}✓{Colors.END} Configured journald")
    
    def operational_menu(self):
        """Show operational menu"""
        while True:
            self.print_banner("OPERATIONAL MODE", Colors.CYAN)
            self.show_status()
            
            print(f"\n{Colors.BOLD}Available Commands:{Colors.END}\n")
            print(f"  {Colors.GREEN}1{Colors.END}. Pre-flight checks")
            print(f"  {Colors.GREEN}2{Colors.END}. Randomize MAC address")
            print(f"  {Colors.GREEN}3{Colors.END}. Change hostname")
            print(f"  {Colors.GREEN}4{Colors.END}. Disable telemetry services")
            print(f"  {Colors.GREEN}5{Colors.END}. Enable firewall killswitch")
            print(f"  {Colors.GREEN}6{Colors.END}. Start VPN watchdog")
            print(f"  {Colors.GREEN}7{Colors.END}. Launch ghost session")
            print(f"  {Colors.GREEN}8{Colors.END}. Cleanup logs")
            print(f"  {Colors.GREEN}9{Colors.END}. Run OSINT checks")
            print(f"  {Colors.YELLOW}s{Colors.END}. Show full status")
            print(f"  {Colors.RED}q{Colors.END}. Quit\n")
            
            choice = input(f"{Colors.CYAN}Select option: {Colors.END}").lower()
            
            if choice == '1':
                self.pre_flight_checks()
            elif choice == '2':
                self.randomize_mac()
            elif choice == '3':
                self.change_hostname()
            elif choice == '4':
                self.disable_services()
            elif choice == '5':
                self.enable_firewall()
            elif choice == '6':
                self.start_watchdog()
            elif choice == '7':
                self.launch_ghost_session()
            elif choice == '8':
                self.cleanup_logs()
            elif choice == '9':
                self.osint_checks()
            elif choice == 's':
                self.show_detailed_status()
            elif choice == 'q':
                print(f"\n{Colors.YELLOW}Stay anonymous. Stay safe.{Colors.END}\n")
                break
            else:
                print(f"{Colors.RED}Invalid option{Colors.END}")
            
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
    
    def show_status(self):
        """Show current system status"""
        print(f"\n{Colors.BOLD}Current Status:{Colors.END}\n")
        
        # Check VPN
        vpn_status = self.check_vpn_status()
        vpn_icon = f"{Colors.GREEN}✓{Colors.END}" if vpn_status else f"{Colors.RED}✗{Colors.END}"
        print(f"  VPN Active:        {vpn_icon}")
        
        # Check Tor
        tor_status = self.check_tor_status()
        tor_icon = f"{Colors.GREEN}✓{Colors.END}" if tor_status else f"{Colors.RED}✗{Colors.END}"
        print(f"  Tor Running:       {tor_icon}")
        
        # Check Firewall
        fw_status = self.check_firewall_status()
        fw_icon = f"{Colors.GREEN}✓{Colors.END}" if fw_status else f"{Colors.RED}✗{Colors.END}"
        print(f"  Firewall Active:   {fw_icon}")
        
        # Check Watchdog
        wd_status = self.check_watchdog_status()
        wd_icon = f"{Colors.GREEN}✓{Colors.END}" if wd_status else f"{Colors.RED}✗{Colors.END}"
        print(f"  Watchdog Running:  {wd_icon}")
    
    def check_vpn_status(self) -> bool:
        """Check if VPN is active"""
        vpn_iface = self.config.get("vpn_interface", "wg-mullvad")
        success, _ = self.run_command(
            ["ip", "link", "show", vpn_iface],
            check=False
        )
        return success
    
    def check_tor_status(self) -> bool:
        """Check if Tor is running"""
        success, output = self.run_command(
            ["ss", "-lnt"],
            check=False
        )
        return success and f":{self.config.get('tor_port', '9050')}" in output
    
    def check_firewall_status(self) -> bool:
        """Check if nftables is active"""
        success, output = self.run_command(
            ["nft", "list", "ruleset"],
            check=False,
            sudo=True
        )
        return success and "inet filter" in output
    
    def check_watchdog_status(self) -> bool:
        """Check if watchdog service is running"""
        success, output = self.run_command(
            ["systemctl", "is-active", "black-fedora-watchdog"],
            check=False
        )
        return success and "active" in output
    
    def pre_flight_checks(self):
        """Perform pre-flight security checks"""
        self.print_banner("PRE-FLIGHT CHECKS", Colors.YELLOW)
        
        checks = [
            ("VPN Connection", self.check_vpn_status),
            ("Tor Service", self.check_tor_status),
            ("Firewall Active", self.check_firewall_status),
            ("Watchdog Running", self.check_watchdog_status),
        ]
        
        all_passed = True
        for name, check_func in checks:
            status = check_func()
            icon = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
            print(f"  {icon} {name}")
            if not status:
                all_passed = False
        
        if all_passed:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All checks passed! System ready for ghost mode.{Colors.END}")
        else:
            print(f"\n{Colors.YELLOW}⚠ Some checks failed. Review and fix before proceeding.{Colors.END}")
    
    def randomize_mac(self):
        """Randomize MAC address"""
        self.check_root()
        self.print_banner("RANDOMIZING MAC ADDRESS", Colors.BLUE)
        
        iface = self.config.get("primary_interface")
        self.run_command(["ip", "link", "set", iface, "down"], f"Bringing down {iface}", sudo=True)
        self.run_command(["macchanger", "-r", iface], f"Randomizing MAC on {iface}", sudo=True)
        self.run_command(["ip", "link", "set", iface, "up"], f"Bringing up {iface}", sudo=True)
        
        self.state["mac_randomized"] = True
        self.save_state()
    
    def change_hostname(self):
        """Change system hostname"""
        self.check_root()
        self.print_banner("CHANGING HOSTNAME", Colors.BLUE)
        
        new_hostname = f"host-{os.urandom(4).hex()}"
        self.run_command(
            ["hostnamectl", "set-hostname", new_hostname],
            f"Setting hostname to {new_hostname}",
            sudo=True
        )
        
        self.state["hostname_changed"] = True
        self.save_state()
    
    def disable_services(self):
        """Disable telemetry services"""
        self.check_root()
        self.print_banner("DISABLING TELEMETRY SERVICES", Colors.BLUE)
        
        services = ["avahi-daemon", "cups", "bluetooth", "geoclue"]
        for service in services:
            self.run_command(
                ["systemctl", "disable", "--now", service],
                f"Disabling {service}",
                check=False,
                sudo=True
            )
        
        self.state["services_disabled"] = True
        self.save_state()
    
    def enable_firewall(self):
        """Enable nftables killswitch"""
        self.check_root()
        self.print_banner("ENABLING FIREWALL KILLSWITCH", Colors.BLUE)
        
        self.run_command(
            ["nft", "-f", "/etc/nftables/black-fedora-killswitch.nft"],
            "Loading killswitch rules",
            sudo=True
        )
        
        self.run_command(
            ["systemctl", "enable", "nftables"],
            "Enabling nftables service",
            sudo=True
        )
        
        self.state["firewall_active"] = True
        self.save_state()
    
    def start_watchdog(self):
        """Start VPN watchdog service"""
        self.check_root()
        self.print_banner("STARTING VPN WATCHDOG", Colors.BLUE)
        
        self.run_command(
            ["systemctl", "enable", "--now", "black-fedora-watchdog"],
            "Enabling watchdog service",
            sudo=True
        )
    
    def launch_ghost_session(self):
        """Launch ghost session"""
        self.print_banner("LAUNCHING GHOST SESSION", Colors.GREEN)
        
        # Run pre-flight checks first
        if not self.check_vpn_status():
            print(f"{Colors.RED}✗ VPN is not active. Cannot launch ghost session.{Colors.END}")
            return
        
        if not self.check_tor_status():
            print(f"{Colors.YELLOW}⚠ Tor is not running. Starting Tor...{Colors.END}")
            self.run_command(["systemctl", "start", "tor"], "Starting Tor", sudo=True)
        
        # Launch ghost session script
        subprocess.Popen([f"{SCRIPTS_DIR}/ghost-session.sh"])
        print(f"{Colors.GREEN}✓ Ghost session launched{Colors.END}")
    
    def cleanup_logs(self):
        """Cleanup system logs"""
        self.check_root()
        self.print_banner("CLEANING UP LOGS", Colors.BLUE)
        
        self.run_command(
            [f"{SCRIPTS_DIR}/cleanup-logs.sh"],
            "Running log cleanup",
            sudo=True
        )
    
    def osint_checks(self):
        """Show OSINT check URLs"""
        self.print_banner("OSINT SECURITY CHECKS", Colors.MAGENTA)
        
        print(f"{Colors.BOLD}Run these checks in your browser:{Colors.END}\n")
        print(f"  1. DNS Leak Test:     {Colors.CYAN}https://dnsleaktest.com{Colors.END}")
        print(f"  2. Browser Leaks:     {Colors.CYAN}https://browserleaks.com{Colors.END}")
        print(f"  3. Panopticlick:      {Colors.CYAN}https://panopticlick.eff.org{Colors.END}")
        print(f"  4. Cover Your Tracks: {Colors.CYAN}https://coveryourtracks.eff.org{Colors.END}")
        print(f"  5. IP Check:          {Colors.CYAN}https://check.torproject.org{Colors.END}\n")
    
    def show_detailed_status(self):
        """Show detailed system status"""
        self.print_banner("DETAILED STATUS", Colors.CYAN)
        
        # Network interfaces
        print(f"{Colors.BOLD}Network Interfaces:{Colors.END}")
        success, output = self.run_command(["ip", "addr", "show"], check=False)
        if success:
            for line in output.split('\n'):
                if 'inet ' in line or ': <' in line:
                    print(f"  {line.strip()}")
        
        # Routing
        print(f"\n{Colors.BOLD}Default Route:{Colors.END}")
        success, output = self.run_command(["ip", "route", "show", "default"], check=False)
        if success:
            print(f"  {output.strip()}")
        
        # Active connections
        print(f"\n{Colors.BOLD}Active Network Connections:{Colors.END}")
        success, output = self.run_command(["ss", "-tuln"], check=False)
        if success:
            lines = output.split('\n')[:10]  # Show first 10 lines
            for line in lines:
                print(f"  {line}")
        
        # Firewall rules
        print(f"\n{Colors.BOLD}Firewall Rules:{Colors.END}")
        success, output = self.run_command(["nft", "list", "ruleset"], check=False, sudo=True)
        if success and output:
            lines = output.split('\n')[:20]  # Show first 20 lines
            for line in lines:
                print(f"  {line}")
        else:
            print(f"  {Colors.YELLOW}No active firewall rules{Colors.END}")
    
    def configure_mode(self):
        """Interactive configuration mode"""
        self.print_banner("CONFIGURATION MODE", Colors.YELLOW)
        
        print(f"{Colors.BOLD}Current Configuration:{Colors.END}\n")
        print(f"  VPN Interface: {Colors.CYAN}{self.config.get('vpn_interface')}{Colors.END}")
        print(f"  Tor Port:      {Colors.CYAN}{self.config.get('tor_port')}{Colors.END}")
        print(f"  Primary Iface: {Colors.CYAN}{self.config.get('primary_interface')}{Colors.END}\n")
        
        response = input(f"Update configuration? [y/N]: ").lower()
        if response != 'y':
            return
        
        # VPN Interface
        vpn_iface = input(f"VPN Interface [{self.config.get('vpn_interface')}]: ").strip()
        if vpn_iface:
            self.config["vpn_interface"] = vpn_iface
        
        # Tor Port
        tor_port = input(f"Tor Port [{self.config.get('tor_port')}]: ").strip()
        if tor_port:
            self.config["tor_port"] = tor_port
        
        # Primary Interface
        primary_iface = input(f"Primary Interface [{self.config.get('primary_interface')}]: ").strip()
        if primary_iface:
            self.config["primary_interface"] = primary_iface
        
        self.save_config()
        print(f"\n{Colors.GREEN}✓ Configuration saved{Colors.END}")
    
    def run(self):
        """Main application loop"""
        self.print_header()
        
        # Check if installed
        if not self.config.get("installed", False):
            print(f"{Colors.YELLOW}Black Fedora is not installed yet.{Colors.END}\n")
            self.installation_menu()
        else:
            # Show main menu
            while True:
                print(f"\n{Colors.BOLD}Main Menu:{Colors.END}\n")
                print(f"  {Colors.GREEN}1{Colors.END}. Operational Mode")
                print(f"  {Colors.GREEN}2{Colors.END}. Configuration")
                print(f"  {Colors.GREEN}3{Colors.END}. Reinstall")
                print(f"  {Colors.RED}q{Colors.END}. Quit\n")
                
                choice = input(f"{Colors.CYAN}Select option: {Colors.END}").lower()
                
                if choice == '1':
                    self.operational_menu()
                elif choice == '2':
                    self.configure_mode()
                elif choice == '3':
                    self.config["installed"] = False
                    self.save_config()
                    self.installation_menu()
                elif choice == 'q':
                    print(f"\n{Colors.YELLOW}Stay anonymous. Stay safe.{Colors.END}\n")
                    break
                else:
                    print(f"{Colors.RED}Invalid option{Colors.END}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Black Fedora - Advanced OPSEC & Anonymity Manager')
    parser.add_argument('command', nargs='?', choices=['install', 'configure', 'start', 'status'],
                       help='Command to execute')
    
    args = parser.parse_args()
    
    bf = BlackFedora()
    
    if args.command == 'install':
        bf.installation_menu()
    elif args.command == 'configure':
        bf.configure_mode()
    elif args.command == 'start':
        bf.operational_menu()
    elif args.command == 'status':
        bf.show_status()
    else:
        bf.run()

if __name__ == "__main__":
    main()
