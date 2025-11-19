#!/usr/bin/env python3
"""
Black Fedora - Demo Mode
Shows the interface without requiring root or actual installation
"""

import time
import sys

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_slow(text, delay=0.03):
    """Print text with typing effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def demo_header():
    print(f"\n{Colors.BOLD}{Colors.RED}╔══════════════════════════════════════════════════════════╗{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}║{Colors.END}  {Colors.BOLD}{Colors.WHITE}BLACK FEDORA{Colors.END} - Advanced OPSEC & Anonymity Manager  {Colors.BOLD}{Colors.RED}║{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}╚══════════════════════════════════════════════════════════╝{Colors.END}\n")

def demo_banner(text, color=Colors.CYAN):
    print(f"\n{color}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{color}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{color}{Colors.BOLD}{'='*60}{Colors.END}\n")

def demo_installation():
    demo_banner("INSTALLATION MODE", Colors.MAGENTA)
    
    print(f"{Colors.YELLOW}This will install Black Fedora components:{Colors.END}\n")
    print(f"  • System packages (tor, firejail, nftables, etc.)")
    print(f"  • Security scripts")
    print(f"  • Systemd services")
    print(f"  • Firewall killswitch")
    print(f"  • Watchdog services\n")
    
    time.sleep(1)
    print(f"{Colors.CYAN}Continue with installation? [y/N]: {Colors.END}y\n")
    time.sleep(0.5)
    
    demo_banner("INSTALLING PACKAGES", Colors.BLUE)
    
    packages = [
        "nftables", "tor", "torsocks", "macchanger", 
        "proxychains-ng", "firejail", "torbrowser-launcher"
    ]
    
    for pkg in packages:
        print(f"{Colors.CYAN}→{Colors.END} Installing {pkg}...", end=" ")
        time.sleep(0.3)
        print(f"{Colors.GREEN}✓{Colors.END}")
    
    time.sleep(0.5)
    demo_banner("CREATING SCRIPTS", Colors.BLUE)
    
    scripts = [
        "anonymization script",
        "VPN watchdog script", 
        "ghost session script",
        "log cleanup script"
    ]
    
    for script in scripts:
        print(f"{Colors.GREEN}✓{Colors.END} Created {script}")
        time.sleep(0.2)
    
    time.sleep(0.5)
    demo_banner("CREATING SYSTEMD SERVICES", Colors.BLUE)
    print(f"{Colors.GREEN}✓{Colors.END} Created systemd services")
    time.sleep(0.3)
    
    demo_banner("CONFIGURING FIREWALL", Colors.BLUE)
    print(f"{Colors.GREEN}✓{Colors.END} Created nftables killswitch")
    time.sleep(0.3)
    
    demo_banner("CREATING FIREJAIL PROFILES", Colors.BLUE)
    print(f"{Colors.GREEN}✓{Colors.END} Created firejail profiles")
    time.sleep(0.3)
    
    demo_banner("CONFIGURING LOGGING", Colors.BLUE)
    print(f"{Colors.GREEN}✓{Colors.END} Configured journald")
    time.sleep(0.5)
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Installation completed successfully!{Colors.END}\n")
    print(f"{Colors.YELLOW}Next steps:{Colors.END}")
    print(f"  1. Import your Mullvad WireGuard config")
    print(f"  2. Run: black-fedora configure")
    print(f"  3. Start using: black-fedora start\n")
    time.sleep(2)

def demo_operational():
    demo_banner("OPERATIONAL MODE", Colors.CYAN)
    
    print(f"\n{Colors.BOLD}Current Status:{Colors.END}\n")
    print(f"  VPN Active:        {Colors.GREEN}✓{Colors.END}")
    print(f"  Tor Running:       {Colors.GREEN}✓{Colors.END}")
    print(f"  Firewall Active:   {Colors.GREEN}✓{Colors.END}")
    print(f"  Watchdog Running:  {Colors.GREEN}✓{Colors.END}")
    
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
    
    time.sleep(1)
    print(f"{Colors.CYAN}Select option: {Colors.END}1\n")
    time.sleep(0.5)
    
    demo_banner("PRE-FLIGHT CHECKS", Colors.YELLOW)
    
    checks = [
        ("VPN Connection", True),
        ("Tor Service", True),
        ("Firewall Active", True),
        ("Watchdog Running", True),
    ]
    
    for name, status in checks:
        icon = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
        print(f"  {icon} {name}")
        time.sleep(0.3)
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}All checks passed! System ready for ghost mode.{Colors.END}")
    time.sleep(2)
    
    print(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
    time.sleep(1)
    
    print(f"\n{Colors.CYAN}Select option: {Colors.END}7\n")
    time.sleep(0.5)
    
    demo_banner("LAUNCHING GHOST SESSION", Colors.GREEN)
    
    print(f"{Colors.CYAN}→{Colors.END} Verifying VPN connection...", end=" ")
    time.sleep(0.5)
    print(f"{Colors.GREEN}✓{Colors.END}")
    
    print(f"{Colors.CYAN}→{Colors.END} Checking Tor service...", end=" ")
    time.sleep(0.5)
    print(f"{Colors.GREEN}✓{Colors.END}")
    
    print(f"{Colors.CYAN}→{Colors.END} Validating default route...", end=" ")
    time.sleep(0.5)
    print(f"{Colors.GREEN}✓{Colors.END}")
    
    print(f"{Colors.CYAN}→{Colors.END} Launching sandboxed Tor Browser...", end=" ")
    time.sleep(0.8)
    print(f"{Colors.GREEN}✓{Colors.END}")
    
    print(f"\n{Colors.GREEN}✓ Ghost session launched{Colors.END}")
    time.sleep(2)

def demo_status():
    demo_banner("DETAILED STATUS", Colors.CYAN)
    
    print(f"{Colors.BOLD}Network Interfaces:{Colors.END}")
    print(f"  2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP>")
    print(f"      inet 192.168.1.100/24")
    print(f"  3: wg-mullvad: <POINTOPOINT,UP,LOWER_UP>")
    print(f"      inet 10.8.0.2/32")
    
    print(f"\n{Colors.BOLD}Default Route:{Colors.END}")
    print(f"  default via 10.8.0.1 dev wg-mullvad")
    
    print(f"\n{Colors.BOLD}Active Network Connections:{Colors.END}")
    print(f"  LISTEN    0    128    127.0.0.1:9050    0.0.0.0:*")
    print(f"  LISTEN    0    128    127.0.0.1:9051    0.0.0.0:*")
    
    print(f"\n{Colors.BOLD}Firewall Rules:{Colors.END}")
    print(f"  table inet filter {{")
    print(f"    chain output {{")
    print(f"      type filter hook output priority 0; policy drop;")
    print(f"      oif \"wg-mullvad\" accept")
    print(f"      ct state established,related accept")
    print(f"    }}")
    print(f"  }}")
    
    time.sleep(3)

def demo_osint():
    demo_banner("OSINT SECURITY CHECKS", Colors.MAGENTA)
    
    print(f"{Colors.BOLD}Run these checks in your browser:{Colors.END}\n")
    print(f"  1. DNS Leak Test:     {Colors.CYAN}https://dnsleaktest.com{Colors.END}")
    print(f"  2. Browser Leaks:     {Colors.CYAN}https://browserleaks.com{Colors.END}")
    print(f"  3. Panopticlick:      {Colors.CYAN}https://panopticlick.eff.org{Colors.END}")
    print(f"  4. Cover Your Tracks: {Colors.CYAN}https://coveryourtracks.eff.org{Colors.END}")
    print(f"  5. IP Check:          {Colors.CYAN}https://check.torproject.org{Colors.END}\n")
    
    time.sleep(3)

def main():
    """Run demo sequence"""
    demo_header()
    
    print(f"{Colors.YELLOW}BLACK FEDORA - DEMO MODE{Colors.END}\n")
    print("This demo showcases the Black Fedora interface.\n")
    time.sleep(2)
    
    # Show installation
    input(f"{Colors.CYAN}Press Enter to see installation process...{Colors.END}")
    demo_installation()
    
    # Show operational mode
    input(f"{Colors.CYAN}Press Enter to see operational mode...{Colors.END}")
    demo_operational()
    
    # Show detailed status
    input(f"{Colors.CYAN}Press Enter to see detailed status...{Colors.END}")
    demo_status()
    
    # Show OSINT checks
    input(f"{Colors.CYAN}Press Enter to see OSINT checks...{Colors.END}")
    demo_osint()
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Demo complete!{Colors.END}\n")
    print(f"To install Black Fedora for real:")
    print(f"  1. Run: {Colors.CYAN}sudo ./install.sh{Colors.END}")
    print(f"  2. Then: {Colors.CYAN}sudo black-fedora install{Colors.END}\n")
    print(f"{Colors.YELLOW}Stay anonymous. Stay safe. 🎭{Colors.END}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo interrupted.{Colors.END}\n")
        sys.exit(0)
