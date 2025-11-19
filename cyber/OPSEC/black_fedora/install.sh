#!/usr/bin/env bash
set -e

echo "=========================================="
echo "  Black Fedora Installation Script"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: Please run as root (use sudo)"
    exit 1
fi

# Check if Fedora
if [ ! -f /etc/fedora-release ]; then
    echo "WARNING: This script is designed for Fedora Linux"
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Installing Black Fedora CLI tool..."

# Install Python if not present
if ! command -v python3 &> /dev/null; then
    echo "Installing Python 3..."
    dnf install -y python3
fi

# Copy the main script
echo "Installing black-fedora command..."
cp black-fedora.py /usr/local/bin/black-fedora
chmod +x /usr/local/bin/black-fedora

# Create necessary directories
mkdir -p /usr/local/sbin/black-fedora
mkdir -p /etc/nftables
mkdir -p ~/.config/black-fedora

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: black-fedora install"
echo "  2. Import your Mullvad WireGuard config"
echo "  3. Run: black-fedora configure"
echo "  4. Start using: black-fedora start"
echo ""
echo "Or simply run: black-fedora"
echo ""
