#!/usr/bin/env bash
# Install udev rule for the Waveshare RP2040-Zero 3-button serial footswitch.
#
# Creates a stable symlink at /dev/serial-footswitch and grants rw access
# so no dialout membership is needed.
#
# Usage:
#   bash scripts/setup/install_serial_footswitch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RULE_SRC="$REPO_ROOT/hardware/99-serial-footswitch.rules"
RULE_DST="/etc/udev/rules.d/99-serial-footswitch.rules"

if [[ ! -f "$RULE_SRC" ]]; then
    echo "Error: rule file not found at $RULE_SRC" >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    echo "This script needs root. Re-running with sudo..."
    exec sudo bash "$0" "$@"
fi

cp "$RULE_SRC" "$RULE_DST"
echo "Installed: $RULE_DST"

udevadm control --reload-rules
udevadm trigger --subsystem-match=tty
echo "Udev rules reloaded."

# Check if the device is already plugged in
if ls /dev/serial-footswitch &>/dev/null; then
    echo "Device found: $(readlink -f /dev/serial-footswitch)"
else
    echo ""
    echo "Device not detected yet. Unplug and replug the RP2040-Zero, then verify:"
    echo "  ls -la /dev/serial-footswitch"
fi

echo ""
echo "Done. The serial footswitch is ready to use with type: serial in fello_config.yaml."
