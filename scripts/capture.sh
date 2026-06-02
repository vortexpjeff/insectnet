#!/usr/bin/env bash
# capture.sh — Quick-start an InsectNet capture session on BirdNET-Pi
#
# Usage:
#   ./scripts/capture.sh                        # Default session
#   ./scripts/capture.sh --threshold 0.5        # Higher threshold
#   ./scripts/capture.sh --show                 # Verbose per-WAV scores
#   ./scripts/capture.sh --model 3class.joblib  # Use 3-class model

set -euo pipefail

PI_HOST="${PI_HOST:-birdnetpi@192.168.1.223}"
THRESHOLD="${THRESHOLD:-0.3}"
SHOW="${SHOW:-true}"
MODEL="${MODEL:-classifier.joblib}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --show) SHOW=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --host) PI_HOST="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SHOW_FLAG=""
[ "$SHOW" = true ] && SHOW_FLAG="--show"

echo "=== InsectNet Capture Session ==="
echo "  Host:     ${PI_HOST}"
echo "  Model:    ${MODEL}"
echo "  Threshold: ${THRESHOLD}"
echo ""

# Verify Pi is reachable
ssh "${PI_HOST}" "echo '  ✓ Pi reachable'" || { echo "✗ Cannot reach ${PI_HOST}"; exit 1; }

# Verify classifier exists on Pi
ssh "${PI_HOST}" "test -f ~/insectnet_capture/${MODEL}" && \
    echo "  ✓ Classifier present" || \
    { echo "⚠ Classifier not found. Run deploy.sh first."; }

# Verify BirdNET-Pi services
echo ""
echo "Checking BirdNET-Pi services..."
ssh "${PI_HOST}" "systemctl is-active birdnet_recording birdnet_analysis birdnet_log birdnet_stats" || true

echo ""
echo "Starting capture session..."
echo "  Ctrl+C to stop and pull results"
echo ""

# Start sidecar via SSH with setsid (survives SSH disconnect)
ssh "${PI_HOST}" \
    "setsid ~/birdnet/bin/python3 ~/insectnet_capture/insectnet_capture.py \
        --threshold ${THRESHOLD} ${SHOW_FLAG} \
        > ~/insectnet_capture/sidecar.log 2>&1 &"

echo "Sidecar started. Check logs:"
echo "  ssh ${PI_HOST} 'tail -f ~/insectnet_capture/sidecar.log'"
echo ""
echo "To pull captures later:"
echo "  python -m insectnet.capture --pull"
