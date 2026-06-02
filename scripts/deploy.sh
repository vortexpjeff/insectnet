#!/usr/bin/env bash
# deploy.sh — Deploy InsectNet sidecar + classifier to a BirdNET-Pi
#
# Usage:
#   ./scripts/deploy.sh                        # Deploy to default Pi (192.168.1.223)
#   ./scripts/deploy.sh pi@192.168.1.50        # Deploy to a different BirdNET-Pi
#   ./scripts/deploy.sh --model 3class.joblib  # Deploy a different model
#
# This copies:
#   src/insectnet/capture.py  → ~/insectnet_capture/insectnet_capture.py
#   src/insectnet/birdnet.py  → ~/insectnet_capture/birdnet.py
#   models/*.joblib           → ~/insectnet_capture/classifier.joblib

set -euo pipefail

PI_HOST="${1:-birdnetpi@192.168.1.223}"
MODEL_SRC="${2:-models/6class.joblib}"
CAPTURE_DIR="~/insectnet_capture"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying InsectNet to ${PI_HOST} ==="
echo "  Model: ${MODEL_SRC}"
echo "  Target: ${CAPTURE_DIR}"
echo ""

# Create remote dir
ssh "${PI_HOST}" "mkdir -p ${CAPTURE_DIR}"

# Deploy capture and birdnet modules
scp "${SCRIPT_DIR}/src/insectnet/capture.py" "${PI_HOST}:${CAPTURE_DIR}/insectnet_capture.py"
scp "${SCRIPT_DIR}/src/insectnet/birdnet.py" "${PI_HOST}:${CAPTURE_DIR}/birdnet.py"
echo "  ✓ capture.py deployed"

# Deploy classifier
scp "${SCRIPT_DIR}/${MODEL_SRC}" "${PI_HOST}:${CAPTURE_DIR}/classifier.joblib"
echo "  ✓ classifier deployed ($(basename ${MODEL_SRC}))"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Start the sidecar on the Pi:"
echo "  ssh ${PI_HOST}"
echo "  cd ${CAPTURE_DIR}"
echo "  python3 insectnet_capture.py --threshold 0.3 --show"
