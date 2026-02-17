#!/bin/bash
# Download V13 results from Lambda Labs and terminate instance
# Usage: bash lambda_download.sh [instance_ip] [seed]
#
# Example:
#   bash lambda_download.sh 132.145.213.116 42

set -e

IP="${1:-132.145.213.116}"
SEED="${2:-42}"
KEY="$HOME/.ssh/google_compute_engine"
REMOTE_DIR="/home/ubuntu/results/v13_s${SEED}"
LOCAL_DIR="$(dirname "$0")/results/v13_s${SEED}"

echo "=== Downloading V13 Results ==="
echo "Instance: ubuntu@${IP}"
echo "Remote: ${REMOTE_DIR}"
echo "Local: ${LOCAL_DIR}"
echo ""

# Check if run is complete
STATUS=$(ssh -i "$KEY" -o ConnectTimeout=10 "ubuntu@${IP}" \
    "python3 -c \"import json; d=json.load(open('${REMOTE_DIR}/v13_progress.json')); print(d.get('status','unknown'))\"" 2>/dev/null)

echo "Status: ${STATUS}"
if [ "$STATUS" != "complete" ]; then
    LAST=$(ssh -i "$KEY" "ubuntu@${IP}" "tail -1 ${REMOTE_DIR}/../v13_s${SEED}.log" 2>/dev/null)
    echo "Last log line: ${LAST}"
    echo ""
    read -p "Run not complete. Download anyway? [y/N] " yn
    if [ "$yn" != "y" ] && [ "$yn" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# Download
mkdir -p "$LOCAL_DIR"
echo ""
echo "Downloading..."
scp -r -i "$KEY" "ubuntu@${IP}:${REMOTE_DIR}/" "$LOCAL_DIR/"
echo "Downloaded to: ${LOCAL_DIR}"

# Show what we got
echo ""
echo "Contents:"
ls -lh "$LOCAL_DIR/"
echo ""
ls -lh "$LOCAL_DIR/snapshots/" 2>/dev/null || echo "(no snapshots dir)"

echo ""
echo "=== Generate Figures ==="
echo "  python v13_visualize.py ${LOCAL_DIR}"
echo ""
echo "=== IMPORTANT: Terminate Instance ==="
echo "  Remember to terminate the Lambda instance to save credits!"
echo "  curl -u \$LAMBDA_API_KEY: https://cloud.lambda.ai/api/v1/instance-operations/terminate -d '{\"instance_ids\": [\"INSTANCE_ID\"]}' -H 'Content-Type: application/json'"
