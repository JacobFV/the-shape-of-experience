#!/bin/bash
# Chain V13 runs across seeds on Lambda instance
# Each seed runs 30 cycles, results saved, then next seed starts
# Usage: bash lambda_chain_seeds.sh <ip> [start_seed]

set -e

IP="${1:?Usage: bash lambda_chain_seeds.sh <ip> [start_seed]}"
START_SEED="${2:-123}"
KEY="$HOME/.ssh/google_compute_engine"

SEEDS=($START_SEED 7 999)

for SEED in "${SEEDS[@]}"; do
    echo "=== Launching seed=$SEED on $IP ==="
    ssh -i "$KEY" "ubuntu@${IP}" "cd /home/ubuntu/experiment/empirical/experiments/study_ca_affect && \
        nohup python3 -u v13_gpu_run.py --cycles 30 --channels 16 --grid 128 --seed $SEED \
        --output /home/ubuntu/results/v13_s${SEED} > /home/ubuntu/results/v13_s${SEED}.log 2>&1 &
        echo PID: \$!"
    echo "Waiting for completion..."

    # Poll every 60s until done
    while true; do
        STATUS=$(ssh -i "$KEY" -o ConnectTimeout=10 "ubuntu@${IP}" \
            "python3 -c \"import json; d=json.load(open('/home/ubuntu/results/v13_s${SEED}/v13_progress.json')); print(d.get('status','unknown'))\"" 2>/dev/null || echo "not_started")

        if [ "$STATUS" = "complete" ]; then
            echo "Seed $SEED complete!"
            LAST=$(ssh -i "$KEY" "ubuntu@${IP}" "tail -1 /home/ubuntu/results/v13_s${SEED}.log")
            echo "  $LAST"
            break
        fi

        LAST=$(ssh -i "$KEY" "ubuntu@${IP}" "tail -1 /home/ubuntu/results/v13_s${SEED}.log" 2>/dev/null || echo "waiting...")
        echo "  $(date +%H:%M) | $LAST"
        sleep 60
    done
done

echo ""
echo "=== All seeds complete ==="
echo "Download with:"
for SEED in "${SEEDS[@]}"; do
    echo "  bash lambda_download.sh $IP $SEED"
done
