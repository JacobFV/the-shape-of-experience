#!/bin/bash
# Setup script for Lambda Labs GPU instances
# Run: bash lambda_setup.sh

set -e

echo "=== Lambda Labs V13 Setup ==="

# Install dependencies
pip install -q jax[cuda12] numpy scipy matplotlib

# Clone repo (or update if exists)
if [ -d /root/experiment ]; then
    cd /root/experiment
    git pull
else
    git clone https://github.com/JacobFV/the-shape-of-experience.git /root/experiment
fi

cd /root/experiment/empirical/experiments/study_ca_affect

# Verify GPU
python -c "
import jax
print(f'JAX devices: {jax.devices()}')
print(f'GPU available: {len(jax.devices(\"gpu\")) > 0}')
"

echo ""
echo "=== Setup complete ==="
echo "Run experiments with:"
echo "  cd /root/experiment/empirical/experiments/study_ca_affect"
echo "  python v13_run.py smoke"
echo "  python v13_run.py evolve 30 --channels 16 --grid 128"
echo "  python v13_run.py pipeline --channels 16 --grid 128"
