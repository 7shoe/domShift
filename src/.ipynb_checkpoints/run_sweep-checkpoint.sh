#!/bin/bash
# run_sweep.sh
# This script creates a sweep using ssl_sweep_template.yaml and launches agents.
# Usage: ./run_sweep.sh

TEMPLATE_CONFIG="ssl_sweep_template.yaml"

SWEEP_ID=$(wandb sweep "$TEMPLATE_CONFIG" | grep -oP 'sweep with ID: \K.*')
if [ -z "$SWEEP_ID" ]; then
    echo "Error creating sweep."
    exit 1
fi

echo "Sweep created with ID: $SWEEP_ID"
echo "View sweep at: https://wandb.ai/7shoe/domShift-src/sweeps/$SWEEP_ID"