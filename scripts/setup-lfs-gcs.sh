#!/usr/bin/env bash
# Setup Git LFS to pull from GCS backend via lfs-dal
#
# Prerequisites:
#   brew install regen100/lfs-dal/lfs-dal
#
# This script configures your local git to use lfs-dal as the LFS transfer agent,
# pointing at gs://shape-of-experience-data/lfs/ for object storage.
#
# You need a .lfsdalconfig file in the repo root with GCS credentials.
# Ask Jacob for the service account key, then run:
#   base64 -i path/to/key.json | tr -d '\n'
# and put it in .lfsdalconfig as:
#   [lfs-dal]
#       scheme = gcs
#       bucket = shape-of-experience-data
#       root = lfs
#       credential = <base64-encoded-key>

set -euo pipefail

# Check lfs-dal is installed
if ! command -v lfs-dal &>/dev/null; then
    echo "Error: lfs-dal not found. Install with:"
    echo "  brew tap regen100/lfs-dal && brew install lfs-dal"
    exit 1
fi

# Check .lfsdalconfig exists
if [ ! -f .lfsdalconfig ]; then
    echo "Error: .lfsdalconfig not found in repo root."
    echo "Create it with GCS credentials. See this script's header for instructions."
    exit 1
fi

# Configure local git for lfs-dal
git config --local lfs.standalonetransferagent lfs-dal
git config --local lfs.customtransfer.lfs-dal.path lfs-dal
git config --local lfs.customtransfer.lfs-dal.concurrent true
git config --local lfs.customtransfer.lfs-dal.concurrenttransfers 4

echo "LFS-DAL configured. You can now run 'git lfs pull' to fetch LFS objects from GCS."
