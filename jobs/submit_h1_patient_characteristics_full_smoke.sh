#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
WORKER_SCRIPT="$REPO/jobs/smoke_h1_patient_characteristics_full.sh"
LOG_DIR="$REPO/grid_logs/h1_patient_characteristics_full/smoke"

command -v grid_run >/dev/null 2>&1 || { echo "grid_run not found" >&2; exit 1; }
test -x "$PYTHON" || { echo "Python not found: $PYTHON" >&2; exit 1; }
test -x "$WORKER_SCRIPT" || { echo "Worker is not executable: $WORKER_SCRIPT" >&2; exit 1; }
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export REPO PYTHON
cd "$LOG_DIR"
grid_run --grid_submit=batch --grid_ncpus=1 --grid_mem=8G "$WORKER_SCRIPT"
echo "Smoke job submitted. Monitor with qstat."
