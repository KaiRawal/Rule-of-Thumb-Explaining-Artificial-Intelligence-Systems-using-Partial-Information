#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TIMING_SOURCE="../JudicialCaseOutcomePrediction/TIMING.csv"
TIMING_LOCAL="./TIMING.csv"
RUNTIMES_CSV="./RunTimes.csv"
SCRIPT_A="./final_a.py"
SCRIPT_B="./final_b.py"
AUGMENT_UTIL="./augment_runtimes_csv.py"

stage() {
  echo "[runtimes] $1"
}

warn() {
  echo "[runtimes][warn] $1"
}

error() {
  echo "[runtimes][error] $1" >&2
}

require_file() {
  local path="$1"
  local message="$2"

  if [[ ! -f "$path" ]]; then
    error "$message"
    return 1
  fi

  return 0
}

augment_runtimes_csv() {
  local include_judicial_row="$1"

  require_file "$AUGMENT_UTIL" "Missing helper utility: $AUGMENT_UTIL" || return 1
  python3 "$AUGMENT_UTIL" --include-judicial-row "$include_judicial_row"
}

run_script() {
  local script_path="$1"
  stage "Running $(basename "$script_path")"
  python3 "$script_path"
}

main() {
  local timing_ready=0

  stage "Starting runtimes controller"

  require_file "$RUNTIMES_CSV" "Missing RunTimes.csv in Runtimes directory" || exit 1
  require_file "$SCRIPT_A" "Missing converted script: $SCRIPT_A" || exit 1
  require_file "$SCRIPT_B" "Missing converted script: $SCRIPT_B" || exit 1

  if [[ -f "$TIMING_SOURCE" ]]; then
    stage "Copying TIMING.csv from JudicialCaseOutcomePrediction"
    cp "$TIMING_SOURCE" "$TIMING_LOCAL"
    timing_ready=1
  else
    warn "TIMING source not found at $TIMING_SOURCE; skipping notebook-derived figures (a/b)"
  fi

  stage "Augmenting RunTimes.csv with judicial timing summary when available"
  augment_runtimes_csv "$timing_ready"

  if (( timing_ready == 1 )); then
    run_script "$SCRIPT_A"
    run_script "$SCRIPT_B"
  else
    warn "Stage a/b skipped because judicial TIMING source is unavailable"
  fi

  stage "Running main.py for figures c/d"
  python3 main.py

  stage "Completed runtimes controller"
}

main "$@"
