#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TOTAL_STEPS=9
STEP_IDS=(
  "adversarial_attack"
  "scientific_discovery"
  "ai_auditing"
  "explanation_example"
  "openxai_benchmark"
  "resume_filtering"
  "movie_review_sentiments"
  "judicial_case_outcome_prediction"
  "runtimes"
)
STEP_NAMES=(
  "AdversarialAttack"
  "ScientificDiscovery"
  "AIAuditing"
  "ExplanationExample"
  "OpenXAIBenchmark"
  "SyntheticResumeFiltering"
  "MovieReviewSentiments"
  "JudicialCaseOutcomePrediction"
  "Runtimes"
)
STEP_DIRS=(
  "AdversarialAttack"
  "ScientificDiscovery"
  "AIAuditing"
  "ExplanationExample"
  "OpenXAIBenchmark"
  "SyntheticResumeFiltering"
  "MovieReviewSentiments"
  "JudicialCaseOutcomePrediction"
  "Runtimes"
)

FROM_STEP="1"
TO_STEP="$TOTAL_STEPS"
LOG_ROOT=".experiment_logs"
FAIL_FAST=0
LIST_STEPS=0
HEARTBEAT_SECONDS=60
STALL_WARN_SECONDS=300
MONITOR_POLL_SECONDS=5

print_help() {
  cat <<'EOF'
Usage: bash run_all_experiments.sh [options]

Options:
  --from-step <index|id>   Start from this step (default: 1)
  --to-step <index|id>     Stop after this step (default: 9)
  --log-dir <path>         Output directory for run logs/artifacts (default: .experiment_logs)
  --heartbeat-seconds <n>  Print step liveness heartbeat every n seconds (default: 60)
  --stall-warn-seconds <n> Warn when a running step log shows no growth for n seconds (default: 300)
  --fail-fast              Stop immediately on first failure
  --list-steps             Print step index + ids and exit
  -h, --help               Show this help message

Step IDs:
  1 adversarial_attack
  2 scientific_discovery
  3 ai_auditing
  4 explanation_example
  5 openxai_benchmark
  6 resume_filtering
  7 movie_review_sentiments
  8 judicial_case_outcome_prediction
  9 runtimes
EOF
}

list_steps() {
  local i
  for ((i=1; i<=TOTAL_STEPS; i++)); do
    echo "$i ${STEP_IDS[$((i-1))]} (${STEP_NAMES[$((i-1))]})"
  done
}

resolve_step_selector() {
  local selector="$1"
  local i

  if [[ "$selector" =~ ^[0-9]+$ ]]; then
    if (( selector < 1 || selector > TOTAL_STEPS )); then
      echo "Invalid step index: $selector" >&2
      return 1
    fi
    echo "$selector"
    return 0
  fi

  for ((i=1; i<=TOTAL_STEPS; i++)); do
    if [[ "${STEP_IDS[$((i-1))]}" == "$selector" ]]; then
      echo "$i"
      return 0
    fi
  done

  echo "Invalid step selector: $selector" >&2
  return 1
}

resolve_positive_int() {
  local name="$1"
  local value="$2"

  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "$name must be a positive integer: $value" >&2
    return 1
  fi

  echo "$value"
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-step)
      FROM_STEP="${2:-}"
      shift 2
      ;;
    --to-step)
      TO_STEP="${2:-}"
      shift 2
      ;;
    --log-dir)
      LOG_ROOT="${2:-}"
      shift 2
      ;;
    --heartbeat-seconds)
      HEARTBEAT_SECONDS="${2:-}"
      shift 2
      ;;
    --stall-warn-seconds)
      STALL_WARN_SECONDS="${2:-}"
      shift 2
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --list-steps)
      LIST_STEPS=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

if (( LIST_STEPS == 1 )); then
  list_steps
  exit 0
fi

if [[ -z "$FROM_STEP" || -z "$TO_STEP" ]]; then
  echo "--from-step and --to-step require a value" >&2
  exit 1
fi

HEARTBEAT_SECONDS="$(resolve_positive_int "--heartbeat-seconds" "$HEARTBEAT_SECONDS")" || exit 1
STALL_WARN_SECONDS="$(resolve_positive_int "--stall-warn-seconds" "$STALL_WARN_SECONDS")" || exit 1

if (( STALL_WARN_SECONDS < HEARTBEAT_SECONDS )); then
  STALL_WARN_SECONDS="$HEARTBEAT_SECONDS"
fi

FROM_INDEX="$(resolve_step_selector "$FROM_STEP")" || exit 1
TO_INDEX="$(resolve_step_selector "$TO_STEP")" || exit 1

if (( FROM_INDEX > TO_INDEX )); then
  echo "from-step must be <= to-step" >&2
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/runs/$RUN_TS"
STEP_LOG_DIR="$RUN_DIR/steps"
FAIL_DIR="$LOG_ROOT/failures/$RUN_TS"
CONSOLE_LOG_FILE="$RUN_DIR/console_output.log"
LATEST_CONSOLE_POINTER="$LOG_ROOT/latest_console_log.txt"

mkdir -p "$STEP_LOG_DIR" "$FAIL_DIR"

SUMMARY_FILE="$RUN_DIR/summary.tsv"
RUN_MANIFEST="$RUN_DIR/run_manifest.txt"
LATEST_POINTER="$LOG_ROOT/latest_run.txt"
FIRST_FAIL_FILE="$RUN_DIR/first_failure.txt"

# Mirror this script's terminal output to a run-level log while preserving live console output.
if [[ -z "${RUN_ALL_EXPERIMENTS_TEE_ACTIVE:-}" ]]; then
  export RUN_ALL_EXPERIMENTS_TEE_ACTIVE=1
  exec > >(tee -a "$CONSOLE_LOG_FILE")
  exec 2> >(tee -a "$CONSOLE_LOG_FILE" >&2)
fi

echo "$CONSOLE_LOG_FILE" > "$LATEST_CONSOLE_POINTER"

cat > "$RUN_MANIFEST" <<EOF
timestamp=$RUN_TS
repo_root=$SCRIPT_DIR
from_step_input=$FROM_STEP
to_step_input=$TO_STEP
from_step_index=$FROM_INDEX
to_step_index=$TO_INDEX
fail_fast=$FAIL_FAST
heartbeat_seconds=$HEARTBEAT_SECONDS
stall_warn_seconds=$STALL_WARN_SECONDS
console_log_file=$CONSOLE_LOG_FILE
latest_console_pointer=$LATEST_CONSOLE_POINTER
EOF

echo "$RUN_DIR" > "$LATEST_POINTER"

{
  echo -e "index\tstep_id\tstep_name\tstatus\texit_code\tduration_seconds\tlog_file"
} > "$SUMMARY_FILE"

STATUS_CODES=()
FAILED_COUNT=0
SKIPPED_COUNT=0
SUCCESS_COUNT=0
FIRST_FAIL_INDEX=0

get_file_size_bytes() {
  local path="$1"
  local size="0"

  if [[ -f "$path" ]]; then
    size="$(wc -c < "$path" 2>/dev/null | tr -d '[:space:]')"
    if [[ -z "$size" ]]; then
      size="0"
    fi
  fi

  echo "$size"
}

get_file_mtime_epoch() {
  local path="$1"

  if [[ ! -f "$path" ]]; then
    echo "0"
    return 0
  fi

  if stat -f %m "$path" >/dev/null 2>&1; then
    stat -f %m "$path"
    return 0
  fi

  if stat -c %Y "$path" >/dev/null 2>&1; then
    stat -c %Y "$path"
    return 0
  fi

  echo "0"
}

run_monitored_step() {
  local step_dir="$1"
  local step_log="$2"
  local command="$3"
  local idx="$4"
  local step_id="$5"
  local step_name="$6"
  local pid
  local start_ts
  local last_heartbeat_ts
  local last_growth_ts
  local last_stall_notice_ts
  local now
  local elapsed
  local size
  local previous_size
  local mtime_epoch
  local log_age
  local stall_elapsed

  (
    cd "$SCRIPT_DIR/$step_dir" || exit 2
    # Avoid login shell startup files so we inherit the caller's active Python env.
    bash -c "$command"
  ) >>"$step_log" 2>&1 &
  pid=$!

  start_ts="$(date +%s)"
  last_heartbeat_ts="$start_ts"
  last_growth_ts="$start_ts"
  last_stall_notice_ts=0
  previous_size="$(get_file_size_bytes "$step_log")"

  echo "[${idx}/${TOTAL_STEPS}] MONITOR ${step_name} pid=${pid} heartbeat=${HEARTBEAT_SECONDS}s stall_warn=${STALL_WARN_SECONDS}s"

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$MONITOR_POLL_SECONDS"
    now="$(date +%s)"

    if (( now - last_heartbeat_ts < HEARTBEAT_SECONDS )); then
      continue
    fi

    size="$(get_file_size_bytes "$step_log")"
    if (( size > previous_size )); then
      last_growth_ts="$now"
    fi
    previous_size="$size"

    mtime_epoch="$(get_file_mtime_epoch "$step_log")"
    if (( mtime_epoch > 0 )); then
      log_age=$((now - mtime_epoch))
      if (( log_age < 0 )); then
        log_age=0
      fi
    else
      log_age=-1
    fi

    elapsed=$((now - start_ts))
    stall_elapsed=$((now - last_growth_ts))

    echo "[${idx}/${TOTAL_STEPS}] ALIVE ${step_name} pid=${pid} elapsed=${elapsed}s log_size=${size}B log_age=${log_age}s"

    if (( stall_elapsed >= STALL_WARN_SECONDS )) && (( now - last_stall_notice_ts >= HEARTBEAT_SECONDS )); then
      echo "[${idx}/${TOTAL_STEPS}] WARN_STALL ${step_name} pid=${pid} no_log_growth=${stall_elapsed}s"
      last_stall_notice_ts="$now"
    fi

    last_heartbeat_ts="$now"
  done

  wait "$pid"
}

log_python_environment() {
  local step_log="$1"
  local py_path=""
  local py_exec=""

  {
    echo "[env] python_diagnostics_start"

    if command -v python3 >/dev/null 2>&1; then
      py_path="$(command -v python3)"
      echo "[env] python3_path=$py_path"
      echo "[env] python3_version=$(python3 --version 2>&1)"

      py_exec="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
      if [[ -n "$py_exec" ]]; then
        echo "[env] sys_executable=$py_exec"
      else
        echo "[env] sys_executable=<unavailable>"
      fi
    else
      echo "[env] python3_path=<not found>"
      echo "[env] python3_version=<not found>"
      echo "[env] sys_executable=<not found>"
    fi

    echo "[env] python_diagnostics_end"
  } >>"$step_log" 2>&1
}

run_explanation_example_step() {
  local step_log="$1"
  local pid

  (
    cd "$SCRIPT_DIR/ExplanationExample" || exit 2
    python3 app.py --non-interactive
  ) >>"$step_log" 2>&1 &
  pid=$!

  if kill -0 "$pid" 2>/dev/null; then
    echo "[info] explanation_example: Gradio process started with pid=$pid; stopping to keep run non-interactive." >>"$step_log"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    return 0
  fi

  wait "$pid"
  return $?
}

capture_failure_artifacts() {
  local idx="$1"
  local step_id="$2"
  local step_name="$3"
  local step_dir="$4"
  local step_command="$5"
  local step_log="$6"
  local exit_code="$7"
  local duration="$8"

  local target_dir="$FAIL_DIR/${idx}_${step_id}"
  mkdir -p "$target_dir"

  printf '%s\n' "$step_command" > "$target_dir/command.txt"
  printf '%s\n' "$SCRIPT_DIR/$step_dir" > "$target_dir/working_directory.txt"
  printf '%s\n' "$step_name" > "$target_dir/step_name.txt"
  printf '%s\n' "$step_id" > "$target_dir/step_id.txt"
  printf '%s\n' "$exit_code" > "$target_dir/exit_code.txt"
  printf '%s\n' "$duration" > "$target_dir/duration_seconds.txt"

  cp "$step_log" "$target_dir/output.log"
  tail -n 80 "$step_log" > "$target_dir/tail.log"

  {
    date
    uname -a
    command -v python3 >/dev/null 2>&1 && python3 --version
    command -v pip >/dev/null 2>&1 && pip --version
  } > "$target_dir/env_snapshot.txt" 2>&1
}

record_summary() {
  local idx="$1"
  local step_id="$2"
  local step_name="$3"
  local status="$4"
  local exit_code="$5"
  local duration="$6"
  local step_log="$7"

  echo -e "${idx}\t${step_id}\t${step_name}\t${status}\t${exit_code}\t${duration}\t${step_log}" >> "$SUMMARY_FILE"
}

echo "Running steps ${FROM_INDEX}..${TO_INDEX}"
echo "Heartbeat: every ${HEARTBEAT_SECONDS}s (stall warning after ${STALL_WARN_SECONDS}s without log growth)"

for ((idx=1; idx<=TOTAL_STEPS; idx++)); do
  step_id="${STEP_IDS[$((idx-1))]}"
  step_name="${STEP_NAMES[$((idx-1))]}"
  step_dir="${STEP_DIRS[$((idx-1))]}"
  step_log="$STEP_LOG_DIR/${idx}_${step_id}.log"

  if (( idx < FROM_INDEX || idx > TO_INDEX )); then
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    STATUS_CODES+=("SKIPPED")
    record_summary "$idx" "$step_id" "$step_name" "SKIPPED" "0" "0" "$step_log"
    continue
  fi

  start_time="$(date +%s)"
  exit_code=0

  echo "[${idx}/${TOTAL_STEPS}] START ${step_name} (${step_id})"
  echo "=== START $(date) ===" > "$step_log"
  log_python_environment "$step_log"

  case "$step_id" in
    adversarial_attack)
      step_cmd="bash RUN.sh"
      run_monitored_step "$step_dir" "$step_log" "$step_cmd" "$idx" "$step_id" "$step_name"
      exit_code=$?
      ;;
    scientific_discovery)
      step_cmd="bash RUN.sh"
      run_monitored_step "$step_dir" "$step_log" "$step_cmd" "$idx" "$step_id" "$step_name"
      exit_code=$?
      ;;
    ai_auditing)
      step_cmd="bash RUN.sh"
      run_monitored_step "$step_dir" "$step_log" "$step_cmd" "$idx" "$step_id" "$step_name"
      exit_code=$?
      ;;
    explanation_example)
      step_cmd="python3 app.py --non-interactive"
      run_explanation_example_step "$step_log"
      exit_code=$?
      ;;
    openxai_benchmark)
      step_cmd="bash RUN.sh"
      run_monitored_step "$step_dir" "$step_log" "$step_cmd" "$idx" "$step_id" "$step_name"
      exit_code=$?
      ;;
    resume_filtering)
      step_cmd="bash RUN.sh"
      run_monitored_step "$step_dir" "$step_log" "$step_cmd" "$idx" "$step_id" "$step_name"
      exit_code=$?
      ;;
    movie_review_sentiments)
      step_cmd="bash RUN.sh"
      run_monitored_step "$step_dir" "$step_log" "$step_cmd" "$idx" "$step_id" "$step_name"
      exit_code=$?
      ;;
    judicial_case_outcome_prediction)
      step_cmd="bash RUN.sh --non-interactive"
      run_monitored_step "$step_dir" "$step_log" "$step_cmd" "$idx" "$step_id" "$step_name"
      exit_code=$?
      ;;
    runtimes)
      step_cmd="python3 main.py"
      run_monitored_step "$step_dir" "$step_log" "$step_cmd" "$idx" "$step_id" "$step_name"
      exit_code=$?
      ;;
    *)
      step_cmd="unsupported"
      echo "Unknown step id: $step_id" >> "$step_log"
      exit_code=99
      ;;
  esac

  end_time="$(date +%s)"
  duration=$((end_time - start_time))
  echo "=== END $(date) (exit=${exit_code} duration=${duration}s) ===" >> "$step_log"

  if (( exit_code == 0 )); then
    echo "[${idx}/${TOTAL_STEPS}] OK ${step_name} (${duration}s)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    STATUS_CODES+=("OK")
    record_summary "$idx" "$step_id" "$step_name" "OK" "$exit_code" "$duration" "$step_log"
  else
    echo "[${idx}/${TOTAL_STEPS}] FAILED ${step_name} exit=${exit_code} (${duration}s)"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    STATUS_CODES+=("FAILED")
    record_summary "$idx" "$step_id" "$step_name" "FAILED" "$exit_code" "$duration" "$step_log"
    capture_failure_artifacts "$idx" "$step_id" "$step_name" "$step_dir" "$step_cmd" "$step_log" "$exit_code" "$duration"

    if (( FIRST_FAIL_INDEX == 0 )); then
      FIRST_FAIL_INDEX="$idx"
      {
        echo "index=$idx"
        echo "id=$step_id"
        echo "name=$step_name"
      } > "$FIRST_FAIL_FILE"
    fi

    if (( FAIL_FAST == 1 )); then
      echo "Fail-fast enabled, stopping after first failure."
      break
    fi
  fi
done

echo ""
echo "Run complete."
echo "Console output log: $CONSOLE_LOG_FILE"
echo "Summary file: $SUMMARY_FILE"
echo "Run manifest: $RUN_MANIFEST"
echo "Failure artifacts root: $FAIL_DIR"
echo ""
echo "Counts: OK=$SUCCESS_COUNT FAILED=$FAILED_COUNT SKIPPED=$SKIPPED_COUNT"

if (( FAILED_COUNT > 0 )); then
  fail_id="${STEP_IDS[$((FIRST_FAIL_INDEX-1))]}"
  echo "First failure: step $FIRST_FAIL_INDEX ($fail_id)"
  echo "Resume example: bash run_all_experiments.sh --from-step $FIRST_FAIL_INDEX"
  echo "Resume example: bash run_all_experiments.sh --from-step $fail_id"
  exit 1
fi

exit 0
