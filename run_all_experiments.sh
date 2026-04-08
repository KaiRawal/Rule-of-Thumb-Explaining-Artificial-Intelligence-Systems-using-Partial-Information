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

print_help() {
  cat <<'EOF'
Usage: bash run_all_experiments.sh [options]

Options:
  --from-step <index|id>   Start from this step (default: 1)
  --to-step <index|id>     Stop after this step (default: 9)
  --log-dir <path>         Output directory for run logs/artifacts (default: .experiment_logs)
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

mkdir -p "$STEP_LOG_DIR" "$FAIL_DIR"

SUMMARY_FILE="$RUN_DIR/summary.tsv"
RUN_MANIFEST="$RUN_DIR/run_manifest.txt"
LATEST_POINTER="$LOG_ROOT/latest_run.txt"
FIRST_FAIL_FILE="$RUN_DIR/first_failure.txt"

cat > "$RUN_MANIFEST" <<EOF
timestamp=$RUN_TS
repo_root=$SCRIPT_DIR
from_step_input=$FROM_STEP
to_step_input=$TO_STEP
from_step_index=$FROM_INDEX
to_step_index=$TO_INDEX
fail_fast=$FAIL_FAST
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

run_regular_step() {
  local step_dir="$1"
  local step_log="$2"
  local command="$3"

  (
    cd "$SCRIPT_DIR/$step_dir" || exit 2
    bash -lc "$command"
  ) >>"$step_log" 2>&1
}

run_explanation_example_step() {
  local step_log="$1"
  local pid

  (
    cd "$SCRIPT_DIR/ExplanationExample" || exit 2
    python3 app.py
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

run_judicial_step() {
  local step_log="$1"

  (
    cd "$SCRIPT_DIR/JudicialCaseOutcomePrediction" || exit 2
    awk '!/python3 -m http.server/' RUN.sh | bash
  ) >>"$step_log" 2>&1
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

  case "$step_id" in
    adversarial_attack)
      step_cmd="bash RUN.sh"
      run_regular_step "$step_dir" "$step_log" "$step_cmd"
      exit_code=$?
      ;;
    scientific_discovery)
      step_cmd="bash RUN.sh"
      run_regular_step "$step_dir" "$step_log" "$step_cmd"
      exit_code=$?
      ;;
    ai_auditing)
      step_cmd="bash RUN.sh"
      run_regular_step "$step_dir" "$step_log" "$step_cmd"
      exit_code=$?
      ;;
    explanation_example)
      step_cmd="python3 app.py (launched then terminated intentionally for terminal-only run)"
      run_explanation_example_step "$step_log"
      exit_code=$?
      ;;
    openxai_benchmark)
      step_cmd="bash RUN.sh"
      run_regular_step "$step_dir" "$step_log" "$step_cmd"
      exit_code=$?
      ;;
    resume_filtering)
      step_cmd="bash RUN.sh"
      run_regular_step "$step_dir" "$step_log" "$step_cmd"
      exit_code=$?
      ;;
    movie_review_sentiments)
      step_cmd="bash RUN.sh"
      run_regular_step "$step_dir" "$step_log" "$step_cmd"
      exit_code=$?
      ;;
    judicial_case_outcome_prediction)
      step_cmd="awk '!/python3 -m http.server/' RUN.sh | bash"
      run_judicial_step "$step_log"
      exit_code=$?
      ;;
    runtimes)
      step_cmd="python3 main.py"
      run_regular_step "$step_dir" "$step_log" "$step_cmd"
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
