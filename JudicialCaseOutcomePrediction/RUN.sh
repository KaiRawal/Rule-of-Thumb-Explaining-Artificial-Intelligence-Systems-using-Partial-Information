#!/usr/bin/env bash


# Ensure relative paths are resolved from this script's directory.
cd "$(dirname "$0")"

serve_results=1

while [[ $# -gt 0 ]]; do
	case "$1" in
		--non-interactive)
			serve_results=0
			shift
			;;
		--serve)
			serve_results=1
			shift
			;;
		-h|--help)
			echo "Usage: bash RUN.sh [--non-interactive|--serve]"
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			echo "Usage: bash RUN.sh [--non-interactive|--serve]" >&2
			exit 1
			;;
	esac
done

run_time_uid="$(date +%s)-$$"
timing_file="./TIMING.csv"
timing_backup_dir="./DATA/TIMING_BACKUPS/${run_time_uid}"

backup_timing_after() {
	if [ -d "$timing_backup_dir" ] && [ -f "$timing_file" ]; then
		cp "$timing_file" "$timing_backup_dir/TIMING_after.csv" || true
	fi
}

trap backup_timing_after EXIT

mkdir -p ./DATA/PREP
mkdir -p "$timing_backup_dir"

touch "$timing_file"
cp "$timing_file" "$timing_backup_dir/TIMING_before.csv"
: > "$timing_file"

echo "TIMING backup run uid: $run_time_uid"
echo "Saved pre-run timing backup: $timing_backup_dir/TIMING_before.csv"

echo "Step 1/16: running 00_oracle_test_filter.py"
python3 ./Code/00_oracle_test_filter.py

echo "Step 2/16: running 01_filter_data.py"
python3 ./Code/01_filter_data.py

echo "Step 3/16: running 02_create_subset.py"
python3 ./Code/02_create_subset.py

echo "Step 4/16: running 03_gen_embeddings.py"
python3 ./Code/03_gen_embeddings.py

echo "Step 5/16: running 04_gen_rot_inputs.py"
python3 ./Code/04_gen_rot_inputs.py

echo "Step 6/16: running 05_merge_tokens_generic.py"
python3 ./Code/05_merge_tokens_generic.py

echo "Step 7/16: running 06_v05_run_RoT.py with train_set EMBEDDINGS_token_stride"
python3 ./Code/06_v05_run_RoT.py --train_set EMBEDDINGS_token_stride

echo "Step 8/16: running baseline explainers: shap (nsamps=500)"
python3 ./Code/06_baseline_explainers.py --shap --nsamps 500 --overwrite

echo "Step 9/16: running baseline explainers: lime (nsamps=500) [skipped for feasibility, uncomment ids_to_test override in the script to run on a smaller subset of examples]"
# python3 ./Code/06_baseline_explainers.py --lime --nsamps 500 --overwrite

echo "Step 10/16: running baseline explainers: lime (nsamps=5000) [skipped for feasibility, uncomment ids_to_test override in the script to run on a smaller subset of examples]"
# Note: you may want to uncomment ids_to_test override inside the script to speed up processing between step 9 and 10
# python3 ./Code/06_baseline_explainers.py --lime --nsamps 5000 --overwrite

echo "Step 11/16: running baseline explainers: ig"
python3 ./Code/06_baseline_explainers.py --ig --overwrite

echo "Step 12/16: running random_baseline.py"
python3 ./Code/random_baseline.py

echo "Step 13/16: running 05_merge_tokens_generic.py with shap_samples=500"
python3 ./Code/05_merge_tokens_generic.py --shap_samples 500

echo "Step 14/16: running 05_merge_lime.py with lime_samples=500"
python3 ./Code/05_merge_lime.py --lime_samples 500

echo "Step 15/16: running 05_merge_lime.py with lime_samples=5000"
python3 ./Code/05_merge_lime.py --lime_samples 5000

echo "Step 16/16: running 05_merge_tokens_generic.py with ig_dir IG"
python3 ./Code/05_merge_tokens_generic.py --ig_dir IG

if [ "$serve_results" -eq 1 ]; then
	echo "All steps completed, starting local server to serve results at http://localhost:8000"
	python3 -m http.server
else
	echo "All steps completed, non-interactive mode enabled, skipping local server"
fi
