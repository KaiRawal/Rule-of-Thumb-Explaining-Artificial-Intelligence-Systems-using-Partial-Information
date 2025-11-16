#!/usr/bin/env bash


mkdir ./DATA/PREP

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
python3 ./Code/06_baseline_explainers.py --shap --nsamps 500

echo "Step 9/16: running baseline explainers: lime (nsamps=500)"
python3 ./Code/06_baseline_explainers.py --lime --nsamps 500

echo "Step 10/16: running baseline explainers: lime (nsamps=5000)"
# Note: you may want to uncomment ids_to_test override inside the script to speed up processing between step 9 and 10
python3 ./Code/06_baseline_explainers.py --lime --nsamps 5000

echo "Step 11/16: running baseline explainers: ig"
python3 ./Code/06_baseline_explainers.py --ig

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

echo "All steps completed, starting local server to serve results at http://localhost:8000"

python3 -m http.server
