

mkdir ./DATA/EMBEDDINGS
python3 ./Code/gen_token_embeddings.py --input_file ./DATA/movies/train.jsonl


mkdir ./DATA/CSVs
python3 ./Code/gen_bitstrings.py


mkdir ./DATA/MERGED_HUMAN
mkdir ./DATA/MERGED_GPT
python3 ./Code/merge_tokens.py --input_file ./DATA/CSVs/human_train.csv --output_dir ./DATA/MERGED_HUMAN
python3 ./Code/merge_tokens.py --input_file ./DATA/CSVs/gpt_train.csv --output_dir ./DATA/MERGED_GPT


python3 ./Code/display_preRoT_stats.py

mkdir ./DATA/CACHE
mkdir ./DATA/SPAN_EXPS
python3 ./Code/run_RoT.py --train_file ./DATA/CSVs/rot_input_gpt_train.csv --test_file ./DATA/CSVs/rot_input_human_train.csv --explanation_dir ./DATA/SPAN_EXPS --cache_dir ./DATA/CACHE --learning_rate 0.000001 --epochs 128


echo "Analyzing results to compute metrics: AUROC and PR-AUC:"
python3 ./Code/analyse_results.py --directory ./DATA/SPAN_EXPS --metadata ./DATA/CACHE/meta.csv --plots ./DATA/CACHE --mask 0

echo "Analyzing results to compute metrics: AUROC and PR-AUC with flipped labels:"
python3 ./Code/analyse_results.py --directory ./DATA/SPAN_EXPS --metadata ./DATA/CACHE/meta.csv --plots ./DATA/CACHE --mask 0 --flip

