
echo "index, comment, roberta_correct_only, case_count, AUC_weighted_case" > table1.csv
python3 ./Code/analyze_results_CSV.py --formula flip --directory SPAN_EXPS_stride_35_128_1e-05 --comment 'Token-level RoT 35' --csv >> 'table1.csv'
python3 ./Code/analyze_results_CSV.py --formula flip --directory IG_MERGED --comment 'Integrated Gradients' --csv >> 'table1.csv'
python3 ./Code/analyze_results_CSV.py --formula flip --directory SHAP_MERGED_500 --comment 'Default 500 Sample SHAP' --csv >> 'table1.csv'
python3 ./Code/analyze_results_CSV.py --formula flip --directory LIME_MERGED_5000 --comment 'Default 5000 Sample LIME' --csv >> 'table1.csv'
python3 ./Code/analyze_results_CSV.py --formula flip --directory UNIFORM_spans --comment 'Uniform Random Importance per PredEx span' --csv >> 'table1.csv'

