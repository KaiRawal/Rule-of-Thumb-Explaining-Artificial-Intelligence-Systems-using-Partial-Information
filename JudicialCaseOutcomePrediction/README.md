This experiment uses data and models from PredEx; details and citations can be found in the paper. The dataset file used is: (https://huggingface.co/datasets/L-NLProc/PredEx/blob/main/test.csv)[https://huggingface.co/datasets/L-NLProc/PredEx/blob/main/test.csv], and is included at [./DATA/PredEx/test_original.csv](./DATA/PredEx/test_original.csv). `oracle_selected_cases.txt` is a prefiltered list of cases where the model prediction is always correct, and which matches our quality filtration criteria.

Starting state:

```
IndianCases/
├── README.md
├── Code/
│   ├── 00_oracle_test_filter.py
│   ├── 01_filter_data.py
│   └── ...
└── DATA/
    ├── PREP/
    └── PredEx/
        ├── test_original.csv
        └── oracle_selected_cases.txt
```

Steps to reproduce figure 2:

1. Run python script: `$ python3 ./Code/00_oracle_test_filter.py`
2. Run python script: `$ python3 ./Code/01_filter_data.py`
3. Run python script: `$ python3 Code/02_create_subset.py`
4. Run python script: `$ python3 ./Code/03_gen_embeddings.py`
5. Run python script: `$ python3 ./Code/04_gen_rot_inputs.py`
6. Run python script: `$ python3 ./Code/05_merge_tokens_generic.py`
7. Run python script: `$ python3 ./Code/06_v05_run_RoT.py --train_set EMBEDDINGS_token_stride`
8. Run python script: `$ python3 ./Code/06_baseline_explainers.py --shap --nsamps 500`
9. Run python script: `$ python3 ./Code/06_baseline_explainers.py --lime --nsamps 500`
10. Run python script: `$ python3 ./Code/06_baseline_explainers.py --lime --nsamps 5000 # might be useful to uncomment the ids_to_test override to speed up processing by only explaining select cases (selected by ids from test set)`
11. Run python script: `$ python3 ./Code/06_baseline_explainers.py --ig`
12. Run python script: `$ python3 ./Code/random_baseline.py`
13. Run python script: `$ python3 ./Code/05_merge_tokens_generic.py --shap_samples 500`
14. Run python script: `$ python3 ./Code/05_merge_lime.py --lime_samples 500`
15. Run python script: `$ python3 ./Code/05_merge_lime.py --lime_samples 5000`
16. Run python script: `$ python3 ./Code/05_merge_tokens_generic.py --ig_dir IG`
17. Start a python server: `$ python3 -m http.server`
18. Navigate to http://localhost:8000/Code/results.html

The case used in Figure 2 is titled `377__GordonWoodroffeeLeatherManufacturingCoVsTheCommissionerOfIncomeTaxMadras`. (Tip: reload the page when changing explainer for accurate results.)

To reproduce table 1 you can now run `$ bash create_table.sh` to create the file `table1.csv`.


All results from the paper can also be viewed directly by navigating to https://kairawal.github.io/Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information/JudicialCaseOutcomePrediction/Code/results.html

