

We start by downloading movie review data from [https://www.eraserbenchmark.com/zipped/movies.tar.gz](https://www.eraserbenchmark.com/zipped/movies.tar.gz).

`generate_gpt_predictions.py` was used to get the initial LLM outputs. These are already included here in `./DATA/gpt/`, so our repo starts out looking like:

```
ReviewSentimentPrediction/
├── README.md
├── Code/
│   ├── generate_gpt_predictions.py
│   ├── analyse_results.py
│   └── ...
└── DATA/
    ├── gpt/
    │   ├── test_predictions.jsonl
    │   ├── train_predictions.jsonl
    │   └── val_predictions.jsonl
    └── movies/
        └── ...
```

Running commands individually from `RUN.sh` or all together (`$ bash RUN.sh`) produces the AUC-ROC and the PR-AUC metrics from the main text and table 3, while populating the `DATA` subdirectory above similar to the experiment with judicial cases.