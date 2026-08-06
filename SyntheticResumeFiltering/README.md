


Start by cloning and running code from [https://github.com/akshajkumarv/LLMResumeBiasAnalysis](https://github.com/akshajkumarv/LLMResumeBiasAnalysis) to get the LLM API predictions we wish to explain with Rule of Thumb. We use a csv of resume summaries from the categories: `information-technology`, `accountant`, `aviation`, `construction`, `chef`, `advocate`, `teacher`, and `sales`; filtered using `GPT-4.1-nano` to select only IT workers. We provide these results already in [./Dataset/classification.csv](./Dataset/classification.csv). We then explain the behaviour of the zero shot LLM classifier without making additional API calls.

Starting repo structure:

```
SyntheticResumeFiltering/
├── README.md
├── RUN.sh
├── Code/
│   ├── prep_RoT_inputs.py
│   ├── rot_class.py
│   ├── rule_of_thumb.py
│   ├── run_RoT.py
│   ├── viz.py
│   └── word_clouds.py
└── LLMAnalysis/
    └── ...
```

`RUN.sh` provides end-to-end code to generate figures 3 and 14: `$ bash RUN.sh`.

## Visualising per-resume (token-level) importance

`./Code/new_token_viz.html` renders the Rule of Thumb token-level importance for **every** resume in `DATA/TOKEN_EXPS/` in one view (one section per resume), colouring each token by its importance (green = positive contribution, red = negative; color intensity scales with `|scaled_importance|`). It also shows each resume's `LLM Prediction · Human Label · RoT Prediction` pulled from `DATA/meta.csv`.

To view it:

1. Make sure the required data is present (produced by `RUN.sh`):
   - `DATA/TOKEN_EXPS/` (one `*.csv` per resume)
   - `DATA/meta.csv`

2. Run the server **from inside the `SyntheticResumeFiltering/` directory** (a local server is required so the page can read the `DATA/TOKEN_EXPS/` directory listing and fetch `meta.csv`; it will not work from `file://`):
   ```
   cd SyntheticResumeFiltering
   python3 -m http.server
   ```

3. Open this URL in the browser (do **not** double-click the `.html` file):
   ```
   http://localhost:8000/Code/new_token_viz.html
   ```

Tips:
- If the page shows an error, first stop any other `http.server` already running on port 8000 (it will not reuse an occupied port), restart, and **reload** the page. The page's error message now reports the exact URL/protocol it tried.
- Reload the page if the colors do not render accurately.

