# Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information
Code to replicate experiments and graphics from the paper "Rule of Thumb: Explaining Artificial Intelligence Systems using Partial Information"

**anonymous.4open.science users: if you encounter a persistent "loading" page or a 404 error while navigating this repository, please reload the webpage or try a new incongito window.** The site has been tested on chrome, firefox, and safari.

The core implementation for the new Rule of Thumb (RoT) explainer is in `rot_class.py`, including a subclass designed specifically to operate on text embedding data. This is typically called through a wrapper `rule_of_thumb.py` to match the standard OpenXAI benchmark interface. Eventually, `rot_class.py` and `rule_of_thumb.py` can be generalised into a python package that can be imported and configured by researchers and practitioners, like SHAP or LIME. Each experiment uses these core files, with minor changes as applicable for the particular experiment.

Most experiments consist of a single `RUN.sh` file that is expected to run the entire experiment end to end, from downloading the data to producing the final plots. Some experiments were carried out via jupyter notebooks instead of orchestrated through shell scripts. The table below links to the relevant subdirectory for each experiment. Each subdirectory has a `README.md` file explaining the setup and providing basic instructions to replicate figures from the paper.


| SubDirectory | Description |
| ---------------- | ----------------------- |
| [ExplanationExampleLocal](./ExplanationExampleLocal/README.md) | **Fig. 1 – Explanation Disagreement Example using an Interactive Gradio Demo** (Pima Indians Diabetes). |
| [ExplanationExampleRemote](./ExplanationExampleRemote/README.md) | **Fig. 0 – Explanation Disagreement Example using GPT-4o-mini for Image Classification** (Cat/Dog). |
| [JudicialCaseOutcomePrediction](./JudicialCaseOutcomePrediction/README.md) | **Fig. 2 (+ Table 1) – Explaining Case Outcome Prediction** using a finetuned RoBERTa model. |
| [MovieReviewSentiments](./MovieReviewSentiments/README.md) | **Table 3 – Explaining Movie Review Sentiment Prediction** using LLM APIs. |
| [SyntheticResumeFiltering](./SyntheticResumeFiltering/README.md) | **Fig. 3 – Explaining Resume Filtering** using LLM APIs. |
| [AIAuditing](./AIAuditing/README.md) | **Fig. 4 (+ Table 2) – XAI for Auditing Proprietary AIs without making Model Inference Calls**. |
| [ScientificDiscovery](./ScientificDiscovery/README.md) | **Fig. 5 – XAI for Scientific Discovery**. |
| [AdversarialAttack](./AdversarialAttack/README.md) | **Fig. 6 – Invulnerability to Adversarial Attacks on Explanations**. |
| [OpenXAIBenchmark](./OpenXAIBenchmark/README.md) | **Fig. 7 – OpenXAI Benchmark** experiments. |
| [Runtimes](./Runtimes/README.md) | Scripts and results for **Fig. 9 – Runtime** analysis. |
| [LitReview](./LitReview/README.md) | Dataset (CSV files) examining papers which apply SHAP and XAI in scientific discovery. |


All experiments were run on a MacBook Pro with 24 GB memory using python 3.10.0.


## Setup

Create and activate a virtual environment from the repository root, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run all experiments (terminal-only)

Use the root-level orchestrator to run all experiments sequentially:

```bash
bash run_all_experiments.sh
```

Execution order:
1. `adversarial_attack`
2. `scientific_discovery`
3. `ai_auditing`
4. `explanation_example` (remote: `orchestrate.sh`, then local: `app.py --non-interactive`)
5. `openxai_benchmark`
6. `resume_filtering`
7. `movie_review_sentiments`
8. `judicial_case_outcome_prediction`
9. `runtimes`

Useful options:

```bash
# Show all step IDs
bash run_all_experiments.sh --list-steps

# Resume from a step index
bash run_all_experiments.sh --from-step 5

# Resume from a step ID
bash run_all_experiments.sh --from-step openxai_benchmark

# Stop at a specific step
bash run_all_experiments.sh --from-step 3 --to-step 7

# Stop immediately on first failure
bash run_all_experiments.sh --fail-fast

# Configure heartbeat and stall warning cadence
bash run_all_experiments.sh --heartbeat-seconds 60 --stall-warn-seconds 300
```

During execution, the runner now prints periodic `ALIVE` messages for the current step (default every 60 seconds), including pid, elapsed time, and log age/size. If a running step log does not grow for a while, it prints `WARN_STALL`.

### Monitor progress while a run is active

Use these commands in another terminal:

```bash
# Path to the most recent run directory
RUN_DIR="$(cat .experiment_logs/latest_run.txt)"
echo "$RUN_DIR"

# See which steps are currently running/completed/failed
tail -n 20 "$RUN_DIR/summary.tsv"

# Follow the newest step log in real time
tail -f "$(ls -1t "$RUN_DIR"/steps/*.log | head -n 1)"
```

If you want to inspect a specific step log directly:

```bash
tail -f .experiment_logs/runs/<timestamp>/steps/8_judicial_case_outcome_prediction.log
```

If a step fails, detailed artifacts are saved under `.experiment_logs/failures/<timestamp>/<step>/` including:
- command used
- working directory
- exit code
- full output log and short tail log
- runtime duration
- environment snapshot

Run summaries are saved under `.experiment_logs/runs/<timestamp>/summary.tsv`.

This runner is intentionally terminal-only and does not execute notebooks. Notebook-dependent post-processing/plots remain available in each experiment subdirectory.

