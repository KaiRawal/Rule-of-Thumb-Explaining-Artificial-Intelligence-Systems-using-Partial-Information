# Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information
Code to replicate experiments and graphics from the paper "Rule of Thumb: Explaining Artificial Intelligence Systems using Partial Information"

The core implementation for the new Rule of Thumb (RoT) explainer is in `rot_class.py`, including a subclass designed specifically to operate on text embedding data. This is typically called through a wrapper `rule_of_thumb.py` to match the standard OpenXAI benchmark interface. Eventually, `rot_class.py` and `rule_of_thumb.py` can be generalised into a python package that can be imported and configured by researchers and practitioners, like SHAP or LIME. Each experiment uses these core files, with minor changes as applicable for the particular experiment.

Most experiments consist of a single `RUN.sh` file that is expected to run the entire experiment end to end, from downloading the data to producing the final plots. Some experiments were carried out via jupyter notebooks instead of orchestrated through shell scripts. The table below links to the relevant subdirectory for each experiment. Each subdirectory has a `README.md` file explaining the setup and providing basic instructions to replicate figures from the paper.


| SubDirectory | Description |
| ---------------- | ----------------------- |
| [ExplanationExample](./ExplanationExample) | **Fig. 1 – Explanation Disagreement Example using an Interactive Gradio Demo**. |
| [JudicialCaseOutcomePrediction](./JudicialCaseOutcomePrediction) | **Fig. 2 (+ Table 1) – Explaining Case Outcome Prediction** using a finetuned RoBERTa model. |
| [MovieReviewSentiments](./MovieReviewSentiments) | **Table 3 – Explaining Movie Review Sentiment Prediction** using LLM APIs. |
| [SyntheticResumeFiltering](./SyntheticResumeFiltering) | **Fig. 3 – Explaining Resume Filtering** using LLM APIs. |
| [AIAuditing](./AIAuditing) | **Fig. 4 (+ Table 2) – XAI for Auditing Proprietary AIs without making Model Inference Calls**. |
| [ScientificDiscovery](./ScientificDiscovery) | **Fig. 5 – XAI for Scientific Discovery**. |
| [AdversarialAttack](./AdversarialAttack) | **Fig. 6 – Invulnerability to Adversarial Attacks on Explanations**. |
| [OpenXAIBenchmark](./OpenXAIBenchmark) | **Fig. 7 – OpenXAI Benchmark** experiments. |
| [Runtimes](./Runtimes) | Scripts and results for **Fig. 9 – Runtime** analysis. |
| [LitReview](./LitReview) | Dataset (CSV files) examining papers which apply SHAP and XAI in scientific discovery. |

