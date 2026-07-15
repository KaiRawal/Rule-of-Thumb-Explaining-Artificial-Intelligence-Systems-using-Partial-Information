This subdirectory contains code to compute RoT explanations for image data for the example figure zero.

`orchestrate.sh` runs the full pipeline:
1. Downloads the Kaggle cat/dog classification dataset
2. Runs `classify_openai.py` — classifies images via GPT-4o-mini (optional)
3. Runs `run.py` — trains RoT using MobileNet-v3 as backbone, generates saliency-overlay visualisations
4. Copies 6 canonincal example saliency PDFs to `ExampleImages/`
