mkdir ./Dataset
cp LLMResumeBiasAnalysis/results/gpt/summaries/classification/classification.csv ./Dataset/classification.csv

mkdir ./DATA
# mkdir ./DATA/distplots
mkdir ./DATA/EMBEDDINGS
mkdir ./DATA/TOKEN_EXPS
mkdir ./PLOTS/
mkdir ./PLOTS/dists
mkdir ./PLOTS/distplots
mkdir ./PLOTS/clouds
mkdir ./PLOTS/hists


python3 ./Code/prep_RoT_inputs.py
python3 ./Code/run_RoT.py
python3 ./Code/viz.py
python3 ./Code/word_clouds.py

