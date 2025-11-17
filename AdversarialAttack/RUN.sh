

git clone https://github.com/dylan-slack/Fooling-LIME-SHAP.git
cd Fooling-LIME-SHAP

mv ../rot_class.py rot_class.py
mv ../rule_of_thumb.py rule_of_thumb.py
mv ../overwrite/cc_experiment.py cc_experiment.py
mv ../overwrite/german_experiment.py german_experiment.py
mv ../overwrite/compas_experiment.py compas_experiment.py
mv ../overwrite/utils.py utils.py
mv ../final.ipynb final.ipynb

mkdir RESULTS
mkdir RESULTS/PLOTS
mkdir RESULTS/cc
mkdir RESULTS/german
mkdir RESULTS/compas

python3 compas_experiment.py 
python3 german_experiment.py 
python3 cc_experiment.py 

